# SyN (Symmetric Normalization) diffeomorphic registration implementation
#
# Based on torchreg: https://github.com/codingfisch/torchreg
#
# SyN uses a velocity field parameterization with scaling-and-squaring for
# guaranteed diffeomorphic (smooth, invertible) transformations.

using NNlib
using Zygote: ignore_derivatives

# ============================================================================
# Grid Cache for Spatial Transforms
# ============================================================================

"""
Cache for identity grids to avoid reallocation.
"""
const GRID_CACHE = Dict{Tuple{Tuple{Vararg{Int}}, DataType}, Array}()

"""
    get_identity_grid(spatial_size::NTuple{3, Int}, ::Type{T}) where T

Get or create a cached identity grid for the given spatial size.
Grid values are in normalized [-1, 1] coordinates with shape (3, X, Y, Z).
"""
function get_identity_grid(spatial_size::NTuple{3, Int}, ::Type{T}) where T
    key = (spatial_size, T)
    if !haskey(GRID_CACHE, key)
        GRID_CACHE[key] = create_identity_grid(spatial_size, T)
    end
    return GRID_CACHE[key]
end

"""
    clear_grid_cache!()

Clear the identity grid cache. Useful for memory management.
"""
function clear_grid_cache!()
    empty!(GRID_CACHE)
    return nothing
end

# ============================================================================
# Core SyN Transform Functions
# ============================================================================

"""
    spatial_transform(x::AbstractArray{T, 5}, v::AbstractArray{T, 5}; id_grid=nothing) where T

Warp image `x` using velocity/displacement field `v`.

# Arguments
- `x`: Image to warp, shape `(X, Y, Z, C, N)` - Julia convention
- `v`: Velocity/displacement field, shape `(X, Y, Z, 3, N)` where the 4th dimension
       contains (v_x, v_y, v_z) displacement components in normalized [-1, 1] coordinates
- `id_grid`: Optional pre-computed identity grid. If not provided, creates/retrieves from cache.

# Returns
- Warped image of same shape as `x`

# Notes
- Uses NNlib.grid_sample for bilinear interpolation
- Grid coordinates are: new_pos = id_grid + v
- The velocity field `v` should be in normalized coordinates (same scale as grid)
- Uses `:reflection` padding mode to match torchreg behavior

# Implementation Details
The spatial transform applies a displacement field to warp an image:
1. Create identity grid: positions (x, y, z) in normalized [-1, 1] range
2. Add displacement: new_pos = (x + v_x, y + v_y, z + v_z)
3. Sample input image at new positions using bilinear interpolation
"""
function spatial_transform(
    x::AbstractArray{T, 5},
    v::AbstractArray{T, 5};
    id_grid::Union{Nothing, AbstractArray{T}} = nothing
) where T
    X, Y, Z, C_v, N = size(v)
    @assert C_v == 3 "Velocity field must have 3 components, got $C_v"
    @assert size(x)[1:3] == (X, Y, Z) "Spatial dimensions must match: x=$(size(x)[1:3]), v=($X, $Y, $Z)"
    @assert size(x, 5) == N "Batch sizes must match: x=$(size(x, 5)), v=$N"

    # Get identity grid: (3, X, Y, Z)
    if id_grid === nothing
        id_grid = ignore_derivatives() do
            get_identity_grid((X, Y, Z), T)
        end
    end

    # Permute v from (X, Y, Z, 3, N) to (3, X, Y, Z, N) for grid construction
    # v[:, :, :, c, n] contains component c for batch n
    v_perm = permutedims(v, (4, 1, 2, 3, 5))  # (3, X, Y, Z, N)

    # Add displacement to identity grid to get sampling positions
    # id_grid: (3, X, Y, Z) -> expand to (3, X, Y, Z, N)
    # grid = id_grid + v
    grid = ignore_derivatives() do
        # Create expanded identity grid for batch
        repeat(reshape(id_grid, 3, X, Y, Z, 1), 1, 1, 1, 1, N)
    end
    grid = grid .+ v_perm  # (3, X, Y, Z, N)

    # NNlib.grid_sample expects grid of shape (3, X, Y, Z, N)
    # and input of shape (X, Y, Z, C, N)
    # Note: NNlib doesn't have :reflection mode, use :border instead
    result = NNlib.grid_sample(x, grid; padding_mode=:border)

    return result
end

"""
    diffeomorphic_transform(v::AbstractArray{T, 5}; time_steps::Int=7, id_grid=nothing) where T

Convert a stationary velocity field to a diffeomorphic displacement field using
scaling and squaring.

# Arguments
- `v`: Stationary velocity field, shape `(X, Y, Z, 3, N)`
- `time_steps`: Number of squaring iterations (default 7). Higher = more accurate.
- `id_grid`: Optional pre-computed identity grid.

# Returns
- Displacement field of same shape as `v`

# Algorithm: Scaling and Squaring
The scaling and squaring method computes exp(v) efficiently:

1. Scale: v_scaled = v / 2^N  (where N = time_steps)
2. For small v_scaled: exp(v_scaled) ≈ v_scaled (first-order approximation)
3. Square N times: exp(v) = exp(v_scaled)^(2^N) = (((v_scaled)²)²...)²

Each squaring step composes the field with itself:
- v_new = v_old + spatial_transform(v_old, v_old)
- This is equivalent to: φ_{2t} = φ_t ∘ φ_t

# Mathematical Background
For a stationary velocity field v, the diffeomorphism φ = exp(v) is the solution
to the ODE: dφ/dt = v(φ), with φ(0) = Id.

Scaling and squaring exploits: exp(v) = exp(v/2^N)^(2^N)

The exponential of a small field (v/2^N) can be approximated by v/2^N itself,
then repeated composition reconstructs the full exponential.

# Notes
- time_steps=7 (default) means 2^7 = 128 subdivisions, usually sufficient
- Larger time_steps increases accuracy but also computation time
- The result is guaranteed to be diffeomorphic (smooth, invertible) if v is smooth
"""
function diffeomorphic_transform(
    v::AbstractArray{T, 5};
    time_steps::Int = 7,
    id_grid::Union{Nothing, AbstractArray{T}} = nothing
) where T
    X, Y, Z, C, N = size(v)
    @assert C == 3 "Velocity field must have 3 components, got $C"

    # Get or create identity grid
    if id_grid === nothing
        id_grid = ignore_derivatives() do
            get_identity_grid((X, Y, Z), T)
        end
    end

    # Scale down velocity field: v = v / 2^time_steps
    scale_factor = T(2^time_steps)
    v_scaled = v ./ scale_factor

    # Squaring loop: compose field with itself time_steps times
    # Each iteration: v = v + spatial_transform(v, v)
    # This computes: exp(v/2^N)^(2^N) = exp(v)
    for _ in 1:time_steps
        v_warped = spatial_transform(v_scaled, v_scaled; id_grid=id_grid)
        v_scaled = v_scaled .+ v_warped
    end

    return v_scaled
end

"""
    composition_transform(v1::AbstractArray{T, 5}, v2::AbstractArray{T, 5}; id_grid=nothing) where T

Compose two velocity/displacement fields.

# Arguments
- `v1`: First velocity field, shape `(X, Y, Z, 3, N)`
- `v2`: Second velocity field, shape `(X, Y, Z, 3, N)`
- `id_grid`: Optional pre-computed identity grid.

# Returns
- Composed velocity field of same shape

# Formula
The composition of two displacement fields φ₁ and φ₂ is:
    (φ₂ ∘ φ₁)(x) = φ₂(x) + φ₁(φ₂(x))

In terms of displacement fields:
    v_composed = v2 + spatial_transform(v1, v2)

This means: first apply v2 to get new positions, then sample v1 at those positions
and add to v2.

# Notes
- Order matters: composition_transform(v1, v2) ≠ composition_transform(v2, v1)
- Following torchreg convention: v2 + v1(v2)
"""
function composition_transform(
    v1::AbstractArray{T, 5},
    v2::AbstractArray{T, 5};
    id_grid::Union{Nothing, AbstractArray{T}} = nothing
) where T
    @assert size(v1) == size(v2) "Velocity fields must have same shape"

    # Compose: v2 + v1(v2)
    # Sample v1 at positions displaced by v2
    v1_warped = spatial_transform(v1, v2; id_grid=id_grid)
    return v2 .+ v1_warped
end

# ============================================================================
# Apply Flows (Bidirectional Warping)
# ============================================================================

"""
    FlowResult{T}

Named tuple containing the results of apply_flows.

# Fields
- `images`: NamedTuple with keys `xy_half`, `yx_half`, `xy_full`, `yx_full`
- `flows`: NamedTuple with keys `xy_half`, `yx_half`, `xy_full`, `yx_full`
"""
const FlowResult{T} = @NamedTuple{
    images::@NamedTuple{xy_half::Array{T,5}, yx_half::Array{T,5}, xy_full::Array{T,5}, yx_full::Array{T,5}},
    flows::@NamedTuple{xy_half::Array{T,5}, yx_half::Array{T,5}, xy_full::Array{T,5}, yx_full::Array{T,5}}
}

"""
    apply_flows(x, y, v_xy, v_yx; time_steps=7)

Apply bidirectional flows to images for symmetric registration.

# Arguments
- `x`: Moving image, shape `(X, Y, Z, C, N)`
- `y`: Static image, shape `(X, Y, Z, C, N)`
- `v_xy`: Velocity field x→y (moving to static), shape `(X, Y, Z, 3, N)`
- `v_yx`: Velocity field y→x (static to moving), shape `(X, Y, Z, 3, N)`
- `time_steps`: Number of scaling-and-squaring steps (default 7)

# Returns
Named tuple with two fields:
- `images`: NamedTuple containing:
  - `xy_half`: x warped halfway toward y
  - `yx_half`: y warped halfway toward x
  - `xy_full`: x fully warped to y space
  - `yx_full`: y fully warped to x space
- `flows`: NamedTuple containing:
  - `xy_half`, `yx_half`, `xy_full`, `yx_full`: corresponding displacement fields

# Algorithm
1. Compute half flows: exp(v_xy), exp(v_yx), exp(-v_xy), exp(-v_yx) using diffeomorphic_transform
2. Compute half images: warp x and y using forward half flows
3. Compute full flows: compose half flows (forward half + inverted backward half)
4. Compute full images: warp x and y using full flows

# Notes
The midpoint images (xy_half, yx_half) should be similar - this is the symmetric loss term.
Full images represent the complete transformation from one space to the other.
"""
function apply_flows(
    x::AbstractArray{T, 5},
    y::AbstractArray{T, 5},
    v_xy::AbstractArray{T, 5},
    v_yx::AbstractArray{T, 5};
    time_steps::Int = 7
) where T
    @assert size(x) == size(y) "x and y must have same shape"
    @assert size(v_xy) == size(v_yx) "v_xy and v_yx must have same shape"
    @assert size(x)[1:3] == size(v_xy)[1:3] "Spatial dimensions must match"

    X, Y, Z = size(x)[1:3]
    N = size(x, 5)

    # Get identity grid for all operations
    id_grid = ignore_derivatives() do
        get_identity_grid((X, Y, Z), T)
    end

    # Concatenate velocity fields for batch processing:
    # [v_xy, v_yx, -v_xy, -v_yx] along batch dimension
    # This computes all 4 diffeomorphic transforms at once
    neg_v_xy = -v_xy
    neg_v_yx = -v_yx
    v_all = cat(v_xy, v_yx, neg_v_xy, neg_v_yx; dims=5)  # (X, Y, Z, 3, 4N)

    # Compute all half flows using diffeomorphic transform
    half_flows_all = diffeomorphic_transform(v_all; time_steps=time_steps, id_grid=id_grid)

    # Split back into individual flows
    # half_flows_all has shape (X, Y, Z, 3, 4N) where batch dim contains:
    # [flow_xy (N), flow_yx (N), flow_neg_xy (N), flow_neg_yx (N)]
    half_flow_xy = half_flows_all[:, :, :, :, 1:N]           # exp(v_xy)
    half_flow_yx = half_flows_all[:, :, :, :, N+1:2N]        # exp(v_yx)
    half_flow_neg_xy = half_flows_all[:, :, :, :, 2N+1:3N]   # exp(-v_xy)
    half_flow_neg_yx = half_flows_all[:, :, :, :, 3N+1:4N]   # exp(-v_yx)

    # Compute half images: warp x with half_flow_xy, y with half_flow_yx
    # Concatenate images for batch processing
    xy_cat = cat(x, y; dims=5)  # (X, Y, Z, C, 2N)
    half_flows_forward = cat(half_flow_xy, half_flow_yx; dims=5)  # (X, Y, Z, 3, 2N)
    half_images_all = spatial_transform(xy_cat, half_flows_forward; id_grid=id_grid)

    # Split half images
    C = size(x, 4)
    xy_half = half_images_all[:, :, :, :, 1:N]      # x warped halfway
    yx_half = half_images_all[:, :, :, :, N+1:2N]   # y warped halfway

    # Compute full flows by composition:
    # full_xy = half_xy ∘ half_neg_yx (forward then back of inverse)
    # full_yx = half_yx ∘ half_neg_xy
    # Following torchreg: composition_transform(half_flows[:2], half_flows[2:].flip(0))
    # half_flows[2:].flip(0) means: [half_neg_yx, half_neg_xy] (reversed order)
    full_flow_xy = composition_transform(half_flow_xy, half_flow_neg_yx; id_grid=id_grid)
    full_flow_yx = composition_transform(half_flow_yx, half_flow_neg_xy; id_grid=id_grid)

    # Compute full images
    full_flows = cat(full_flow_xy, full_flow_yx; dims=5)
    full_images_all = spatial_transform(xy_cat, full_flows; id_grid=id_grid)

    xy_full = full_images_all[:, :, :, :, 1:N]      # x fully warped to y space
    yx_full = full_images_all[:, :, :, :, N+1:2N]   # y fully warped to x space

    # Package results
    images = (
        xy_half = xy_half,
        yx_half = yx_half,
        xy_full = xy_full,
        yx_full = yx_full
    )

    flows = (
        xy_half = half_flow_xy,
        yx_half = half_flow_yx,
        xy_full = full_flow_xy,
        yx_full = full_flow_yx
    )

    return (images = images, flows = flows)
end

# ============================================================================
# Gaussian Smoothing for Velocity Fields
# ============================================================================

"""
    gauss_smoothing(x::AbstractArray{T, 5}, sigma::Union{T, AbstractVector{T}}) where T

Apply Gaussian smoothing to a 5D array (velocity field or image).

# Arguments
- `x`: Input array, shape `(X, Y, Z, C, N)`
- `sigma`: Smoothing standard deviation. Can be:
  - A scalar (applied uniformly to all spatial dimensions)
  - A vector of length 3 (separate sigma for X, Y, Z)

# Returns
- Smoothed array of same shape

# Algorithm
1. Compute kernel size based on spatial dimensions: half_ks = spatial_size ÷ 50 (clamped to min 1)
2. Kernel size = 1 + 2 * half_ks (ensures odd size)
3. Create separable Gaussian kernel
4. Apply depthwise 3D convolution with replicate padding

# Notes
- Kernel size adapts to image size to ensure adequate smoothing coverage
- Uses replicate padding at boundaries
- Convolution is applied channel-wise (depthwise convolution)

# Example
```julia
v = randn(Float32, 64, 64, 64, 3, 1)  # Velocity field
sigma = 0.2f0
v_smooth = gauss_smoothing(v, sigma)
```
"""
function gauss_smoothing(
    x::AbstractArray{T, 5},
    sigma
) where T
    X, Y, Z, C, N = size(x)

    # Convert sigma to a vector of the correct type
    if sigma isa Number
        sigma_vec = fill(T(sigma), 3)
    else
        @assert length(sigma) == 3 "sigma vector must have length 3"
        sigma_vec = T.(sigma)
    end

    # Compute kernel size based on spatial dimensions (following torchreg)
    # half_kernel_size = spatial_size // 50, clamped to min 1
    half_ks_x = max(1, X ÷ 50)
    half_ks_y = max(1, Y ÷ 50)
    half_ks_z = max(1, Z ÷ 50)

    # Kernel size = 1 + 2 * half_ks (odd sizes)
    ks_x = 1 + 2 * half_ks_x
    ks_y = 1 + 2 * half_ks_y
    ks_z = 1 + 2 * half_ks_z

    kernel_size = (ks_x, ks_y, ks_z)

    # Create Gaussian kernel (constant w.r.t. optimization, so no gradient needed)
    kernel = ignore_derivatives() do
        smooth_kernel(kernel_size, NTuple{3, T}(sigma_vec))  # (ks_x, ks_y, ks_z)
    end

    # Apply depthwise 3D convolution with replicate padding
    # Need to reshape kernel for NNlib.conv: (W, H, D, Cin, Cout)
    # For depthwise conv: Cin=1, Cout=1, applied C times with groups=C

    # Pad input with replicate padding
    # Padding: (left, right, top, bottom, front, back) -> in Julia order: (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    pad_x = half_ks_x
    pad_y = half_ks_y
    pad_z = half_ks_z

    # Reshape kernel for conv: (ks_x, ks_y, ks_z) -> (ks_x, ks_y, ks_z, 1, 1)
    kernel_5d = ignore_derivatives() do
        reshape(kernel, ks_x, ks_y, ks_z, 1, 1)
    end

    # Pad the entire array first
    x_padded = pad_replicate_full(x, pad_x, pad_y, pad_z)

    # Apply convolution to each channel independently using map/stack
    # This avoids in-place mutation that Zygote can't handle
    results = [
        begin
            # Extract single channel across all batches: (X_pad, Y_pad, Z_pad, 1, N)
            slice = x_padded[:, :, :, c:c, :]
            # Conv each batch element
            conv_results = map(1:N) do n
                s = slice[:, :, :, :, n:n]  # (X_pad, Y_pad, Z_pad, 1, 1)
                NNlib.conv(s, kernel_5d)    # (X, Y, Z, 1, 1)
            end
            cat(conv_results...; dims=5)  # (X, Y, Z, 1, N)
        end
        for c in 1:C
    ]

    # Concatenate along channel dimension
    return cat(results...; dims=4)
end

"""
    pad_replicate_full(x::AbstractArray{T, 5}, pad_x::Int, pad_y::Int, pad_z::Int) where T

Apply replicate (edge) padding to a 5D array without mutation.
"""
function pad_replicate_full(
    x::AbstractArray{T, 5},
    pad_x::Int,
    pad_y::Int,
    pad_z::Int
) where T
    X, Y, Z, C, N = size(x)

    new_X = X + 2 * pad_x
    new_Y = Y + 2 * pad_y
    new_Z = Z + 2 * pad_z

    # Create index arrays for replicate padding
    ix = [clamp(i - pad_x, 1, X) for i in 1:new_X]
    iy = [clamp(j - pad_y, 1, Y) for j in 1:new_Y]
    iz = [clamp(k - pad_z, 1, Z) for k in 1:new_Z]

    return x[ix, iy, iz, :, :]
end

# ============================================================================
# SyNRegistration Type
# ============================================================================

"""
    SyNRegistration{T, F, R, O} <: AbstractRegistration

Symmetric Normalization (SyN) diffeomorphic registration.

SyN is a powerful non-rigid registration algorithm that:
1. Guarantees diffeomorphic (smooth, invertible) transformations
2. Uses bidirectional (symmetric) optimization
3. Employs velocity field parameterization with scaling-and-squaring

# Type Parameters
- `T`: Element type for parameters (typically Float32)
- `F`: Dissimilarity function type
- `R`: Regularization function type
- `O`: Optimizer type from Optimisers.jl

# Fields

## Configuration
- `scales::Tuple{Vararg{Int}}`: Multi-resolution pyramid scales (e.g., (4, 2, 1))
- `iterations::Tuple{Vararg{Int}}`: Number of iterations per scale level
- `learning_rates::Vector{T}`: Learning rates per scale level
- `verbose::Bool`: Whether to display progress during registration

## Loss Functions
- `dissimilarity_fn::F`: Image dissimilarity function (e.g., MSE, NCC)
- `regularization_fn::R`: Regularization function for flow fields
- `optimizer::O`: Optimizer type from Optimisers.jl

## Regularization Parameters
- `sigma_img::T`: Gaussian smoothing sigma for images (0 to disable)
- `sigma_flow::T`: Gaussian smoothing sigma for velocity fields
- `lambda_::T`: Regularization weight

## Diffeomorphic Settings
- `time_steps::Int`: Number of scaling-and-squaring steps (default 7)

## Learned State (populated after registration)
- `v_xy::Union{Nothing, Array{T, 5}}`: Velocity field moving→static
- `v_yx::Union{Nothing, Array{T, 5}}`: Velocity field static→moving

# Example
```julia
# Create SyN registration
reg = SyNRegistration(
    scales=(4, 2, 1),
    iterations=(30, 30, 10),
    learning_rate=1e-2f0,
    lambda_=2e-5f0
)

# Register moving to static
moved_xy, moved_yx, flow_xy, flow_yx = register(moving, static, reg)
```

# See Also
- [`register`](@ref): Run registration
- `diffeomorphic_transform`: Core scaling-and-squaring algorithm
- `spatial_transform`: Warp images using displacement fields
- `apply_flows`: Compute bidirectional warped images
"""
mutable struct SyNRegistration{T, F, R, O} <: AbstractRegistration
    # Configuration
    scales::Tuple{Vararg{Int}}
    iterations::Tuple{Vararg{Int}}
    learning_rates::Vector{T}
    verbose::Bool

    # Loss functions
    dissimilarity_fn::F
    regularization_fn::R
    optimizer::O

    # Regularization parameters
    sigma_img::T     # Gaussian smoothing for images
    sigma_flow::T    # Gaussian smoothing for velocity fields
    lambda_::T       # Regularization weight

    # Diffeomorphic settings
    time_steps::Int  # Number of scaling-and-squaring steps

    # Learned state (populated after registration)
    v_xy::Union{Nothing, Array{T, 5}}  # Velocity field: moving → static
    v_yx::Union{Nothing, Array{T, 5}}  # Velocity field: static → moving
end

"""
    SyNRegistration(;
        scales=(4, 2, 1),
        iterations=(30, 30, 10),
        learning_rate=1e-2,
        verbose=true,
        dissimilarity_fn=mse_loss,
        regularization_fn=LinearElasticity(),
        optimizer=Adam,
        sigma_img=0.2f0,
        sigma_flow=0.2f0,
        lambda_=2e-5f0,
        time_steps=7
    )

Create a SyN registration object with the specified configuration.

# Arguments

## Multi-resolution Pyramid
- `scales::Tuple=(4, 2, 1)`: Downsampling factors for pyramid levels.
  Images are downsampled by 1/scale at each level.
- `iterations::Tuple=(30, 30, 10)`: Optimization iterations per pyramid level.
  Must have same length as `scales`.

## Optimization
- `learning_rate=1e-2`: Learning rate for optimizer. Can be a scalar
  (same for all scales) or a tuple/vector matching scales.
- `verbose::Bool=true`: Display progress during registration
- `dissimilarity_fn=mse_loss`: Loss function for image similarity
- `regularization_fn=nothing`: Regularization for flow smoothness (LinearElasticity not yet Zygote-compatible)
- `optimizer=Adam`: Optimizer type from Optimisers.jl

## Regularization
- `sigma_img=0.2f0`: Gaussian smoothing sigma for images. Set to 0 to disable.
- `sigma_flow=0.2f0`: Gaussian smoothing sigma for velocity fields.
- `lambda_=2e-5f0`: Weight for regularization term in loss.

## Diffeomorphic Settings
- `time_steps::Int=7`: Number of scaling-and-squaring steps.
  Higher values give more accurate diffeomorphism.

# Returns
- `SyNRegistration`: Registration object ready for use with `register()`

# Notes
- Default regularization is `nothing` (LinearElasticity not yet Zygote-compatible)
- If regularization_fn is `nothing`, only dissimilarity is used
- Learning rate can be a scalar or per-scale tuple
"""
function SyNRegistration(;
    scales::Tuple{Vararg{Int}} = (4, 2, 1),
    iterations::Tuple{Vararg{Int}} = (30, 30, 10),
    learning_rate = 1e-2,
    verbose::Bool = true,
    dissimilarity_fn = mse_loss,
    regularization_fn = nothing,  # LinearElasticity not yet Zygote-compatible
    optimizer = Adam,
    sigma_img = Float32(0.2),
    sigma_flow = Float32(0.2),
    lambda_ = Float32(2e-5),
    time_steps::Int = 7
)
    @assert length(scales) == length(iterations) "scales and iterations must have same length"
    @assert all(s > 0 for s in scales) "all scales must be positive"
    @assert all(i > 0 for i in iterations) "all iterations must be positive"

    T = Float32

    # Create learning rate vector
    if learning_rate isa Number
        learning_rates = fill(T(learning_rate), length(scales))
    else
        @assert length(learning_rate) == length(scales) "learning_rate must match scales length"
        learning_rates = T.(collect(learning_rate))
    end

    return SyNRegistration{T, typeof(dissimilarity_fn), typeof(regularization_fn), typeof(optimizer)}(
        scales,
        iterations,
        learning_rates,
        verbose,
        dissimilarity_fn,
        regularization_fn,
        optimizer,
        T(sigma_img),
        T(sigma_flow),
        T(lambda_),
        time_steps,
        nothing,  # v_xy
        nothing   # v_yx
    )
end

# ============================================================================
# SyN Fit and Register Functions
# ============================================================================

"""
    upsample_velocity(v::AbstractArray{T, 5}, target_size::NTuple{3, Int}) where T

Upsample velocity field to target spatial size using trilinear interpolation.
"""
function upsample_velocity(v::AbstractArray{T, 5}, target_size::NTuple{3, Int}) where T
    current_size = size(v)[1:3]
    if current_size == target_size
        return copy(v)
    end

    X, Y, Z, C, N = size(v)
    X_t, Y_t, Z_t = target_size

    # Create identity affine for interpolation
    affine = identity_affine(3, N, T)

    # Treat velocity field channels as "image channels" for interpolation
    # v: (X, Y, Z, 3, N) - treat 3 velocity components as channels
    return affine_transform(v, affine; shape=target_size, padding_mode=:border)
end

"""
    fit!(reg::SyNRegistration, moving, static, iterations::Int, learning_rate; verbose=nothing)

Run SyN optimization for a single resolution level.

# Arguments
- `reg`: SyNRegistration with velocity fields already initialized at correct size
- `moving`: Moving image at current resolution, shape `(X, Y, Z, C, N)`
- `static`: Static target image, same shape as moving
- `iterations`: Number of optimization iterations
- `learning_rate`: Learning rate for this scale

# Notes
- Modifies `reg.v_xy` and `reg.v_yx` in place
- Uses Zygote for gradient computation
- Uses Optimisers.jl for parameter updates
"""
function fit!(
    reg::SyNRegistration{T},
    moving::AbstractArray{T, 5},
    static::AbstractArray{T, 5},
    iterations::Int,
    learning_rate::T;
    verbose::Union{Nothing, Bool} = nothing
) where T
    verbose = verbose === nothing ? reg.verbose : verbose

    X, Y, Z = size(moving)[1:3]
    N = size(moving, 5)

    # Get current velocity fields at this resolution
    v_xy = reg.v_xy
    v_yx = reg.v_yx

    # Smoothing sigma for velocity fields
    sigma_flow = fill(reg.sigma_flow, 3)

    # Pack parameters for optimization
    params = (v_xy, v_yx)

    # Setup optimizer
    opt_rule = reg.optimizer(learning_rate)
    opt_state = Optimisers.setup(opt_rule, params)

    # Optimization loop
    local loss_val, dissim_val, reg_val
    for iter in 1:iterations
        # Compute loss and gradients
        (loss_val, dissim_val, reg_val), grads = Zygote.withgradient(params) do p
            vxy, vyx = p

            # Smooth velocity fields
            vxy_smooth = gauss_smoothing(vxy, sigma_flow)
            vyx_smooth = gauss_smoothing(vyx, sigma_flow)

            # Apply flows
            result = apply_flows(moving, static, vxy_smooth, vyx_smooth;
                                 time_steps=reg.time_steps)

            # Symmetric dissimilarity loss:
            # - moving should match yx_full (static warped to moving space)
            # - static should match xy_full (moving warped to static space)
            # - half images should match at midpoint
            dissimilarity = (
                reg.dissimilarity_fn(moving, result.images.yx_full) +
                reg.dissimilarity_fn(static, result.images.xy_full) +
                reg.dissimilarity_fn(result.images.xy_half, result.images.yx_half)
            )

            # Regularization on flow fields
            regularization = T(0)
            if reg.regularization_fn !== nothing
                regularization = (
                    reg.regularization_fn(result.flows.xy_full) +
                    reg.regularization_fn(result.flows.yx_full)
                )
            end

            total_loss = dissimilarity + reg.lambda_ * regularization

            return total_loss, dissimilarity, regularization
        end

        # Get gradients
        grad_vxy, grad_vyx = grads[1]

        # Update parameters
        opt_state, params = Optimisers.update!(opt_state, params, (grad_vxy, grad_vyx))

        # Display progress
        if verbose && (iter == 1 || iter % 10 == 0 || iter == iterations)
            println("  Iteration $iter/$iterations: loss=$(round(loss_val; digits=6)), " *
                    "dissim=$(round(dissim_val; digits=6)), reg=$(round(reg_val; digits=6))")
        end
    end

    # Update stored velocity fields with final smoothed versions
    v_xy, v_yx = params
    reg.v_xy = gauss_smoothing(v_xy, sigma_flow)
    reg.v_yx = gauss_smoothing(v_yx, sigma_flow)

    return loss_val, dissim_val, reg_val
end

"""
    register(moving, static, reg::SyNRegistration; v_xy=nothing, v_yx=nothing, return_moved=true)

Perform SyN diffeomorphic registration of moving image to static image.

# Arguments
- `moving`: Moving image to register, shape `(X, Y, Z, C, N)`
- `static`: Static target image, same shape convention
- `reg`: SyNRegistration configuration object
- `v_xy`: Optional initial velocity field moving→static
- `v_yx`: Optional initial velocity field static→moving
- `return_moved`: If true, compute and return warped images

# Returns
If `return_moved=true`, returns tuple:
- `moved_xy`: Moving image warped to static space (full transform)
- `moved_yx`: Static image warped to moving space (full transform)
- `flow_xy`: Displacement field moving→static, shape `(X, Y, Z, 3, N)`
- `flow_yx`: Displacement field static→moving, shape `(X, Y, Z, 3, N)`

If `return_moved=false`, returns `nothing`. Use `apply_flows` manually to get results.

# Notes
- Initializes velocity fields to zeros if not provided
- Runs multi-resolution optimization pyramid
- After completion, `reg.v_xy` and `reg.v_yx` contain learned velocity fields

# Example
```julia
reg = SyNRegistration(scales=(4, 2, 1), iterations=(30, 30, 10))
moved_xy, moved_yx, flow_xy, flow_yx = register(moving, static, reg)
```
"""
function register(
    moving::AbstractArray{T, 5},
    static::AbstractArray{T, 5},
    reg::SyNRegistration{T};
    v_xy::Union{Nothing, AbstractArray{T, 5}} = nothing,
    v_yx::Union{Nothing, AbstractArray{T, 5}} = nothing,
    return_moved::Bool = true
) where T
    # Validate input shapes
    @assert size(moving) == size(static) "moving and static must have same shape"
    @assert ndims(moving) == 5 "Expected 5D arrays (X, Y, Z, C, N)"

    X, Y, Z, C, N = size(moving)
    spatial_shape = (X, Y, Z)

    # Initialize velocity fields to zeros if not provided
    if v_xy === nothing
        v_xy = zeros(T, X, Y, Z, 3, N)
    end
    if v_yx === nothing
        v_yx = zeros(T, X, Y, Z, 3, N)
    end

    # Store initial velocity fields
    reg.v_xy = copy(v_xy)
    reg.v_yx = copy(v_yx)

    if reg.verbose
        println("Starting SyN registration...")
        println("  Image shape: $spatial_shape × $C channels × $N batch")
        println("  Scales: $(reg.scales)")
        println("  Iterations: $(reg.iterations)")
    end

    # Multi-resolution optimization
    for (i, (scale, iters)) in enumerate(zip(reg.scales, reg.iterations))
        lr = reg.learning_rates[i]

        if reg.verbose
            println("\nScale 1/$scale (lr=$lr):")
        end

        # Compute shape at this scale
        scaled_shape = ntuple(d -> max(1, round(Int, spatial_shape[d] / scale)), 3)

        # Downsample images if needed
        if scaled_shape != spatial_shape
            moving_small = upsample_velocity(moving, scaled_shape)  # Works for images too
            static_small = upsample_velocity(static, scaled_shape)
        else
            moving_small = moving
            static_small = static
        end

        # Upsample velocity fields to current resolution
        reg.v_xy = upsample_velocity(reg.v_xy, scaled_shape)
        reg.v_yx = upsample_velocity(reg.v_yx, scaled_shape)

        # Optional image smoothing
        if reg.sigma_img > 0
            # Scale sigma based on image size (following torchreg)
            sigma_img_scaled = reg.sigma_img .* 200.0f0 ./ T.(collect(scaled_shape))
            moving_small = gauss_smoothing(moving_small, sigma_img_scaled)
            static_small = gauss_smoothing(static_small, sigma_img_scaled)
        end

        # Run optimization at this scale
        fit!(reg, moving_small, static_small, iters, lr)
    end

    # Upsample velocity fields to full resolution
    reg.v_xy = upsample_velocity(reg.v_xy, spatial_shape)
    reg.v_yx = upsample_velocity(reg.v_yx, spatial_shape)

    if reg.verbose
        println("\nRegistration complete.")
    end

    # Return warped images if requested
    if return_moved
        result = apply_flows(moving, static, reg.v_xy, reg.v_yx;
                            time_steps=reg.time_steps)
        return (
            result.images.xy_full,   # moving warped to static
            result.images.yx_full,   # static warped to moving
            result.flows.xy_full,    # flow moving→static
            result.flows.yx_full     # flow static→moving
        )
    else
        return nothing
    end
end
