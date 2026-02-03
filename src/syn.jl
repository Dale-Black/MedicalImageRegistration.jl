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

# Placeholder - will be fully implemented in IMPL-SYN-003
"""
    SyNRegistration{T, F, R, O} <: AbstractRegistration

Symmetric Normalization (SyN) diffeomorphic registration.

SyN is a powerful non-rigid registration algorithm that:
1. Guarantees diffeomorphic (smooth, invertible) transformations
2. Uses bidirectional (symmetric) optimization
3. Employs velocity field parameterization with scaling-and-squaring

# Fields (to be implemented in IMPL-SYN-003)
- Configuration: scales, iterations, learning rates
- Velocity fields: v_xy (moving→static), v_yx (static→moving)
- Regularization parameters: sigma_img, sigma_flow, lambda_

# See Also
- `diffeomorphic_transform`: Core scaling-and-squaring algorithm
- `spatial_transform`: Warp images using displacement fields
- `composition_transform`: Compose displacement fields
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

# Default constructor - will be implemented fully in IMPL-SYN-003
# For now, just a minimal version for testing
function SyNRegistration(;
    scales::Tuple{Vararg{Int}} = (4, 2, 1),
    iterations::Tuple{Vararg{Int}} = (30, 30, 10),
    learning_rate = 1e-2,
    verbose::Bool = true,
    dissimilarity_fn = mse_loss,
    regularization_fn = nothing,  # Will default to LinearElasticity
    optimizer = Adam,
    sigma_img = 0.2f0,
    sigma_flow = 0.2f0,
    lambda_ = 2e-5f0,
    time_steps::Int = 7
)
    T = Float32

    # Create learning rate vector
    lr = T(learning_rate)
    learning_rates = fill(lr, length(scales))

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
