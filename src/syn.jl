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
