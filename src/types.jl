# Core types for MedicalImageRegistration.jl

"""
Abstract type for all registration algorithms.
"""
abstract type AbstractRegistration end

# ============================================================================
# Affine Registration Types
# ============================================================================

"""
    AffineParameters{T}

Stores the learnable parameters for affine registration.

# Fields
- `translation::Vector{T}`: Translation parameters, shape `(ndim,)` per batch.
- `rotation::Matrix{T}`: Rotation matrix, shape `(ndim, ndim)` per batch.
- `zoom::Vector{T}`: Scaling factors, shape `(ndim,)` per batch.
- `shear::Vector{T}`: Shear parameters, shape `(ndim,)` per batch.

# Notes
- For batch_size > 1, parameters are stored as 2D/3D arrays with last dimension being batch.
- Translation is initialized to zeros.
- Rotation is initialized to identity matrix.
- Zoom is initialized to ones.
- Shear is initialized to zeros.
"""
struct AffineParameters{T}
    translation::Array{T}  # (ndim, N) or (ndim,) for N=1
    rotation::Array{T}     # (ndim, ndim, N) or (ndim, ndim) for N=1
    zoom::Array{T}         # (ndim, N) or (ndim,) for N=1
    shear::Array{T}        # (ndim, N) or (ndim,) for N=1
end

"""
    AffineRegistration{T, F, O} <: AbstractRegistration

Affine registration using gradient descent optimization.

Supports translation, rotation, zoom (scaling), and shear transformations.
Uses multi-resolution pyramid for robust convergence.

# Type Parameters
- `T`: Element type for parameters (typically Float32)
- `F`: Dissimilarity function type
- `O`: Optimizer type from Optimisers.jl

# Fields

## Configuration
- `ndims::Int`: Number of spatial dimensions (2 or 3)
- `scales::NTuple{N, Int}`: Multi-resolution pyramid scales (e.g., (4, 2) means 1/4, 1/2 resolution)
- `iterations::NTuple{N, Int}`: Number of iterations per scale level
- `learning_rate::T`: Learning rate for optimizer

## Optimization
- `verbose::Bool`: Whether to display progress during registration
- `dissimilarity_fn::F`: Loss function (e.g., MSE, NCC, dice_loss)
- `optimizer::Type{O}`: Optimizer type from Optimisers.jl (e.g., Adam)

## Transformation Flags
- `with_translation::Bool`: Optimize translation parameters
- `with_rotation::Bool`: Optimize rotation parameters
- `with_zoom::Bool`: Optimize zoom (scale) parameters
- `with_shear::Bool`: Optimize shear parameters

## Interpolation Settings
- `interp_mode::Symbol`: Interpolation mode (:bilinear for 2D, :trilinear for 3D)
- `padding_mode::Symbol`: Padding mode for out-of-bounds (:border, :zeros)
- `align_corners::Bool`: Whether to align corners in grid sampling

## Learned State (populated after calling `register`)
- `parameters::Union{Nothing, AffineParameters{T}}`: Learned transformation parameters
- `loss::Union{Nothing, T}`: Final loss value after registration

# Example
```julia
# Create registration object
reg = AffineRegistration(;
    ndims=3,
    scales=(4, 2),
    iterations=(500, 100),
    learning_rate=1e-2f0,
    with_translation=true,
    with_rotation=true,
    with_zoom=true,
    with_shear=false
)

# Register moving to static image
moved = register(moving, static, reg)

# Get the learned affine matrix
affine = get_affine(reg)

# Transform another image with the learned parameters
other_moved = transform(other_image, reg)
```

# See Also
- [`register`](@ref): Run registration
- [`transform`](@ref): Apply learned transformation
- [`get_affine`](@ref): Get affine transformation matrix
- [`compose_affine`](@ref): Build affine matrix from parameters
"""
mutable struct AffineRegistration{T, F, O} <: AbstractRegistration
    # Configuration
    ndims::Int
    scales::Tuple{Vararg{Int}}
    iterations::Tuple{Vararg{Int}}
    learning_rate::T

    # Optimization
    verbose::Bool
    dissimilarity_fn::F
    optimizer::O  # Store the optimizer type or instance

    # Transformation flags
    with_translation::Bool
    with_rotation::Bool
    with_zoom::Bool
    with_shear::Bool

    # Interpolation settings
    interp_mode::Symbol
    padding_mode::Symbol
    align_corners::Bool

    # Initial parameters (optional)
    init_translation::Union{Nothing, Array{T}}
    init_rotation::Union{Nothing, Array{T}}
    init_zoom::Union{Nothing, Array{T}}
    init_shear::Union{Nothing, Array{T}}

    # Learned state (populated after registration)
    parameters::Union{Nothing, AffineParameters{T}}
    loss::Union{Nothing, T}
end
