# Common types for MedicalImageRegistration.jl
# GPU-first architecture with Mooncake rrule!! for AD

# ============================================================================
# AffineRegistration Configuration and State
# ============================================================================

"""
    AffineRegistration{T, A<:AbstractArray{T}}

Holds configuration and learned parameters for affine image registration.

# Fields
- `scales`: Tuple of scales for multi-resolution pyramid (e.g., (4, 2))
- `iterations`: Number of iterations at each scale
- `learning_rate`: Learning rate for optimizer
- `is_3d`: Whether to use 3D (true) or 2D (false) registration
- `align_corners`: Align corners mode for grid sampling
- `padding_mode`: Padding mode for grid sampling (:zeros or :border)
- `with_translation`: Enable translation parameters
- `with_rotation`: Enable rotation parameters
- `with_zoom`: Enable zoom/scale parameters
- `with_shear`: Enable shear parameters
- `translation`: Translation parameters (D, N)
- `rotation`: Rotation matrix parameters (D, D, N)
- `zoom`: Scale/zoom parameters (D, N)
- `shear`: Shear parameters (D, N)
- `loss_history`: Loss values during training

# Example
```julia
using Metal

# Create registration model
reg = AffineRegistration{Float32}(
    scales=(4, 2),
    iterations=(500, 100),
    is_3d=true
)

# Register moving image to static image
moved = register(reg, moving, static)
```
"""
mutable struct AffineRegistration{T, A<:AbstractArray{T}}
    # Configuration
    scales::Tuple{Vararg{Int}}
    iterations::Tuple{Vararg{Int}}
    learning_rate::T
    is_3d::Bool
    align_corners::Bool
    padding_mode::Symbol

    # Parameter flags
    with_translation::Bool
    with_rotation::Bool
    with_zoom::Bool
    with_shear::Bool

    # Learned parameters (D, N) or (D, D, N) for rotation
    translation::A
    rotation::AbstractArray{T,3}
    zoom::A
    shear::A

    # Training history
    loss_history::Vector{T}
end

"""
    AffineRegistration{T}(; kwargs...) where T

Create a new AffineRegistration with default parameters.

# Keyword Arguments
- `scales::Tuple=(4, 2)`: Multi-resolution pyramid scales
- `iterations::Tuple=(500, 100)`: Iterations per scale
- `learning_rate::T=T(1e-2)`: Optimizer learning rate
- `is_3d::Bool=true`: 3D (true) or 2D (false) registration
- `align_corners::Bool=true`: Grid sampling align_corners
- `padding_mode::Symbol=:border`: Grid sampling padding mode
- `with_translation::Bool=true`: Enable translation
- `with_rotation::Bool=true`: Enable rotation
- `with_zoom::Bool=true`: Enable zoom
- `with_shear::Bool=false`: Enable shear
- `batch_size::Int=1`: Batch size for parameters
- `array_type::Type=Array`: Array type (Array, MtlArray, CuArray, etc.)
"""
function AffineRegistration{T}(;
    scales::Tuple{Vararg{Int}}=(4, 2),
    iterations::Tuple{Vararg{Int}}=(500, 100),
    learning_rate::T=T(1e-2),
    is_3d::Bool=true,
    align_corners::Bool=true,
    padding_mode::Symbol=:border,
    with_translation::Bool=true,
    with_rotation::Bool=true,
    with_zoom::Bool=true,
    with_shear::Bool=false,
    batch_size::Int=1,
    array_type::Type{<:AbstractArray}=Array
) where T
    # Dimension based on 2D/3D
    D = is_3d ? 3 : 2
    N = batch_size

    # Initialize parameters on CPU first, then convert to target array type
    # This avoids scalar indexing on GPU arrays

    # Translation: zeros (no initial translation)
    translation_cpu = zeros(T, D, N)

    # Rotation: identity matrix
    rotation_cpu = zeros(T, D, D, N)
    for n in 1:N
        for d in 1:D
            rotation_cpu[d, d, n] = one(T)
        end
    end

    # Zoom: ones (no scaling)
    zoom_cpu = ones(T, D, N)

    # Shear: zeros (no shear)
    shear_cpu = zeros(T, D, N)

    # Convert to target array type
    translation = array_type(translation_cpu)
    rotation = array_type(rotation_cpu)
    zoom = array_type(zoom_cpu)
    shear = array_type(shear_cpu)

    # Ensure iterations tuple matches scales length
    iters = length(iterations) >= length(scales) ? iterations[1:length(scales)] : iterations

    return AffineRegistration{T, typeof(translation)}(
        scales, iters, learning_rate, is_3d, align_corners, padding_mode,
        with_translation, with_rotation, with_zoom, with_shear,
        translation, rotation, zoom, shear,
        T[]
    )
end

# Convenience constructor that infers T from array_type
function AffineRegistration(; T::Type{<:AbstractFloat}=Float32, kwargs...)
    return AffineRegistration{T}(; kwargs...)
end

"""
    reset!(reg::AffineRegistration{T}) where T

Reset parameters to identity transformation.
"""
function reset!(reg::AffineRegistration{T}) where T
    D = reg.is_3d ? 3 : 2
    N = size(reg.translation, 2)

    fill!(reg.translation, zero(T))
    fill!(reg.zoom, one(T))
    fill!(reg.shear, zero(T))

    # Reset rotation to identity - create on CPU then copy to avoid scalar indexing
    rotation_cpu = zeros(T, D, D, N)
    for n in 1:N
        for d in 1:D
            rotation_cpu[d, d, n] = one(T)
        end
    end
    copyto!(reg.rotation, rotation_cpu)

    empty!(reg.loss_history)

    return reg
end

"""
    get_affine(reg::AffineRegistration)

Get the current affine transformation matrix from parameters.

# Returns
- `theta`: Affine matrix (D, D+1, N)
"""
function get_affine(reg::AffineRegistration)
    return compose_affine(reg.translation, reg.rotation, reg.zoom, reg.shear)
end
