# Affine registration implementation

using Optimisers: Adam

# Default MSE loss for affine registration
"""
    mse_loss(x, y)

Mean squared error loss function.

Returns `mean((x - y)^2)`.
"""
mse_loss(x, y) = mean((x .- y) .^ 2)

# ============================================================================
# AffineRegistration Constructor
# ============================================================================

"""
    AffineRegistration(;
        ndims=3,
        scales=(4, 2),
        iterations=(500, 100),
        learning_rate=1e-2f0,
        verbose=true,
        dissimilarity_fn=mse_loss,
        optimizer=Adam,
        with_translation=true,
        with_rotation=true,
        with_zoom=true,
        with_shear=false,
        interp_mode=nothing,
        padding_mode=:border,
        align_corners=true,
        init_translation=nothing,
        init_rotation=nothing,
        init_zoom=nothing,
        init_shear=nothing
    )

Create an affine registration object with the specified configuration.

# Arguments

## Required Configuration
- `ndims::Int=3`: Number of spatial dimensions (2 or 3)

## Multi-resolution Pyramid
- `scales::Tuple=(4, 2)`: Downsampling factors for pyramid levels.
  Images are downsampled by 1/scale at each level.
- `iterations::Tuple=(500, 100)`: Optimization iterations per pyramid level.
  Must have same length as `scales`.

## Optimization
- `learning_rate::Real=1e-2`: Learning rate for optimizer
- `verbose::Bool=true`: Display progress during registration
- `dissimilarity_fn=mse_loss`: Loss function. Can be:
  - `mse_loss` (default): Mean squared error
  - `dice_loss`: Dice loss for segmentation masks
  - `NCC(kernel_size)`: Normalized cross-correlation
- `optimizer=Adam`: Optimizer type from Optimisers.jl

## Transformation Components
- `with_translation::Bool=true`: Optimize translation
- `with_rotation::Bool=true`: Optimize rotation
- `with_zoom::Bool=true`: Optimize scaling
- `with_shear::Bool=false`: Optimize shear

## Interpolation
- `interp_mode::Union{Symbol, Nothing}=nothing`: Interpolation mode.
  If `nothing`, automatically selects `:bilinear` for 2D, `:trilinear` for 3D.
- `padding_mode::Symbol=:border`: Padding for out-of-bounds.
  Options: `:border` (replicate edge), `:zeros`
- `align_corners::Bool=true`: Align corner pixels in interpolation

## Initial Parameters (optional)
- `init_translation`: Initial translation, shape `(ndims, batch_size)`
- `init_rotation`: Initial rotation matrix, shape `(ndims, ndims, batch_size)`
- `init_zoom`: Initial zoom/scale, shape `(ndims, batch_size)`
- `init_shear`: Initial shear, shape `(ndims, batch_size)`

# Returns
- `AffineRegistration`: Registration object ready for use with `register()`

# Example
```julia
# Basic 3D registration
reg = AffineRegistration(ndims=3)

# 2D registration with custom settings
reg = AffineRegistration(
    ndims=2,
    scales=(4, 2, 1),
    iterations=(200, 100, 50),
    learning_rate=5e-3f0,
    dissimilarity_fn=NCC(7)
)

# Registration without zoom/shear (rigid only)
reg = AffineRegistration(
    ndims=3,
    with_zoom=false,
    with_shear=false
)
```

# Notes
- The `scales` and `iterations` tuples must have the same length.
- For medical images, `padding_mode=:border` (default) usually works best.
- Setting `verbose=true` shows a progress bar during optimization.
- After calling `register()`, the learned parameters are stored in `reg.parameters`.
"""
function AffineRegistration(;
    ndims::Int = 3,
    scales::Tuple{Vararg{Int}} = (4, 2),
    iterations::Tuple{Vararg{Int}} = (500, 100),
    learning_rate::Real = 1e-2,
    verbose::Bool = true,
    dissimilarity_fn = mse_loss,
    optimizer = Adam,
    with_translation::Bool = true,
    with_rotation::Bool = true,
    with_zoom::Bool = true,
    with_shear::Bool = false,
    interp_mode::Union{Symbol, Nothing} = nothing,
    padding_mode::Symbol = :border,
    align_corners::Bool = true,
    init_translation = nothing,
    init_rotation = nothing,
    init_zoom = nothing,
    init_shear = nothing
)
    # Validate inputs
    @assert ndims in (2, 3) "ndims must be 2 or 3, got $ndims"
    @assert length(scales) == length(iterations) "scales and iterations must have same length"
    @assert all(s > 0 for s in scales) "all scales must be positive"
    @assert all(i > 0 for i in iterations) "all iterations must be positive"
    @assert padding_mode in (:border, :zeros) "padding_mode must be :border or :zeros"

    # Auto-select interpolation mode based on dimensions
    if interp_mode === nothing
        interp_mode = ndims == 3 ? :trilinear : :bilinear
    end

    # Convert learning rate to Float32
    T = Float32
    lr = T(learning_rate)

    # Convert initial parameters to Float32 if provided
    init_t = init_translation === nothing ? nothing : T.(init_translation)
    init_r = init_rotation === nothing ? nothing : T.(init_rotation)
    init_z = init_zoom === nothing ? nothing : T.(init_zoom)
    init_s = init_shear === nothing ? nothing : T.(init_shear)

    return AffineRegistration{T, typeof(dissimilarity_fn), typeof(optimizer)}(
        ndims,
        scales,
        iterations,
        lr,
        verbose,
        dissimilarity_fn,
        optimizer,
        with_translation,
        with_rotation,
        with_zoom,
        with_shear,
        interp_mode,
        padding_mode,
        align_corners,
        init_t,
        init_r,
        init_z,
        init_s,
        nothing,  # parameters (learned state)
        nothing   # loss (learned state)
    )
end

# ============================================================================
# Parameter Initialization
# ============================================================================

"""
    init_parameters(reg::AffineRegistration, batch_size::Int) -> AffineParameters

Initialize affine transformation parameters for optimization.

# Arguments
- `reg`: AffineRegistration object with configuration
- `batch_size`: Number of images in the batch

# Returns
- `AffineParameters`: Struct containing translation, rotation, zoom, and shear arrays

# Notes
- Parameters are initialized based on `reg.init_*` fields if provided
- Translation initialized to zeros
- Rotation initialized to identity matrix
- Zoom initialized to ones
- Shear initialized to zeros
- Parameter gradients are controlled by `reg.with_*` flags during optimization
"""
function init_parameters(reg::AffineRegistration{T}, batch_size::Int) where T
    ndim = reg.ndims

    # Initialize translation: (ndim, batch_size)
    if reg.init_translation !== nothing
        translation = copy(reg.init_translation)
    else
        translation = zeros(T, ndim, batch_size)
    end

    # Initialize rotation: (ndim, ndim, batch_size) - identity matrices
    if reg.init_rotation !== nothing
        rotation = copy(reg.init_rotation)
    else
        rotation = zeros(T, ndim, ndim, batch_size)
        @inbounds for n in 1:batch_size
            for i in 1:ndim
                rotation[i, i, n] = one(T)
            end
        end
    end

    # Initialize zoom: (ndim, batch_size) - ones
    if reg.init_zoom !== nothing
        zoom = copy(reg.init_zoom)
    else
        zoom = ones(T, ndim, batch_size)
    end

    # Initialize shear: (ndim, batch_size) - zeros
    if reg.init_shear !== nothing
        shear = copy(reg.init_shear)
    else
        shear = zeros(T, ndim, batch_size)
    end

    return AffineParameters{T}(translation, rotation, zoom, shear)
end

"""
    check_parameter_shapes(params::AffineParameters, ndim::Int, batch_size::Int)

Validate that all parameter arrays have the correct shapes.

Throws `ArgumentError` if any shapes are incorrect.
"""
function check_parameter_shapes(params::AffineParameters, ndim::Int, batch_size::Int)
    # Check translation shape: (ndim, batch_size)
    if size(params.translation) != (ndim, batch_size)
        throw(ArgumentError(
            "Expected translation shape ($ndim, $batch_size), got $(size(params.translation))"
        ))
    end

    # Check rotation shape: (ndim, ndim, batch_size)
    if size(params.rotation) != (ndim, ndim, batch_size)
        throw(ArgumentError(
            "Expected rotation shape ($ndim, $ndim, $batch_size), got $(size(params.rotation))"
        ))
    end

    # Check zoom shape: (ndim, batch_size)
    if size(params.zoom) != (ndim, batch_size)
        throw(ArgumentError(
            "Expected zoom shape ($ndim, $batch_size), got $(size(params.zoom))"
        ))
    end

    # Check shear shape: (ndim, batch_size)
    if size(params.shear) != (ndim, batch_size)
        throw(ArgumentError(
            "Expected shear shape ($ndim, $batch_size), got $(size(params.shear))"
        ))
    end
end

# ============================================================================
# Compose Affine Matrix
# ============================================================================

"""
    compose_affine(params::AffineParameters{T}) where T
    compose_affine(translation, rotation, zoom, shear)

Compose an affine transformation matrix from individual components.

# Arguments
- `params`: AffineParameters struct, or individual arrays:
  - `translation`: Translation vector, shape `(ndim, batch_size)`
  - `rotation`: Rotation matrix, shape `(ndim, ndim, batch_size)`
  - `zoom`: Scaling factors, shape `(ndim, batch_size)`
  - `shear`: Shear parameters, shape `(ndim, batch_size)`

# Returns
- Affine matrix of shape `(ndim, ndim+1, batch_size)`

# Matrix Construction

For 3D (ndim=3), the affine matrix is constructed as:

```
[zoom_x  shear_xy  shear_xz] [r00 r01 r02]   [tx]
[  0     zoom_y    shear_yz] [r10 r11 r12] | [ty]
[  0       0       zoom_z  ] [r20 r21 r22]   [tz]

= rotation @ scale_shear + translation
```

Where:
- First, a scale/shear matrix is created with zoom on diagonal and shear off-diagonal
- The rotation matrix is multiplied with this scale/shear matrix
- Translation is concatenated as the last column

For 2D (ndim=2):
- shear has 1 component: shear_xy
- Matrix is 2×3

# Example
```julia
params = init_parameters(reg, 1)
affine = compose_affine(params)  # (3, 4, 1) for 3D

# Or with individual arrays
affine = compose_affine(translation, rotation, zoom, shear)
```

# Notes
- Identity parameters produce identity affine (no transformation)
- The matrix is in homogeneous coordinates: applies rotation/scale/shear then translation
"""
function compose_affine(params::AffineParameters{T}) where T
    return compose_affine(params.translation, params.rotation, params.zoom, params.shear)
end

function compose_affine(
    translation::AbstractArray{T},
    rotation::AbstractArray{T},
    zoom::AbstractArray{T},
    shear::AbstractArray{T}
) where T
    ndim = size(zoom, 1)
    batch_size = size(zoom, 2)

    # Validate shapes
    @assert size(translation) == (ndim, batch_size) "translation shape mismatch"
    @assert size(rotation) == (ndim, ndim, batch_size) "rotation shape mismatch"
    @assert size(shear) == (ndim, batch_size) "shear shape mismatch"

    # Create scale/shear matrix: diagonal embed of zoom with shear off-diagonal
    # This is equivalent to torch.diag_embed(zoom) with shear entries added
    scale_shear = zeros(T, ndim, ndim, batch_size)

    @inbounds for n in 1:batch_size
        # Diagonal: zoom factors
        for i in 1:ndim
            scale_shear[i, i, n] = zoom[i, n]
        end

        # Off-diagonal: shear
        if ndim == 3
            # 3D case: shear has 3 components
            # square_matrix[..., 0, 1:] = shear[..., :2]  -> row 0, cols 1,2
            # square_matrix[..., 1, 2] = shear[..., 2]    -> row 1, col 2
            # Julia: 1-indexed, so row 1 cols 2,3 and row 2 col 3
            scale_shear[1, 2, n] = shear[1, n]  # shear_xy
            scale_shear[1, 3, n] = shear[2, n]  # shear_xz
            scale_shear[2, 3, n] = shear[3, n]  # shear_yz
        else  # ndim == 2
            # 2D case: shear has 1 component (but stored as 2 for consistency)
            # square_matrix[..., 0, 1] = shear[..., 0]
            scale_shear[1, 2, n] = shear[1, n]  # shear_xy
        end
    end

    # Multiply rotation @ scale_shear for each batch element
    # rotation: (ndim, ndim, batch_size)
    # scale_shear: (ndim, ndim, batch_size)
    # result: (ndim, ndim, batch_size)
    rot_scale = zeros(T, ndim, ndim, batch_size)
    @inbounds for n in 1:batch_size
        # Matrix multiplication: rotation[:,:,n] * scale_shear[:,:,n]
        for i in 1:ndim, j in 1:ndim
            for k in 1:ndim
                rot_scale[i, j, n] += rotation[i, k, n] * scale_shear[k, j, n]
            end
        end
    end

    # Concatenate translation as last column
    # affine: (ndim, ndim+1, batch_size)
    affine = zeros(T, ndim, ndim + 1, batch_size)
    @inbounds for n in 1:batch_size
        # Copy rotation @ scale_shear part
        for i in 1:ndim, j in 1:ndim
            affine[i, j, n] = rot_scale[i, j, n]
        end
        # Add translation as last column
        for i in 1:ndim
            affine[i, ndim + 1, n] = translation[i, n]
        end
    end

    return affine
end

"""
    get_affine(reg::AffineRegistration) -> Array

Get the composed affine transformation matrix from a registration object.

# Returns
- Affine matrix of shape `(ndim, ndim+1, batch_size)`

# Notes
- Must be called after `register()` has been run
- Throws error if `reg.parameters` is `nothing`
"""
function get_affine(reg::AffineRegistration)
    if reg.parameters === nothing
        error("No parameters available. Run register() first.")
    end
    return compose_affine(reg.parameters)
end

# ============================================================================
# Affine Transform
# ============================================================================

"""
    affine_transform(x, affine; shape=nothing, padding_mode=:border)

Apply an affine transformation to an image using grid sampling.

# Arguments
- `x`: Input image array
  - 2D: shape `(X, Y, C, N)` where C=channels, N=batch
  - 3D: shape `(X, Y, Z, C, N)`
- `affine`: Affine transformation matrix
  - 2D: shape `(2, 3, N)`
  - 3D: shape `(3, 4, N)`
- `shape`: Optional output spatial shape. If `nothing`, uses input spatial shape.
  - 2D: `(X_out, Y_out)`
  - 3D: `(X_out, Y_out, Z_out)`
- `padding_mode`: How to handle out-of-bounds samples
  - `:border` (default): Use border pixel values
  - `:zeros`: Use zeros for out-of-bounds

# Returns
- Transformed image with same type as input
  - Shape: `(shape..., C, N)` or same as input if shape not specified

# Example
```julia
# Transform 3D image with affine matrix
x = randn(Float32, 64, 64, 64, 1, 1)  # (X, Y, Z, C, N)
affine = identity_affine(3, 1)  # Identity transform
y = affine_transform(x, affine)  # Same as input

# Resize during transform
y_small = affine_transform(x, affine; shape=(32, 32, 32))
```

# Notes
- Uses `affine_grid` to create sampling grid from affine matrix
- Uses `NNlib.grid_sample` for bilinear/trilinear interpolation
- Coordinates are in normalized [-1, 1] range
"""
function affine_transform(
    x::AbstractArray{T, 4},  # 2D: (X, Y, C, N)
    affine::AbstractArray{T, 3};  # (2, 3, N)
    shape::Union{Nothing, NTuple{2, Int}} = nothing,
    padding_mode::Symbol = :border
) where T
    X, Y, C, N = size(x)
    @assert size(affine) == (2, 3, N) "Affine shape mismatch: expected (2, 3, $N), got $(size(affine))"
    @assert padding_mode in (:border, :zeros) "padding_mode must be :border or :zeros"

    # Determine output shape
    out_shape = shape === nothing ? (X, Y) : shape

    # Create sampling grid from affine transformation
    grid = affine_grid(affine, out_shape)  # (2, X_out, Y_out, N)

    # Use NNlib.grid_sample for interpolation
    # NNlib expects grid with coordinates in first dimension
    result = NNlib.grid_sample(x, grid; padding_mode=padding_mode)

    return result
end

function affine_transform(
    x::AbstractArray{T, 5},  # 3D: (X, Y, Z, C, N)
    affine::AbstractArray{T, 3};  # (3, 4, N)
    shape::Union{Nothing, NTuple{3, Int}} = nothing,
    padding_mode::Symbol = :border
) where T
    X, Y, Z, C, N = size(x)
    @assert size(affine) == (3, 4, N) "Affine shape mismatch: expected (3, 4, $N), got $(size(affine))"
    @assert padding_mode in (:border, :zeros) "padding_mode must be :border or :zeros"

    # Determine output shape
    out_shape = shape === nothing ? (X, Y, Z) : shape

    # Create sampling grid from affine transformation
    grid = affine_grid(affine, out_shape)  # (3, X_out, Y_out, Z_out, N)

    # Use NNlib.grid_sample for interpolation
    result = NNlib.grid_sample(x, grid; padding_mode=padding_mode)

    return result
end

"""
    affine_transform(x, affine, reg::AffineRegistration; shape=nothing)

Apply affine transformation using settings from a registration object.

Uses the `padding_mode` from the registration configuration.
"""
function affine_transform(
    x::AbstractArray{T},
    affine::AbstractArray{T, 3},
    reg::AffineRegistration;
    shape = nothing
) where T
    return affine_transform(x, affine; shape=shape, padding_mode=reg.padding_mode)
end
