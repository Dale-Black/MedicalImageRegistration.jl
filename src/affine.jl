# Affine registration implementation

using Optimisers: Adam

# Note: _constant() is defined in utils.jl

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

    # Build scale/shear matrix in a Zygote-compatible way
    # We construct the matrix elements explicitly and use cat/stack
    if ndim == 3
        # 3D case: scale_shear is a 3x3 upper triangular matrix
        # [zoom_x  shear_xy  shear_xz]
        # [  0     zoom_y    shear_yz]
        # [  0       0       zoom_z  ]
        z_zeros = _constant() do
            zeros(T, 1, batch_size)
        end

        # Row 1: [zoom[1], shear[1], shear[2]]
        row1 = cat(
            reshape(zoom[1, :], 1, 1, batch_size),
            reshape(shear[1, :], 1, 1, batch_size),
            reshape(shear[2, :], 1, 1, batch_size),
            dims=2
        )  # (1, 3, batch_size)

        # Row 2: [0, zoom[2], shear[3]]
        row2 = cat(
            reshape(z_zeros, 1, 1, batch_size),
            reshape(zoom[2, :], 1, 1, batch_size),
            reshape(shear[3, :], 1, 1, batch_size),
            dims=2
        )  # (1, 3, batch_size)

        # Row 3: [0, 0, zoom[3]]
        row3 = cat(
            reshape(z_zeros, 1, 1, batch_size),
            reshape(z_zeros, 1, 1, batch_size),
            reshape(zoom[3, :], 1, 1, batch_size),
            dims=2
        )  # (1, 3, batch_size)

        scale_shear = cat(row1, row2, row3, dims=1)  # (3, 3, batch_size)
    else
        # 2D case: scale_shear is a 2x2 upper triangular matrix
        # [zoom_x  shear_xy]
        # [  0     zoom_y  ]
        z_zeros = _constant() do
            zeros(T, 1, batch_size)
        end

        # Row 1: [zoom[1], shear[1]]
        row1 = cat(
            reshape(zoom[1, :], 1, 1, batch_size),
            reshape(shear[1, :], 1, 1, batch_size),
            dims=2
        )  # (1, 2, batch_size)

        # Row 2: [0, zoom[2]]
        row2 = cat(
            reshape(z_zeros, 1, 1, batch_size),
            reshape(zoom[2, :], 1, 1, batch_size),
            dims=2
        )  # (1, 2, batch_size)

        scale_shear = cat(row1, row2, dims=1)  # (2, 2, batch_size)
    end

    # Batched matrix multiply: rotation @ scale_shear
    # rotation: (ndim, ndim, batch_size)
    # scale_shear: (ndim, ndim, batch_size)
    # Use NNlib.batched_mul for Zygote compatibility
    rot_scale = NNlib.batched_mul(rotation, scale_shear)  # (ndim, ndim, batch_size)

    # Concatenate translation as last column
    # translation: (ndim, batch_size) -> reshape to (ndim, 1, batch_size)
    trans_col = reshape(translation, ndim, 1, batch_size)

    # affine = [rot_scale | translation]
    affine = cat(rot_scale, trans_col, dims=2)  # (ndim, ndim+1, batch_size)

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

# ============================================================================
# Image Resizing for Multi-resolution Pyramid
# ============================================================================

"""
    downsample(x::AbstractArray{T, 4}, scale::Int) where T

Downsample a 2D image by the given scale factor using bilinear interpolation.

Input shape: `(X, Y, C, N)` → Output: `(X÷scale, Y÷scale, C, N)`
"""
function downsample(x::AbstractArray{T, 4}, scale::Int) where T
    X, Y, C, N = size(x)
    new_X = max(1, X ÷ scale)
    new_Y = max(1, Y ÷ scale)

    # Create identity affine for resampling
    affine = identity_affine(2, N, T)

    return affine_transform(x, affine; shape=(new_X, new_Y), padding_mode=:border)
end

"""
    downsample(x::AbstractArray{T, 5}, scale::Int) where T

Downsample a 3D image by the given scale factor using trilinear interpolation.

Input shape: `(X, Y, Z, C, N)` → Output: `(X÷scale, Y÷scale, Z÷scale, C, N)`
"""
function downsample(x::AbstractArray{T, 5}, scale::Int) where T
    X, Y, Z, C, N = size(x)
    new_X = max(1, X ÷ scale)
    new_Y = max(1, Y ÷ scale)
    new_Z = max(1, Z ÷ scale)

    # Create identity affine for resampling
    affine = identity_affine(3, N, T)

    return affine_transform(x, affine; shape=(new_X, new_Y, new_Z), padding_mode=:border)
end

# ============================================================================
# Manual Gradient Computation Helpers
# ============================================================================

"""
    _affine_grid_backward(d_grid, spatial_size, batch_size, ::Type{T}) where T

Compute gradient of affine_grid with respect to theta (affine matrix).

Given d_grid (gradient w.r.t. output grid), computes d_theta.

affine_grid computes: output = theta @ homogeneous_coords
So the gradient is: d_theta = d_output @ homogeneous_coords^T
"""
function _affine_grid_backward(
    d_grid::AbstractArray{T},
    spatial_size::NTuple{N, Int},
    batch_size::Int
) where {T, N}
    ndim = N
    num_points = prod(spatial_size)

    # Create identity grid and homogeneous coordinates
    id_grid = create_identity_grid(spatial_size, T)
    flat_grid = reshape(id_grid, ndim, num_points)
    ones_row = ones(T, 1, num_points)
    homogeneous_grid = vcat(flat_grid, ones_row)  # (ndim+1, num_points)

    # d_grid is (ndim, ...spatial..., batch_size)
    d_grid_flat = reshape(d_grid, ndim, num_points, batch_size)

    # d_theta = d_grid_flat @ homogeneous_grid^T for each batch
    d_theta = zeros(T, ndim, ndim + 1, batch_size)
    @inbounds for n in 1:batch_size
        d_theta[:, :, n] = d_grid_flat[:, :, n] * homogeneous_grid'
    end

    return d_theta
end

"""
    _compose_affine_backward_3d(d_affine, rotation, zoom, shear)

Compute gradients of compose_affine with respect to translation, rotation, zoom, shear.

For 3D case: affine = [rotation @ scale_shear | translation]
where scale_shear is upper triangular with zoom on diagonal and shear off-diagonal.
"""
function _compose_affine_backward_3d(
    d_affine::AbstractArray{T, 3},
    rotation::AbstractArray{T, 3},
    zoom::AbstractArray{T, 2},
    shear::AbstractArray{T, 2}
) where T
    batch_size = size(zoom, 2)

    # Reconstruct scale_shear matrix
    scale_shear = zeros(T, 3, 3, batch_size)
    @inbounds for n in 1:batch_size
        scale_shear[1, 1, n] = zoom[1, n]
        scale_shear[1, 2, n] = shear[1, n]
        scale_shear[1, 3, n] = shear[2, n]
        scale_shear[2, 2, n] = zoom[2, n]
        scale_shear[2, 3, n] = shear[3, n]
        scale_shear[3, 3, n] = zoom[3, n]
    end

    # d_translation = d_affine[:, 4, :] (last column)
    d_translation = d_affine[:, 4, :]

    # d_rot_scale = d_affine[:, 1:3, :] (first 3 columns)
    d_rot_scale = d_affine[:, 1:3, :]

    # rot_scale = rotation @ scale_shear
    # d_rotation = d_rot_scale @ scale_shear^T
    # d_scale_shear = rotation^T @ d_rot_scale
    d_rotation = NNlib.batched_mul(d_rot_scale, NNlib.batched_transpose(scale_shear))
    d_scale_shear = NNlib.batched_mul(NNlib.batched_transpose(rotation), d_rot_scale)

    # Extract d_zoom and d_shear from d_scale_shear
    d_zoom = zeros(T, 3, batch_size)
    d_shear = zeros(T, 3, batch_size)
    @inbounds for n in 1:batch_size
        d_zoom[1, n] = d_scale_shear[1, 1, n]
        d_zoom[2, n] = d_scale_shear[2, 2, n]
        d_zoom[3, n] = d_scale_shear[3, 3, n]
        d_shear[1, n] = d_scale_shear[1, 2, n]
        d_shear[2, n] = d_scale_shear[1, 3, n]
        d_shear[3, n] = d_scale_shear[2, 3, n]
    end

    return d_translation, d_rotation, d_zoom, d_shear
end

"""
    _compose_affine_backward_2d(d_affine, rotation, zoom, shear)

Compute gradients of compose_affine with respect to translation, rotation, zoom, shear.

For 2D case: affine = [rotation @ scale_shear | translation]
"""
function _compose_affine_backward_2d(
    d_affine::AbstractArray{T, 3},
    rotation::AbstractArray{T, 3},
    zoom::AbstractArray{T, 2},
    shear::AbstractArray{T, 2}
) where T
    batch_size = size(zoom, 2)

    # Reconstruct scale_shear matrix for 2D
    scale_shear = zeros(T, 2, 2, batch_size)
    @inbounds for n in 1:batch_size
        scale_shear[1, 1, n] = zoom[1, n]
        scale_shear[1, 2, n] = shear[1, n]
        scale_shear[2, 2, n] = zoom[2, n]
    end

    # d_translation = d_affine[:, 3, :] (last column for 2D)
    d_translation = d_affine[:, 3, :]

    # d_rot_scale = d_affine[:, 1:2, :] (first 2 columns)
    d_rot_scale = d_affine[:, 1:2, :]

    # Backprop through matrix multiply
    d_rotation = NNlib.batched_mul(d_rot_scale, NNlib.batched_transpose(scale_shear))
    d_scale_shear = NNlib.batched_mul(NNlib.batched_transpose(rotation), d_rot_scale)

    # Extract d_zoom and d_shear
    d_zoom = zeros(T, 2, batch_size)
    d_shear = zeros(T, 2, batch_size)
    @inbounds for n in 1:batch_size
        d_zoom[1, n] = d_scale_shear[1, 1, n]
        d_zoom[2, n] = d_scale_shear[2, 2, n]
        d_shear[1, n] = d_scale_shear[1, 2, n]
    end

    return d_translation, d_rotation, d_zoom, d_shear
end

"""
    _compute_affine_gradients(moving, static, translation, rotation, zoom, shear,
                              target_shape, padding_mode, dissimilarity_fn)

Compute loss and gradients for affine registration using manual backpropagation.

Returns: (loss_value, d_translation, d_rotation, d_zoom, d_shear)
"""
function _compute_affine_gradients(
    moving::AbstractArray{T},
    static::AbstractArray{T},
    translation::AbstractArray{T, 2},
    rotation::AbstractArray{T, 3},
    zoom::AbstractArray{T, 2},
    shear::AbstractArray{T, 2},
    target_shape,
    padding_mode::Symbol,
    dissimilarity_fn
) where T
    ndim = size(zoom, 1)
    batch_size = size(zoom, 2)

    # ========== Forward Pass ==========
    # 1. Compose affine matrix
    affine = compose_affine(translation, rotation, zoom, shear)

    # 2. Create sampling grid
    grid = affine_grid(affine, target_shape)

    # 3. Sample moving image at grid positions
    moved = NNlib.grid_sample(moving, grid; padding_mode=padding_mode)

    # 4. Compute loss (MSE for now, which has simple gradient)
    # For MSE: loss = mean((moved - static)^2)
    diff = moved .- static
    loss_val = mean(diff .^ 2)

    # ========== Backward Pass ==========
    # 1. Gradient of MSE loss: d_moved = 2 * (moved - static) / numel
    d_moved = T(2) .* diff ./ T(length(moved))

    # 2. Gradient through grid_sample using NNlib.∇grid_sample
    _, d_grid = NNlib.∇grid_sample(d_moved, moving, grid; padding_mode=padding_mode)

    # 3. Gradient through affine_grid
    d_affine = _affine_grid_backward(d_grid, target_shape, batch_size)

    # 4. Gradient through compose_affine
    if ndim == 3
        d_translation, d_rotation, d_zoom, d_shear = _compose_affine_backward_3d(
            d_affine, rotation, zoom, shear
        )
    else
        d_translation, d_rotation, d_zoom, d_shear = _compose_affine_backward_2d(
            d_affine, rotation, zoom, shear
        )
    end

    return loss_val, d_translation, d_rotation, d_zoom, d_shear
end

# ============================================================================
# Optimization Loop
# ============================================================================

"""
    fit!(reg::AffineRegistration, moving, static, iterations::Int; verbose=nothing)

Run optimization for a single resolution level.

# Arguments
- `reg`: AffineRegistration with parameters already initialized
- `moving`: Moving image to transform, shape `(...spatial, C, N)`
- `static`: Static target image, same shape as moving
- `iterations`: Number of optimization iterations

# Notes
- Modifies `reg.parameters` in place
- Uses manual gradient computation (no AD library required)
- Uses Optimisers.jl for parameter updates
"""
function fit!(
    reg::AffineRegistration{T},
    moving::AbstractArray{T},
    static::AbstractArray{T},
    iterations::Int;
    verbose::Union{Nothing, Bool} = nothing
) where T
    verbose = verbose === nothing ? reg.verbose : verbose
    params = reg.parameters

    # Get spatial shape of static image
    ndim = ndims(static) - 2  # Remove C and N dimensions
    target_shape = NTuple{ndim, Int}(size(static)[1:ndim])

    # Create parameter tuple based on what we're optimizing
    param_tuple = (
        params.translation,
        params.rotation,
        params.zoom,
        params.shear
    )

    # Setup optimizer
    opt_rule = reg.optimizer(reg.learning_rate)
    opt_state = Optimisers.setup(opt_rule, param_tuple)

    # Optimization loop
    local loss_val
    for iter in 1:iterations
        # Unpack current parameters
        t, r, z, s = param_tuple

        # Compute loss and gradients using manual backpropagation
        loss_val, d_t, d_r, d_z, d_s = _compute_affine_gradients(
            moving, static, t, r, z, s,
            target_shape, reg.padding_mode, reg.dissimilarity_fn
        )

        # Zero out gradients for parameters we're not optimizing
        grad_translation = reg.with_translation ? d_t : zero(d_t)
        grad_rotation = reg.with_rotation ? d_r : zero(d_r)
        grad_zoom = reg.with_zoom ? d_z : zero(d_z)
        grad_shear = reg.with_shear ? d_s : zero(d_s)

        masked_grads = (grad_translation, grad_rotation, grad_zoom, grad_shear)

        # Update parameters
        opt_state, param_tuple = Optimisers.update!(opt_state, param_tuple, masked_grads)

        # Display progress
        if verbose && (iter == 1 || iter % 50 == 0 || iter == iterations)
            println("  Iteration $iter/$iterations: loss = $(round(loss_val; digits=6))")
        end
    end

    # Store updated parameters back
    reg.parameters = AffineParameters{T}(
        param_tuple[1],
        param_tuple[2],
        param_tuple[3],
        param_tuple[4]
    )
    reg.loss = T(loss_val)

    return loss_val
end

"""
    register(moving, static, reg::AffineRegistration; return_moved=true)

Perform affine registration of moving image to static image.

# Arguments
- `moving`: Moving image to register, shape `(...spatial, C, N)`
- `static`: Static target image, same shape convention
- `reg`: AffineRegistration configuration object
- `return_moved`: If true, return the transformed moving image

# Returns
- If `return_moved=true`: Transformed moving image
- If `return_moved=false`: `nothing` (use `transform(image, reg)` to apply later)

# Notes
- Initializes parameters, runs multi-resolution optimization
- After completion, `reg.parameters` contains learned transformation
- Use `get_affine(reg)` to retrieve the affine matrix
- Use `transform(image, reg)` to apply transformation to other images

# Example
```julia
reg = AffineRegistration(ndims=3, scales=(4, 2), iterations=(200, 100))
moved = register(moving, static, reg)

# Apply same transform to another image
other_moved = transform(other_image, reg)
```
"""
function register(
    moving::AbstractArray{T},
    static::AbstractArray{T},
    reg::AffineRegistration{T};
    return_moved::Bool = true
) where T
    # Validate input shapes
    @assert ndims(moving) == ndims(static) "moving and static must have same number of dimensions"
    expected_ndims = reg.ndims + 2  # spatial + C + N
    @assert ndims(moving) == expected_ndims "Expected $(expected_ndims)D arrays for ndims=$(reg.ndims), got $(ndims(moving))D"

    # Get batch size from static image (last dimension)
    batch_size = size(static)[end]

    # Initialize parameters
    reg.parameters = init_parameters(reg, batch_size)

    # Get target spatial shape
    ndim = reg.ndims
    target_shape = size(static)[1:ndim]

    # Resize moving to match static spatial dimensions if needed
    moving_shape = size(moving)[1:ndim]
    if moving_shape != target_shape
        moving_affine = identity_affine(ndim, batch_size, T)
        moving = affine_transform(moving, moving_affine; shape=target_shape, padding_mode=reg.padding_mode)
    end

    # Multi-resolution optimization
    if reg.verbose
        println("Starting affine registration...")
        println("  Target shape: $target_shape")
        println("  Scales: $(reg.scales)")
        println("  Iterations: $(reg.iterations)")
    end

    for (scale, iters) in zip(reg.scales, reg.iterations)
        if reg.verbose
            println("Scale 1/$scale:")
        end

        # Downsample images
        if scale > 1
            moving_small = downsample(moving, scale)
            static_small = downsample(static, scale)
        else
            moving_small = moving
            static_small = static
        end

        # Run optimization at this scale
        fit!(reg, moving_small, static_small, iters)
    end

    if reg.verbose
        println("Registration complete. Final loss: $(reg.loss)")
    end

    # Return transformed image if requested
    if return_moved
        return transform(moving, reg)
    else
        return nothing
    end
end

"""
    transform(x, reg::AffineRegistration; shape=nothing)

Apply learned transformation to an image.

# Arguments
- `x`: Image to transform
- `reg`: AffineRegistration with learned parameters
- `shape`: Optional output shape (uses input shape if not specified)

# Returns
- Transformed image

# Notes
- Must be called after `register()` has been run
"""
function transform(
    x::AbstractArray{T},
    reg::AffineRegistration{T};
    shape = nothing
) where T
    if reg.parameters === nothing
        error("No parameters available. Run register() first.")
    end

    affine = compose_affine(reg.parameters)
    return affine_transform(x, affine; shape=shape, padding_mode=reg.padding_mode)
end
