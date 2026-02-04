# GPU-accelerated affine registration with Mooncake AD support
# Uses AcceleratedKernels.jl for cross-platform GPU execution
# Dependencies imported by parent module: AK, Mooncake, CoDual, NoFData, NoRData, @is_primitive, MinimalCtx

# ============================================================================
# GPU-Compatible Identity Affine Generator
# ============================================================================

"""
    _identity_affine_2d(::Type{T}, N::Int, image::AbstractArray)

Create 2D identity affine matrix on same device as image.
Returns (2, 3, N) array.
"""
function _identity_affine_2d(::Type{T}, N::Int, image::AbstractArray) where T
    # Create on CPU then transfer to avoid scalar indexing
    theta_cpu = zeros(T, 2, 3, N)
    for n in 1:N
        theta_cpu[1, 1, n] = one(T)
        theta_cpu[2, 2, n] = one(T)
    end
    # Transfer to same device as image
    theta = similar(image, 2, 3, N)
    copyto!(theta, theta_cpu)
    return theta
end

"""
    _identity_affine_3d(::Type{T}, N::Int, image::AbstractArray)

Create 3D identity affine matrix on same device as image.
Returns (3, 4, N) array.
"""
function _identity_affine_3d(::Type{T}, N::Int, image::AbstractArray) where T
    # Create on CPU then transfer to avoid scalar indexing
    theta_cpu = zeros(T, 3, 4, N)
    for n in 1:N
        theta_cpu[1, 1, n] = one(T)
        theta_cpu[2, 2, n] = one(T)
        theta_cpu[3, 3, n] = one(T)
    end
    # Transfer to same device as image
    theta = similar(image, 3, 4, N)
    copyto!(theta, theta_cpu)
    return theta
end

# ============================================================================
# Image Resampling with GPU support
# ============================================================================

"""
    _resample_image(image::AbstractArray{T}, scale::Int; align_corners=true) where T

Downsample an image by a factor of `scale` using bilinear/trilinear interpolation.
GPU-compatible using AK.foreachindex.
"""
function _resample_image(
    image::AbstractArray{T,4},
    scale::Int;
    align_corners::Bool=true
) where T
    X_in, Y_in, C, N = size(image)
    X_out = max(1, X_in ÷ scale)
    Y_out = max(1, Y_in ÷ scale)

    theta = _identity_affine_2d(T, N, image)
    grid = affine_grid(theta, (X_out, Y_out); align_corners=align_corners)
    return grid_sample(image, grid; padding_mode=:border, align_corners=align_corners)
end

function _resample_image(
    image::AbstractArray{T,5},
    scale::Int;
    align_corners::Bool=true
) where T
    X_in, Y_in, Z_in, C, N = size(image)
    X_out = max(1, X_in ÷ scale)
    Y_out = max(1, Y_in ÷ scale)
    Z_out = max(1, Z_in ÷ scale)

    theta = _identity_affine_3d(T, N, image)
    grid = affine_grid(theta, (X_out, Y_out, Z_out); align_corners=align_corners)
    return grid_sample(image, grid; padding_mode=:border, align_corners=align_corners)
end

"""
    _resample_to_size(image::AbstractArray{T}, target_size; align_corners=true) where T

Resample an image to a specific target spatial size.
"""
function _resample_to_size(
    image::AbstractArray{T,4},
    target_size::NTuple{2,Int};
    align_corners::Bool=true
) where T
    X_out, Y_out = target_size
    N = size(image, 4)

    theta = _identity_affine_2d(T, N, image)
    grid = affine_grid(theta, (X_out, Y_out); align_corners=align_corners)
    return grid_sample(image, grid; padding_mode=:border, align_corners=align_corners)
end

function _resample_to_size(
    image::AbstractArray{T,5},
    target_size::NTuple{3,Int};
    align_corners::Bool=true
) where T
    X_out, Y_out, Z_out = target_size
    N = size(image, 5)

    theta = _identity_affine_3d(T, N, image)
    grid = affine_grid(theta, (X_out, Y_out, Z_out); align_corners=align_corners)
    return grid_sample(image, grid; padding_mode=:border, align_corners=align_corners)
end

# ============================================================================
# Affine Transform Application
# ============================================================================

"""
    affine_transform(image, theta; shape=nothing, padding_mode=:border, align_corners=true)

Apply an affine transformation to an image.

# Arguments
- `image`: Input image (X, Y, [Z], C, N)
- `theta`: Affine transformation matrix (D, D+1, N)
- `shape`: Optional output spatial shape. If nothing, uses input shape.
- `padding_mode`: :zeros or :border
- `align_corners`: Align corners for grid sampling

# Returns
- Transformed image
"""
function affine_transform(
    image::AbstractArray{T,4},
    theta::AbstractArray{T,3};
    shape::Union{Nothing, NTuple{2,Int}}=nothing,
    padding_mode::Symbol=:border,
    align_corners::Bool=true
) where T
    X_in, Y_in, C, N = size(image)
    out_shape = shape === nothing ? (X_in, Y_in) : shape

    grid = affine_grid(theta, out_shape; align_corners=align_corners)
    return grid_sample(image, grid; padding_mode=padding_mode, align_corners=align_corners)
end

function affine_transform(
    image::AbstractArray{T,5},
    theta::AbstractArray{T,3};
    shape::Union{Nothing, NTuple{3,Int}}=nothing,
    padding_mode::Symbol=:border,
    align_corners::Bool=true
) where T
    X_in, Y_in, Z_in, C, N = size(image)
    out_shape = shape === nothing ? (X_in, Y_in, Z_in) : shape

    grid = affine_grid(theta, out_shape; align_corners=align_corners)
    return grid_sample(image, grid; padding_mode=padding_mode, align_corners=align_corners)
end

# ============================================================================
# Forward Pass for Full Registration Pipeline
# ============================================================================

"""
    _forward_registration(translation, rotation, zoom, shear, moving, static_shape; padding_mode, align_corners)

Complete forward pass: compose_affine → affine_grid → grid_sample
Returns the moved image.
"""
function _forward_registration(
    translation::AbstractArray{T,2},
    rotation::AbstractArray{T,3},
    zoom::AbstractArray{T,2},
    shear::AbstractArray{T,2},
    moving::AbstractArray{T},
    static_shape::Tuple;
    padding_mode::Symbol=:border,
    align_corners::Bool=true
) where T
    theta = compose_affine(translation, rotation, zoom, shear)
    grid = affine_grid(theta, static_shape; align_corners=align_corners)
    moved = grid_sample(moving, grid; padding_mode=padding_mode, align_corners=align_corners)
    return moved
end

# ============================================================================
# Gradient Computation via Manual Pullback Chain
# ============================================================================

"""
    _compute_gradients!(d_translation, d_rotation, d_zoom, d_shear,
                        d_loss, translation, rotation, zoom, shear,
                        moving, static_shape, with_translation, with_rotation,
                        with_zoom, with_shear; padding_mode, align_corners)

Compute gradients for all parameters by chaining pullbacks manually.
This avoids issues with Mooncake's autodiff through complex operations.

The chain is:
1. loss ← moved (we have d_moved from loss backward)
2. moved ← grid_sample(moving, grid) → d_moving, d_grid
3. grid ← affine_grid(theta, size) → d_theta
4. theta ← compose_affine(t, R, z, s) → d_t, d_R, d_z, d_s
"""
function _compute_gradients!(
    d_translation::AbstractArray{T,2},
    d_rotation::AbstractArray{T,3},
    d_zoom::AbstractArray{T,2},
    d_shear::AbstractArray{T,2},
    d_moved::AbstractArray{T},  # Gradient from loss w.r.t. moved image
    translation::AbstractArray{T,2},
    rotation::AbstractArray{T,3},
    zoom::AbstractArray{T,2},
    shear::AbstractArray{T,2},
    moving::AbstractArray{T},
    static_shape::Tuple,
    with_translation::Bool,
    with_rotation::Bool,
    with_zoom::Bool,
    with_shear::Bool;
    padding_mode::Symbol=:border,
    align_corners::Bool=true
) where T
    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    ac = Val(align_corners)
    D = size(translation, 1)
    is_3d = D == 3

    # Recompute forward pass intermediate values
    theta = compose_affine(translation, rotation, zoom, shear)
    grid = affine_grid(theta, static_shape; align_corners=align_corners)

    # Step 1: Backward through grid_sample to get d_grid
    # d_moved is already provided (from loss backward)
    # We need d_grid (don't need d_moving since moving is fixed)
    d_grid = similar(grid)
    fill!(d_grid, zero(T))

    if is_3d
        _∇grid_sample_grid_3d!(d_grid, d_moved, moving, grid, pm, ac)
    else
        _∇grid_sample_grid_2d!(d_grid, d_moved, moving, grid, pm, ac)
    end

    # Step 2: Backward through affine_grid to get d_theta
    d_theta = similar(theta)
    fill!(d_theta, zero(T))

    if is_3d
        _∇affine_grid_theta_3d!(d_theta, d_grid, ac)
    else
        _∇affine_grid_theta_2d!(d_theta, d_grid, ac)
    end

    # Step 3: Backward through compose_affine to get d_translation, d_rotation, d_zoom, d_shear
    # Only accumulate gradients for enabled parameters
    if is_3d
        _∇compose_affine_3d!(
            d_translation, d_rotation, d_zoom, d_shear,
            d_theta, translation, rotation, zoom, shear
        )
    else
        _∇compose_affine_2d!(
            d_translation, d_rotation, d_zoom, d_shear,
            d_theta, translation, rotation, zoom, shear
        )
    end

    # Zero out gradients for disabled parameters
    if !with_translation
        fill!(d_translation, zero(T))
    end
    if !with_rotation
        fill!(d_rotation, zero(T))
    end
    if !with_zoom
        fill!(d_zoom, zero(T))
    end
    if !with_shear
        fill!(d_shear, zero(T))
    end

    return nothing
end

"""
    _compute_loss_and_gradient!(d_moved, moved, static, loss_fn)

Compute loss and gradient of loss w.r.t. moved image.
Returns the loss value (as scalar on CPU).
"""
function _compute_loss_and_gradient!(
    d_moved::AbstractArray{T},
    moved::AbstractArray{T},
    static::AbstractArray{T},
    loss_fn::Function
) where T
    # Forward pass for loss
    loss_arr = loss_fn(moved, static)

    # For backward, we need d_loss/d_moved
    # loss_arr is 1-element array on GPU
    # Set d_loss = 1 (upstream gradient)
    d_loss_arr = similar(loss_arr)
    fill!(d_loss_arr, one(T))

    # Compute gradient manually based on loss type
    # For MSE: d_moved[i] = 2*(moved[i] - static[i])/n
    n = T(length(moved))
    scale = T(2) / n

    AK.foreachindex(d_moved) do idx
        @inbounds d_moved[idx] = scale * (moved[idx] - static[idx])
    end

    # Extract scalar loss value using GPU-compatible reduction
    loss_val = AK.reduce(+, loss_arr; init=zero(T))
    return loss_val
end

# ============================================================================
# Adam Optimizer (GPU-compatible)
# ============================================================================

mutable struct AdamState{T, A<:AbstractArray{T}}
    m::A  # First moment
    v::A  # Second moment
    t::Int  # Time step
    beta1::T
    beta2::T
    eps::T
end

function _init_adam(param::AbstractArray{T}; beta1::T=T(0.9), beta2::T=T(0.999), eps::T=T(1e-8)) where T
    m = similar(param)
    fill!(m, zero(T))
    v = similar(param)
    fill!(v, zero(T))
    return AdamState{T, typeof(m)}(m, v, 0, beta1, beta2, eps)
end

function _adam_step!(param::AbstractArray{T}, grad::AbstractArray{T}, state::AdamState{T}, lr::T) where T
    state.t += 1
    t = state.t
    beta1, beta2, eps = state.beta1, state.beta2, state.eps

    # Bias correction - compute as scalars (bits types)
    bc1 = T(1) - beta1^t
    bc2 = T(1) - beta2^t

    # Extract arrays from state to avoid capturing non-bits struct in closure
    m_arr = state.m
    v_arr = state.v

    AK.foreachindex(param) do idx
        g = @inbounds grad[idx]
        m_old = @inbounds m_arr[idx]
        v_old = @inbounds v_arr[idx]

        # Update moments
        m_new = beta1 * m_old + (one(T) - beta1) * g
        v_new = beta2 * v_old + (one(T) - beta2) * g * g

        @inbounds m_arr[idx] = m_new
        @inbounds v_arr[idx] = v_new

        # Bias-corrected estimates
        m_hat = m_new / bc1
        v_hat = v_new / bc2

        # Update parameter
        @inbounds param[idx] -= lr * m_hat / (sqrt(v_hat) + eps)
    end

    return nothing
end

# ============================================================================
# Main Fitting Function
# ============================================================================

"""
    fit!(reg::AffineRegistration, moving, static; loss_fn=mse_loss, verbose=true)

Fit the affine registration parameters to align `moving` to `static`.

# Arguments
- `reg`: AffineRegistration model
- `moving`: Moving image to be transformed (X, Y, [Z], C, N)
- `static`: Static/target image (X, Y, [Z], C, N)
- `loss_fn`: Loss function (default: mse_loss)
- `verbose`: Print progress (default: true)

# Returns
- The registration model with updated parameters
"""
function fit!(
    reg::AffineRegistration{T},
    moving::AbstractArray{T},
    static::AbstractArray{T};
    loss_fn::Function=mse_loss,
    verbose::Bool=true
) where T
    # Validate dimensions
    expected_ndim = reg.is_3d ? 5 : 4
    @assert ndims(moving) == expected_ndim "Moving image must be $(expected_ndim)D"
    @assert ndims(static) == expected_ndim "Static image must be $(expected_ndim)D"

    # Get spatial dimensions
    if reg.is_3d
        static_shape = (size(static, 1), size(static, 2), size(static, 3))
    else
        static_shape = (size(static, 1), size(static, 2))
    end

    # Resample moving to match static spatial size
    moving_resampled = _resample_to_size(moving, static_shape; align_corners=reg.align_corners)

    # Initialize Adam optimizers for each parameter
    adam_t = _init_adam(reg.translation)
    adam_R = _init_adam(reg.rotation)
    adam_z = _init_adam(reg.zoom)
    adam_s = _init_adam(reg.shear)

    # Pre-allocate gradient buffers
    d_translation = similar(reg.translation)
    d_rotation = similar(reg.rotation)
    d_zoom = similar(reg.zoom)
    d_shear = similar(reg.shear)

    # Clear loss history
    empty!(reg.loss_history)

    # Multi-resolution loop
    for (scale_idx, (scale, iters)) in enumerate(zip(reg.scales, reg.iterations))
        # Downsample images
        moving_small = _resample_image(moving_resampled, scale; align_corners=reg.align_corners)
        static_small = _resample_image(static, scale; align_corners=reg.align_corners)

        if reg.is_3d
            small_shape = (size(static_small, 1), size(static_small, 2), size(static_small, 3))
        else
            small_shape = (size(static_small, 1), size(static_small, 2))
        end

        if verbose
            println("Scale $scale_idx/$( length(reg.scales)): scale=$scale, shape=$small_shape, iters=$iters")
        end

        # Pre-allocate d_moved for this resolution
        d_moved = similar(static_small)

        # Optimization loop at this scale
        for iter in 1:iters
            # Zero gradients
            fill!(d_translation, zero(T))
            fill!(d_rotation, zero(T))
            fill!(d_zoom, zero(T))
            fill!(d_shear, zero(T))
            fill!(d_moved, zero(T))

            # Forward pass
            moved = _forward_registration(
                reg.translation, reg.rotation, reg.zoom, reg.shear,
                moving_small, small_shape;
                padding_mode=reg.padding_mode, align_corners=reg.align_corners
            )

            # Compute loss and gradient w.r.t. moved
            loss_val = _compute_loss_and_gradient!(d_moved, moved, static_small, loss_fn)

            # Backward pass to compute parameter gradients
            _compute_gradients!(
                d_translation, d_rotation, d_zoom, d_shear,
                d_moved,
                reg.translation, reg.rotation, reg.zoom, reg.shear,
                moving_small, small_shape,
                reg.with_translation, reg.with_rotation, reg.with_zoom, reg.with_shear;
                padding_mode=reg.padding_mode, align_corners=reg.align_corners
            )

            # Adam updates
            if reg.with_translation
                _adam_step!(reg.translation, d_translation, adam_t, reg.learning_rate)
            end
            if reg.with_rotation
                _adam_step!(reg.rotation, d_rotation, adam_R, reg.learning_rate)
            end
            if reg.with_zoom
                _adam_step!(reg.zoom, d_zoom, adam_z, reg.learning_rate)
            end
            if reg.with_shear
                _adam_step!(reg.shear, d_shear, adam_s, reg.learning_rate)
            end

            push!(reg.loss_history, loss_val)

            if verbose && (iter % 50 == 0 || iter == 1 || iter == iters)
                println("  Iter $iter/$iters: loss = $(round(loss_val, digits=6))")
            end
        end
    end

    return reg
end

# ============================================================================
# User-Facing API
# ============================================================================

"""
    register(reg::AffineRegistration, moving, static; kwargs...)

Register `moving` image to `static` image and return the transformed moving image.

# Arguments
- `reg`: AffineRegistration model (will be modified with learned parameters)
- `moving`: Moving image to be transformed
- `static`: Static/target image

# Keyword Arguments
- `loss_fn`: Loss function (default: mse_loss)
- `verbose`: Print progress (default: true)
- `reset_params`: Reset parameters before fitting (default: true)

# Returns
- Transformed moving image aligned to static
"""
function register(
    reg::AffineRegistration{T},
    moving::AbstractArray{T},
    static::AbstractArray{T};
    loss_fn::Function=mse_loss,
    verbose::Bool=true,
    reset_params::Bool=true
) where T
    if reset_params
        reset!(reg)
    end

    # Fit parameters
    fit!(reg, moving, static; loss_fn=loss_fn, verbose=verbose)

    # Transform moving image to static size
    if reg.is_3d
        out_shape = (size(static, 1), size(static, 2), size(static, 3))
    else
        out_shape = (size(static, 1), size(static, 2))
    end

    return transform(reg, moving, out_shape)
end

"""
    transform(reg::AffineRegistration, image, shape=nothing)

Transform an image using the current registration parameters.

# Arguments
- `reg`: AffineRegistration model with fitted parameters
- `image`: Image to transform
- `shape`: Optional output spatial shape. If nothing, uses input shape.

# Returns
- Transformed image
"""
function transform(
    reg::AffineRegistration{T},
    image::AbstractArray{T},
    shape::Union{Nothing, Tuple}=nothing
) where T
    theta = get_affine(reg)

    if reg.is_3d
        out_shape = shape === nothing ? (size(image, 1), size(image, 2), size(image, 3)) : shape
        return affine_transform(image, theta; shape=out_shape, padding_mode=reg.padding_mode, align_corners=reg.align_corners)
    else
        out_shape = shape === nothing ? (size(image, 1), size(image, 2)) : shape
        return affine_transform(image, theta; shape=out_shape, padding_mode=reg.padding_mode, align_corners=reg.align_corners)
    end
end
