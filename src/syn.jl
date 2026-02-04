# GPU-accelerated SyN diffeomorphic registration with Mooncake AD support
# Uses AcceleratedKernels.jl for cross-platform GPU execution
# Dependencies imported by parent module: AK, Mooncake, CoDual, NoFData, NoRData, @is_primitive, MinimalCtx, Atomix

# ============================================================================
# Julia Array Convention for SyN: (X, Y, Z, D, N) for velocity/displacement fields
# - D = spatial dimension (3 for 3D)
# - N = batch size
# - Velocity field v: (X, Y, Z, 3, N) stores (vx, vy, vz) displacement at each point
#
# PyTorch torchreg uses: (N, D, Z, Y, X) which is (N, 3, Z, Y, X)
# ============================================================================

# ============================================================================
# Identity Grid Creation (for displacement field application)
# ============================================================================

"""
    _create_identity_grid_3d(shape, reference::AbstractArray{T}) where T

Create a 3D identity grid of normalized coordinates in [-1, 1].
Returns grid of shape (3, X, Y, Z, N) where N=1.
Uses same device as reference array.
"""
function _create_identity_grid_3d(shape::NTuple{3,Int}, reference::AbstractArray{T}) where T
    X, Y, Z = shape
    N = 1

    # Create on CPU then transfer
    grid_cpu = zeros(T, 3, X, Y, Z, N)

    for k in 1:Z, j in 1:Y, i in 1:X
        # Normalized coordinates in [-1, 1]
        x_norm = T(2 * (i - 1) / max(X - 1, 1) - 1)
        y_norm = T(2 * (j - 1) / max(Y - 1, 1) - 1)
        z_norm = T(2 * (k - 1) / max(Z - 1, 1) - 1)

        grid_cpu[1, i, j, k, 1] = x_norm
        grid_cpu[2, i, j, k, 1] = y_norm
        grid_cpu[3, i, j, k, 1] = z_norm
    end

    # Transfer to same device as reference
    grid = similar(reference, 3, X, Y, Z, N)
    copyto!(grid, grid_cpu)
    return grid
end

# ============================================================================
# Spatial Transform (Warp image with displacement field)
# ============================================================================

"""
    spatial_transform(x::AbstractArray{T,5}, v::AbstractArray{T,5}; align_corners=true, padding_mode=:border, interpolation=:bilinear) where T

Warp image x using displacement field v.
The output at position p is sampled from input at position (p + v[p]).

# Arguments
- `x`: Input image of shape (X, Y, Z, C, N)
- `v`: Displacement field of shape (X, Y, Z, D, N) where D=3
- `align_corners`: Grid sampling alignment
- `padding_mode`: Padding mode for out-of-bounds (:zeros, :border)
- `interpolation`: Interpolation mode (:bilinear/:trilinear default, :nearest for HU preservation)

# Returns
- Warped image of same shape as x
"""
function spatial_transform(
    x::AbstractArray{T,5},
    v::AbstractArray{T,5};
    align_corners::Bool=true,
    padding_mode::Symbol=:border,
    interpolation::Symbol=:bilinear
) where T
    X, Y, Z, C, N = size(x)
    X_v, Y_v, Z_v, D, N_v = size(v)

    @assert (X, Y, Z) == (X_v, Y_v, Z_v) "Spatial dimensions must match"
    @assert D == 3 "Displacement field must have 3 channels"
    @assert N == N_v "Batch sizes must match"

    # Create identity grid
    id_grid = _create_identity_grid_3d((X, Y, Z), v)

    # For each batch, add displacement to identity grid
    # Output grid shape: (3, X, Y, Z, N)
    grid = similar(v, 3, X, Y, Z, N)

    # Permute v from (X, Y, Z, D, N) to (D, X, Y, Z, N) format
    # v_perm[d, i, j, k, n] = v[i, j, k, d, n]
    v_perm = similar(v, 3, X, Y, Z, N)
    _permute_velocity_to_grid!(v_perm, v)

    # Combine: grid = identity + displacement
    _add_displacement_to_grid!(grid, id_grid, v_perm)

    # Use grid_sample with the combined grid
    return grid_sample(x, grid; padding_mode=padding_mode, align_corners=align_corners, interpolation=interpolation)
end

"""
    _permute_velocity_to_grid!(v_perm, v)

Permute velocity from (X, Y, Z, D, N) to (D, X, Y, Z, N) format.
GPU-compatible using AK.foreachindex.
"""
function _permute_velocity_to_grid!(
    v_perm::AbstractArray{T,5},
    v::AbstractArray{T,5}
) where T
    X, Y, Z, D, N = size(v)

    AK.foreachindex(v) do idx
        i, j, k, d, n = _linear_to_cartesian_5d_syn(idx, X, Y, Z, D)
        @inbounds v_perm[d, i, j, k, n] = v[i, j, k, d, n]
    end

    return nothing
end

"""
    _add_displacement_to_grid!(grid, id_grid, v_perm)

Add displacement to identity grid: grid = id_grid + v_perm.
Broadcasts identity grid across batch dimension.
"""
function _add_displacement_to_grid!(
    grid::AbstractArray{T,5},
    id_grid::AbstractArray{T,5},
    v_perm::AbstractArray{T,5}
) where T
    _, X, Y, Z, N = size(grid)

    AK.foreachindex(grid) do idx
        d, i, j, k, n = _linear_to_cartesian_5d_grid(idx, X, Y, Z, N)
        # id_grid has N=1, so always use batch index 1
        @inbounds grid[idx] = id_grid[d, i, j, k, 1] + v_perm[d, i, j, k, n]
    end

    return nothing
end

# Index conversion helpers for SyN arrays
@inline function _linear_to_cartesian_5d_syn(idx::Int, X::Int, Y::Int, Z::Int, D::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    idx_0 = idx_0 ÷ Z
    d = idx_0 % D + 1
    n = idx_0 ÷ D + 1
    return i, j, k, d, n
end

@inline function _linear_to_cartesian_5d_grid(idx::Int, X::Int, Y::Int, Z::Int, N::Int)
    idx_0 = idx - 1
    d = idx_0 % 3 + 1  # D is always 3 for 3D
    idx_0 = idx_0 ÷ 3
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    n = idx_0 ÷ Z + 1
    return d, i, j, k, n
end

# ============================================================================
# Diffeomorphic Transform (Scaling and Squaring)
# ============================================================================

"""
    diffeomorphic_transform(v::AbstractArray{T,5}; time_steps::Int=7) where T

Convert velocity field to diffeomorphic displacement using scaling-and-squaring.

The velocity field is first scaled by 2^(-time_steps), then repeatedly composed
with itself `time_steps` times to produce a diffeomorphic deformation.

# Arguments
- `v`: Velocity field of shape (X, Y, Z, 3, N)
- `time_steps`: Number of scaling-and-squaring steps (default: 7)

# Returns
- Diffeomorphic displacement field of same shape as v
"""
function diffeomorphic_transform(
    v::AbstractArray{T,5};
    time_steps::Int=7,
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    # Scale velocity by 2^(-time_steps)
    scale = T(2)^(-time_steps)
    v_scaled = v .* scale

    # Repeated composition: v = v + warp(v, v)
    result = v_scaled
    for _ in 1:time_steps
        result = composition_transform(result, result; align_corners=align_corners, padding_mode=padding_mode)
    end

    return result
end

# ============================================================================
# Composition Transform
# ============================================================================

"""
    composition_transform(v1::AbstractArray{T,5}, v2::AbstractArray{T,5}; align_corners=true, padding_mode=:border) where T

Compose two displacement fields: result[p] = v2[p] + v1[p + v2[p]]

This warps v1 by v2, then adds v2.

# Arguments
- `v1`: First displacement field (X, Y, Z, 3, N)
- `v2`: Second displacement field (X, Y, Z, 3, N)

# Returns
- Composed displacement field
"""
function composition_transform(
    v1::AbstractArray{T,5},
    v2::AbstractArray{T,5};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    # Warp v1 by v2: sample v1 at positions (p + v2[p])
    v1_warped = _warp_displacement(v1, v2; align_corners=align_corners, padding_mode=padding_mode)

    # Add v2: result = v2 + v1_warped
    return v2 .+ v1_warped
end

"""
    _warp_displacement(v1, v2; align_corners, padding_mode)

Warp displacement field v1 by displacement field v2.
Samples v1 at positions (p + v2[p]).
"""
function _warp_displacement(
    v1::AbstractArray{T,5},
    v2::AbstractArray{T,5};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    X, Y, Z, D, N = size(v1)
    @assert D == 3 "Displacement must have 3 channels"

    # Create identity grid
    id_grid = _create_identity_grid_3d((X, Y, Z), v1)

    # Permute v2 and add to identity grid
    v2_perm = similar(v2, 3, X, Y, Z, N)
    _permute_velocity_to_grid!(v2_perm, v2)

    grid = similar(v2, 3, X, Y, Z, N)
    _add_displacement_to_grid!(grid, id_grid, v2_perm)

    # Sample v1 at grid positions - need to reshape v1 for grid_sample
    # v1 is (X, Y, Z, D, N), treat D as channels for grid_sample
    # grid_sample expects input (X, Y, Z, C, N), grid (3, X, Y, Z, N)
    v1_warped = grid_sample(v1, grid; padding_mode=padding_mode, align_corners=align_corners)

    return v1_warped
end

# ============================================================================
# Gaussian Smoothing (for velocity field regularization)
# ============================================================================

"""
    gauss_smoothing(x::AbstractArray{T,5}, sigma::Union{T, AbstractVector{T}}; kernel_size::Union{Nothing, Tuple}=nothing) where T

Apply Gaussian smoothing to a 5D array.
GPU-compatible implementation using separable convolution.

# Arguments
- `x`: Input array of shape (X, Y, Z, C, N)
- `sigma`: Standard deviation (scalar or 3-element vector for x, y, z)
- `kernel_size`: Optional kernel size tuple. If nothing, computed from sigma.

# Returns
- Smoothed array of same shape
"""
function gauss_smoothing(
    x::AbstractArray{T,5},
    sigma::T;
    kernel_size::Union{Nothing, NTuple{3,Int}}=nothing
) where T
    # Convert scalar sigma to vector
    sigma_vec = T[sigma, sigma, sigma]
    return gauss_smoothing(x, sigma_vec; kernel_size=kernel_size)
end

function gauss_smoothing(
    x::AbstractArray{T,5},
    sigma::AbstractVector{T};
    kernel_size::Union{Nothing, NTuple{3,Int}}=nothing
) where T
    X, Y, Z, C, N = size(x)

    # Skip if sigma is too small
    if all(s -> s < T(0.1), sigma)
        return copy(x)
    end

    # Compute kernel size if not provided
    # Following torchreg: kernel_size = 1 + 2 * max(1, shape // 50)
    if kernel_size === nothing
        half_k = max.(1, (X, Y, Z) .÷ 50)
        kernel_size = 1 .+ 2 .* half_k
    end
    kx, ky, kz = kernel_size

    # Apply separable convolution along each axis
    result = _gauss_smooth_axis(x, sigma[1], kx, 1)  # X axis
    result = _gauss_smooth_axis(result, sigma[2], ky, 2)  # Y axis
    result = _gauss_smooth_axis(result, sigma[3], kz, 3)  # Z axis

    return result
end

"""
    _gauss_smooth_axis(x, sigma, kernel_size, axis)

Apply 1D Gaussian smoothing along a single axis.
Uses GPU-compatible sliding window.
"""
function _gauss_smooth_axis(
    x::AbstractArray{T,5},
    sigma::T,
    kernel_size::Int,
    axis::Int
) where T
    if sigma < T(0.1)
        return copy(x)
    end

    X, Y, Z, C, N = size(x)
    output = similar(x)

    # Compute 1D Gaussian kernel on CPU then transfer
    half_k = kernel_size ÷ 2
    kernel_cpu = zeros(T, kernel_size)
    for i in 1:kernel_size
        dist = T(i - 1 - half_k)
        kernel_cpu[i] = exp(-dist^2 / (2 * sigma^2))
    end
    kernel_sum = sum(kernel_cpu)
    kernel_cpu ./= kernel_sum

    # Transfer kernel to GPU
    kernel_arr = similar(x, kernel_size)
    copyto!(kernel_arr, kernel_cpu)

    # Apply convolution along specified axis using AK.foreachindex
    _apply_1d_conv!(output, x, kernel_arr, half_k, axis)

    return output
end

"""
    _apply_1d_conv!(output, x, kernel, half_k, axis)

Apply 1D convolution along specified axis with replicate padding.
"""
function _apply_1d_conv!(
    output::AbstractArray{T,5},
    x::AbstractArray{T,5},
    kernel::AbstractArray{T,1},
    half_k::Int,
    axis::Int
) where T
    X, Y, Z, C, N = size(x)
    kernel_size = length(kernel)

    AK.foreachindex(output) do idx
        i, j, k, c, n = _linear_to_cartesian_5d_syn(idx, X, Y, Z, C)

        acc = zero(T)
        for kk in 1:kernel_size
            offset = kk - 1 - half_k

            # Get position along axis with replicate padding
            if axis == 1
                ii = clamp(i + offset, 1, X)
                @inbounds acc += kernel[kk] * x[ii, j, k, c, n]
            elseif axis == 2
                jj = clamp(j + offset, 1, Y)
                @inbounds acc += kernel[kk] * x[i, jj, k, c, n]
            else  # axis == 3
                kk_idx = clamp(k + offset, 1, Z)
                @inbounds acc += kernel[kk] * x[i, j, kk_idx, c, n]
            end
        end

        @inbounds output[idx] = acc
    end

    return nothing
end

# ============================================================================
# Linear Elasticity Regularization
# ============================================================================

"""
    linear_elasticity(flow::AbstractArray{T,5}; mu::T=T(2), lam::T=T(1)) where T

Compute linear elasticity regularization loss for a flow field.

Penalizes non-smooth deformations based on strain tensor.

# Arguments
- `flow`: Displacement/flow field of shape (X, Y, Z, 3, N)
- `mu`: First Lamé parameter (shear modulus), default 2
- `lam`: Second Lamé parameter (bulk modulus), default 1

# Returns
- Scalar regularization loss (as 1-element array for GPU compatibility)
"""
function linear_elasticity(
    flow::AbstractArray{T,5};
    mu::T=T(2),
    lam::T=T(1)
) where T
    X, Y, Z, D, N = size(flow)
    @assert D == 3 "Flow must have 3 channels"

    # Compute spatial gradients using finite differences
    # ∂u/∂x, ∂u/∂y, ∂u/∂z for each component u, v, w

    # Allocate output
    loss_arr = similar(flow, 1)
    fill!(loss_arr, zero(T))

    # Compute sum of squared strain tensor components
    _compute_linear_elasticity!(loss_arr, flow, mu, lam)

    # Average over all spatial positions
    n_elements = T(X * Y * Z * N)
    scale_arr = similar(loss_arr)
    fill!(scale_arr, one(T) / n_elements)

    result = similar(loss_arr)
    AK.foreachindex(result) do idx
        @inbounds result[idx] = loss_arr[idx] * scale_arr[idx]
    end

    return result
end

"""
    _compute_linear_elasticity!(loss_arr, flow, mu, lam)

Compute linear elasticity loss using GPU-compatible parallel reduction.
"""
function _compute_linear_elasticity!(
    loss_arr::AbstractArray{T,1},
    flow::AbstractArray{T,5},
    mu::T,
    lam::T
) where T
    X, Y, Z, D, N = size(flow)

    # Create temporary array for per-element losses
    elem_losses = similar(flow, X, Y, Z, N)

    AK.foreachindex(elem_losses) do idx
        i, j, k, n = _linear_to_cartesian_4d_syn(idx, X, Y, Z)

        # Skip boundary where we can't compute gradients
        if i < 2 || i > X-1 || j < 2 || j > Y-1 || k < 2 || k > Z-1
            @inbounds elem_losses[idx] = zero(T)
            return nothing
        end

        # Central differences for gradients
        # u_x = ∂u/∂x, etc.
        @inbounds begin
            u_x = (flow[i+1, j, k, 1, n] - flow[i-1, j, k, 1, n]) / 2
            u_y = (flow[i, j+1, k, 1, n] - flow[i, j-1, k, 1, n]) / 2
            u_z = (flow[i, j, k+1, 1, n] - flow[i, j, k-1, 1, n]) / 2

            v_x = (flow[i+1, j, k, 2, n] - flow[i-1, j, k, 2, n]) / 2
            v_y = (flow[i, j+1, k, 2, n] - flow[i, j-1, k, 2, n]) / 2
            v_z = (flow[i, j, k+1, 2, n] - flow[i, j, k-1, 2, n]) / 2

            w_x = (flow[i+1, j, k, 3, n] - flow[i-1, j, k, 3, n]) / 2
            w_y = (flow[i, j+1, k, 3, n] - flow[i, j-1, k, 3, n]) / 2
            w_z = (flow[i, j, k+1, 3, n] - flow[i, j, k-1, 3, n]) / 2
        end

        # Strain tensor components
        e_xx = u_x
        e_yy = v_y
        e_zz = w_z
        e_xy = T(0.5) * (u_y + v_x)
        e_xz = T(0.5) * (u_z + w_x)
        e_yz = T(0.5) * (v_z + w_y)

        # Trace of strain tensor
        trace = e_xx + e_yy + e_zz

        # Stress tensor (linear elastic)
        sigma_xx = 2 * mu * e_xx + lam * trace
        sigma_yy = 2 * mu * e_yy + lam * trace
        sigma_zz = 2 * mu * e_zz + lam * trace
        sigma_xy = 2 * mu * e_xy
        sigma_xz = 2 * mu * e_xz
        sigma_yz = 2 * mu * e_yz

        # Strain energy density
        energy = sigma_xx^2 + sigma_yy^2 + sigma_zz^2 +
                 sigma_xy^2 + sigma_xz^2 + sigma_yz^2

        @inbounds elem_losses[idx] = energy
    end

    # Sum all element losses using AK.reduce
    total_loss = AK.reduce(+, elem_losses; init=zero(T))

    # Store in loss_arr
    loss_arr_cpu = T[total_loss]
    copyto!(loss_arr, loss_arr_cpu)

    return nothing
end

@inline function _linear_to_cartesian_4d_syn(idx::Int, X::Int, Y::Int, Z::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    n = idx_0 ÷ Z + 1
    return i, j, k, n
end

# ============================================================================
# Backward Passes (Gradients)
# ============================================================================

# Gradient for spatial_transform w.r.t. displacement field v
function _∇spatial_transform_v!(
    d_v::AbstractArray{T,5},
    d_output::AbstractArray{T,5},
    x::AbstractArray{T,5},
    v::AbstractArray{T,5};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    X, Y, Z, C, N = size(x)

    # Create identity grid and combined grid
    id_grid = _create_identity_grid_3d((X, Y, Z), v)
    v_perm = similar(v, 3, X, Y, Z, N)
    _permute_velocity_to_grid!(v_perm, v)
    grid = similar(v, 3, X, Y, Z, N)
    _add_displacement_to_grid!(grid, id_grid, v_perm)

    # Get gradient w.r.t. grid from grid_sample backward pass
    d_grid = similar(grid)
    fill!(d_grid, zero(T))

    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    ac = Val(align_corners)
    _∇grid_sample_grid_3d!(d_grid, d_output, x, grid, pm, ac)

    # d_v_perm = d_grid (since grid = id_grid + v_perm and id_grid is constant)
    # Permute back from (3, X, Y, Z, N) to (X, Y, Z, 3, N)
    _permute_grid_to_velocity!(d_v, d_grid)

    return nothing
end

"""
    _permute_grid_to_velocity!(v, grid)

Permute from (D, X, Y, Z, N) to (X, Y, Z, D, N) format.
"""
function _permute_grid_to_velocity!(
    v::AbstractArray{T,5},
    grid::AbstractArray{T,5}
) where T
    _, X, Y, Z, N = size(grid)

    AK.foreachindex(v) do idx
        i, j, k, d, n = _linear_to_cartesian_5d_syn(idx, X, Y, Z, 3)
        @inbounds v[idx] = grid[d, i, j, k, n]
    end

    return nothing
end

# Gradient for diffeomorphic_transform
# This is complex due to the iterative nature - we need to unroll and backprop through each step
function _∇diffeomorphic_transform!(
    d_v::AbstractArray{T,5},
    d_output::AbstractArray{T,5},
    v::AbstractArray{T,5};
    time_steps::Int=7,
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    # Forward pass cache
    scale = T(2)^(-time_steps)
    v_scaled = v .* scale

    # Store intermediate results
    results = Vector{typeof(v)}(undef, time_steps + 1)
    results[1] = v_scaled

    current = v_scaled
    for i in 1:time_steps
        current = composition_transform(current, current; align_corners=align_corners, padding_mode=padding_mode)
        results[i + 1] = current
    end

    # Backward pass through each step
    d_current = copy(d_output)

    for i in time_steps:-1:1
        prev_result = results[i]

        # composition_transform: result = v2 + warp(v1, v2)
        # Here v1 = v2 = prev_result
        # d_v2 += d_result
        # d_v1 += warp_grad(d_result)
        # d_v2 += warp_v2_grad(d_result, v1)

        d_prev = similar(prev_result)
        fill!(d_prev, zero(T))

        _∇composition_transform!(d_prev, d_current, prev_result, prev_result;
                                  align_corners=align_corners, padding_mode=padding_mode)

        d_current = d_prev
    end

    # Scale gradient
    d_v .= d_current .* scale

    return nothing
end

# Gradient for composition_transform
function _∇composition_transform!(
    d_v::AbstractArray{T,5},  # Gradient w.r.t. input v (used for both v1 and v2)
    d_output::AbstractArray{T,5},
    v1::AbstractArray{T,5},
    v2::AbstractArray{T,5};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    # result = v2 + warp(v1, v2)

    # d_v2 from direct addition
    d_v .+= d_output

    # d_v1 and d_v2 from warp(v1, v2)
    d_v1_warp = similar(v1)
    d_v2_warp = similar(v2)
    fill!(d_v1_warp, zero(T))
    fill!(d_v2_warp, zero(T))

    _∇warp_displacement!(d_v1_warp, d_v2_warp, d_output, v1, v2;
                          align_corners=align_corners, padding_mode=padding_mode)

    # Accumulate (since v1 = v2 in diffeomorphic case)
    d_v .+= d_v1_warp .+ d_v2_warp

    return nothing
end

# Gradient for warp_displacement
function _∇warp_displacement!(
    d_v1::AbstractArray{T,5},
    d_v2::AbstractArray{T,5},
    d_output::AbstractArray{T,5},
    v1::AbstractArray{T,5},
    v2::AbstractArray{T,5};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    X, Y, Z, D, N = size(v1)

    # Create grid from v2
    id_grid = _create_identity_grid_3d((X, Y, Z), v1)
    v2_perm = similar(v2, 3, X, Y, Z, N)
    _permute_velocity_to_grid!(v2_perm, v2)
    grid = similar(v2, 3, X, Y, Z, N)
    _add_displacement_to_grid!(grid, id_grid, v2_perm)

    # Gradient w.r.t. v1 (input to grid_sample)
    d_v1_raw = similar(v1)
    fill!(d_v1_raw, zero(T))

    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    ac = Val(align_corners)
    _∇grid_sample_input_3d!(d_v1_raw, d_output, grid, pm, ac)
    d_v1 .= d_v1_raw

    # Gradient w.r.t. grid, then propagate to v2
    d_grid = similar(grid)
    fill!(d_grid, zero(T))
    _∇grid_sample_grid_3d!(d_grid, d_output, v1, grid, pm, ac)

    # Permute d_grid back to d_v2
    _permute_grid_to_velocity!(d_v2, d_grid)

    return nothing
end

# ============================================================================
# Mooncake rrule!! Definitions
# ============================================================================

# Mark as primitives
@is_primitive MinimalCtx Tuple{typeof(spatial_transform), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}
@is_primitive MinimalCtx Tuple{typeof(diffeomorphic_transform), AbstractArray{<:Any,5}}
@is_primitive MinimalCtx Tuple{typeof(composition_transform), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}
@is_primitive MinimalCtx Tuple{typeof(linear_elasticity), AbstractArray{<:Any,5}}

# rrule!! for spatial_transform
function Mooncake.rrule!!(
    ::CoDual{typeof(spatial_transform)},
    x::CoDual{A1, F1},
    v::CoDual{A2, F2};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where {A1<:AbstractArray{<:Any,5}, F1, A2<:AbstractArray{<:Any,5}, F2}
    x_primal = x.x
    x_fdata = x.dx
    v_primal = v.x
    v_fdata = v.dx

    output = spatial_transform(x_primal, v_primal; align_corners=align_corners, padding_mode=padding_mode)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function spatial_transform_pullback(_rdata)
        # Gradient w.r.t. x (input image)
        pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
        ac = Val(align_corners)

        X, Y, Z, C, N = size(x_primal)
        id_grid = _create_identity_grid_3d((X, Y, Z), v_primal)
        v_perm = similar(v_primal, 3, X, Y, Z, N)
        _permute_velocity_to_grid!(v_perm, v_primal)
        grid = similar(v_primal, 3, X, Y, Z, N)
        _add_displacement_to_grid!(grid, id_grid, v_perm)

        _∇grid_sample_input_3d!(x_fdata, output_fdata, grid, pm, ac)

        # Gradient w.r.t. v
        _∇spatial_transform_v!(v_fdata, output_fdata, x_primal, v_primal;
                               align_corners=align_corners, padding_mode=padding_mode)

        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), spatial_transform_pullback
end

# rrule!! for diffeomorphic_transform
function Mooncake.rrule!!(
    ::CoDual{typeof(diffeomorphic_transform)},
    v::CoDual{A, F};
    time_steps::Int=7,
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where {A<:AbstractArray{<:Any,5}, F}
    v_primal = v.x
    v_fdata = v.dx

    output = diffeomorphic_transform(v_primal; time_steps=time_steps, align_corners=align_corners, padding_mode=padding_mode)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function diffeomorphic_transform_pullback(_rdata)
        _∇diffeomorphic_transform!(v_fdata, output_fdata, v_primal;
                                    time_steps=time_steps, align_corners=align_corners, padding_mode=padding_mode)
        return NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), diffeomorphic_transform_pullback
end

# rrule!! for composition_transform
function Mooncake.rrule!!(
    ::CoDual{typeof(composition_transform)},
    v1::CoDual{A1, F1},
    v2::CoDual{A2, F2};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where {A1<:AbstractArray{<:Any,5}, F1, A2<:AbstractArray{<:Any,5}, F2}
    v1_primal = v1.x
    v1_fdata = v1.dx
    v2_primal = v2.x
    v2_fdata = v2.dx

    output = composition_transform(v1_primal, v2_primal; align_corners=align_corners, padding_mode=padding_mode)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function composition_transform_pullback(_rdata)
        # result = v2 + warp(v1, v2)

        # d_v2 from direct addition
        v2_fdata .+= output_fdata

        # d_v1 and d_v2 from warp
        _∇warp_displacement!(v1_fdata, v2_fdata, output_fdata, v1_primal, v2_primal;
                              align_corners=align_corners, padding_mode=padding_mode)

        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), composition_transform_pullback
end

# rrule!! for linear_elasticity
function Mooncake.rrule!!(
    ::CoDual{typeof(linear_elasticity)},
    flow::CoDual{A, F};
    mu::Real=2.0,
    lam::Real=1.0
) where {A<:AbstractArray{<:Any,5}, F}
    flow_primal = flow.x
    flow_fdata = flow.dx
    T = eltype(flow_primal)

    output = linear_elasticity(flow_primal; mu=T(mu), lam=T(lam))
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function linear_elasticity_pullback(_rdata)
        # Gradient of linear elasticity is complex - use finite differences for now
        # This is a placeholder - should be implemented properly for production use
        _∇linear_elasticity!(flow_fdata, output_fdata, flow_primal, T(mu), T(lam))
        return NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), linear_elasticity_pullback
end

# Gradient for linear_elasticity (simplified version using numerical differentiation concept)
function _∇linear_elasticity!(
    d_flow::AbstractArray{T,5},
    d_output::AbstractArray{T,1},
    flow::AbstractArray{T,5},
    mu::T,
    lam::T
) where T
    X, Y, Z, D, N = size(flow)
    n_elements = T(X * Y * Z * N)

    # Extract scalar gradient
    d_out_scalar = AK.reduce(+, d_output; init=zero(T))
    scale = d_out_scalar / n_elements

    AK.foreachindex(d_flow) do idx
        i, j, k, d, n = _linear_to_cartesian_5d_syn(idx, X, Y, Z, D)

        # Skip boundary
        if i < 2 || i > X-1 || j < 2 || j > Y-1 || k < 2 || k > Z-1
            return nothing
        end

        # Compute gradient via chain rule through strain/stress tensors
        # This is a simplified approximation
        @inbounds begin
            # Get neighboring values for finite difference
            if d == 1  # u component
                grad = (flow[i+1, j, k, 1, n] + flow[i-1, j, k, 1, n] - 2*flow[i, j, k, 1, n]) * 2 * (2*mu + lam) +
                       (flow[i, j+1, k, 1, n] + flow[i, j-1, k, 1, n] - 2*flow[i, j, k, 1, n]) * mu +
                       (flow[i, j, k+1, 1, n] + flow[i, j, k-1, 1, n] - 2*flow[i, j, k, 1, n]) * mu
            elseif d == 2  # v component
                grad = (flow[i, j+1, k, 2, n] + flow[i, j-1, k, 2, n] - 2*flow[i, j, k, 2, n]) * 2 * (2*mu + lam) +
                       (flow[i+1, j, k, 2, n] + flow[i-1, j, k, 2, n] - 2*flow[i, j, k, 2, n]) * mu +
                       (flow[i, j, k+1, 2, n] + flow[i, j, k-1, 2, n] - 2*flow[i, j, k, 2, n]) * mu
            else  # w component
                grad = (flow[i, j, k+1, 3, n] + flow[i, j, k-1, 3, n] - 2*flow[i, j, k, 3, n]) * 2 * (2*mu + lam) +
                       (flow[i+1, j, k, 3, n] + flow[i-1, j, k, 3, n] - 2*flow[i, j, k, 3, n]) * mu +
                       (flow[i, j+1, k, 3, n] + flow[i, j-1, k, 3, n] - 2*flow[i, j, k, 3, n]) * mu
            end

            d_flow[idx] += scale * grad
        end
    end

    return nothing
end

# ============================================================================
# SyNRegistration Struct and API
# ============================================================================

"""
    SyNRegistration{T, A<:AbstractArray{T}}

Holds configuration and learned parameters for SyN diffeomorphic registration.

# Fields
- `scales`: Tuple of scales for multi-resolution pyramid
- `iterations`: Number of iterations at each scale
- `learning_rates`: Learning rates for each scale
- `time_steps`: Number of scaling-and-squaring steps
- `sigma_img`: Sigma for image smoothing
- `sigma_flow`: Sigma for flow smoothing
- `lambda_`: Regularization weight
- `align_corners`: Grid sampling alignment
- `padding_mode`: Padding mode for sampling
- `v_xy`: Velocity field from moving to static (X, Y, Z, 3, N)
- `v_yx`: Velocity field from static to moving (X, Y, Z, 3, N)
- `loss_history`: Loss values during training

# Example
```julia
using Metal

reg = SyNRegistration{Float32}(
    scales=(4, 2, 1),
    iterations=(30, 30, 10),
    array_type=MtlArray
)

moved = register(reg, moving, static)
```
"""
mutable struct SyNRegistration{T, A<:AbstractArray{T}}
    # Configuration
    scales::Tuple{Vararg{Int}}
    iterations::Tuple{Vararg{Int}}
    learning_rates::Tuple{Vararg{T}}
    time_steps::Int
    sigma_img::T
    sigma_flow::T
    lambda_::T
    align_corners::Bool
    padding_mode::Symbol
    verbose::Bool

    # Velocity fields
    v_xy::Union{Nothing, A}  # Forward velocity field
    v_yx::Union{Nothing, A}  # Inverse velocity field

    # Array type for initialization
    array_type::Type{<:AbstractArray}

    # Training history
    loss_history::Vector{T}
end

"""
    SyNRegistration{T}(; kwargs...) where T

Create a new SyNRegistration with default parameters.

# Keyword Arguments
- `scales::Tuple=(4, 2, 1)`: Multi-resolution pyramid scales
- `iterations::Tuple=(30, 30, 10)`: Iterations per scale
- `learning_rate::T=T(1e-2)`: Optimizer learning rate (scalar or tuple)
- `time_steps::Int=7`: Scaling-and-squaring steps
- `sigma_img::T=T(0.2)`: Image smoothing sigma
- `sigma_flow::T=T(0.2)`: Flow smoothing sigma
- `lambda_::T=T(2e-5)`: Regularization weight
- `align_corners::Bool=true`: Grid sampling alignment
- `padding_mode::Symbol=:border`: Padding mode
- `verbose::Bool=true`: Print progress
- `array_type::Type=Array`: Array type (Array, MtlArray, CuArray)
"""
function SyNRegistration{T}(;
    scales::Tuple{Vararg{Int}}=(4, 2, 1),
    iterations::Tuple{Vararg{Int}}=(30, 30, 10),
    learning_rate::Union{T, Tuple{Vararg{T}}}=T(1e-2),
    time_steps::Int=7,
    sigma_img::T=T(0.2),
    sigma_flow::T=T(0.2),
    lambda_::T=T(2e-5),
    align_corners::Bool=true,
    padding_mode::Symbol=:border,
    verbose::Bool=true,
    array_type::Type{<:AbstractArray}=Array
) where T
    # Convert scalar learning rate to tuple
    if learning_rate isa T
        learning_rates = ntuple(_ -> learning_rate, length(scales))
    else
        learning_rates = learning_rate
    end

    return SyNRegistration{T, array_type{T,5}}(
        scales, iterations, learning_rates, time_steps,
        sigma_img, sigma_flow, lambda_,
        align_corners, padding_mode, verbose,
        nothing, nothing, array_type,
        T[]
    )
end

# Convenience constructor
function SyNRegistration(; T::Type{<:AbstractFloat}=Float32, kwargs...)
    return SyNRegistration{T}(; kwargs...)
end

"""
    reset!(reg::SyNRegistration{T}) where T

Reset velocity fields to zero.
"""
function reset!(reg::SyNRegistration{T}) where T
    reg.v_xy = nothing
    reg.v_yx = nothing
    empty!(reg.loss_history)
    return reg
end

"""
    _init_velocity_fields!(reg, shape, reference)

Initialize velocity fields to zero on the appropriate device.
"""
function _init_velocity_fields!(
    reg::SyNRegistration{T},
    shape::NTuple{3,Int},
    reference::AbstractArray{T}
) where T
    X, Y, Z = shape
    N = size(reference, 5)  # Batch size

    # Initialize on CPU then transfer
    v_cpu = zeros(T, X, Y, Z, 3, N)

    reg.v_xy = similar(reference, X, Y, Z, 3, N)
    copyto!(reg.v_xy, v_cpu)

    reg.v_yx = similar(reference, X, Y, Z, 3, N)
    copyto!(reg.v_yx, v_cpu)

    return nothing
end

"""
    apply_flows(reg, x, y, v_xy, v_yx; interpolation=:bilinear)

Apply forward and inverse flows to compute warped images and full flows.
Returns (images, flows) dictionaries.

# Arguments
- `interpolation`: Interpolation mode for final warped images (:bilinear default, :nearest for HU preservation)
"""
function apply_flows(
    reg::SyNRegistration{T},
    x::AbstractArray{T,5},  # Moving
    y::AbstractArray{T,5},  # Static
    v_xy::AbstractArray{T,5},
    v_yx::AbstractArray{T,5};
    interpolation::Symbol=:bilinear
) where T
    # Compute half flows via diffeomorphic transform
    # Stack all 4 velocity fields for efficient processing
    v_all = cat(v_xy, v_yx, -v_xy, -v_yx; dims=5)
    half_flows = diffeomorphic_transform(v_all; time_steps=reg.time_steps,
                                          align_corners=reg.align_corners,
                                          padding_mode=reg.padding_mode)

    # Split back
    N = size(x, 5)
    flow_xy_half = half_flows[:, :, :, :, 1:N]
    flow_yx_half = half_flows[:, :, :, :, N+1:2N]
    flow_xy_inv_half = half_flows[:, :, :, :, 2N+1:3N]
    flow_yx_inv_half = half_flows[:, :, :, :, 3N+1:4N]

    # Half-way images (always use bilinear for intermediate computations)
    xy_half = spatial_transform(x, flow_xy_half; align_corners=reg.align_corners, padding_mode=reg.padding_mode, interpolation=:bilinear)
    yx_half = spatial_transform(y, flow_yx_half; align_corners=reg.align_corners, padding_mode=reg.padding_mode, interpolation=:bilinear)

    # Full flows via composition
    # full_xy = compose(half_xy, -half_yx)
    flow_xy_full = composition_transform(flow_xy_half, flow_yx_inv_half;
                                          align_corners=reg.align_corners,
                                          padding_mode=reg.padding_mode)
    flow_yx_full = composition_transform(flow_yx_half, flow_xy_inv_half;
                                          align_corners=reg.align_corners,
                                          padding_mode=reg.padding_mode)

    # Full warped images (use specified interpolation mode)
    xy_full = spatial_transform(x, flow_xy_full; align_corners=reg.align_corners, padding_mode=reg.padding_mode, interpolation=interpolation)
    yx_full = spatial_transform(y, flow_yx_full; align_corners=reg.align_corners, padding_mode=reg.padding_mode, interpolation=interpolation)

    images = Dict(
        :xy_half => xy_half, :yx_half => yx_half,
        :xy_full => xy_full, :yx_full => yx_full
    )

    flows = Dict(
        :xy_half => flow_xy_half, :yx_half => flow_yx_half,
        :xy_full => flow_xy_full, :yx_full => flow_yx_full
    )

    return images, flows
end

"""
    fit!(reg::SyNRegistration, moving, static; loss_fn=mse_loss, verbose=true)

Fit the SyN registration parameters.
"""
function fit!(
    reg::SyNRegistration{T},
    moving::AbstractArray{T,5},
    static::AbstractArray{T,5};
    loss_fn::Function=mse_loss
) where T
    X_s, Y_s, Z_s, C, N = size(static)
    static_shape = (X_s, Y_s, Z_s)

    # Initialize velocity fields if not already done
    if reg.v_xy === nothing
        _init_velocity_fields!(reg, static_shape, static)
    end

    # Clear loss history
    empty!(reg.loss_history)

    # Multi-resolution loop
    for (scale_idx, (scale, iters, lr)) in enumerate(zip(reg.scales, reg.iterations, reg.learning_rates))
        # Compute target shape at this scale
        shape = map(s -> max(1, s ÷ scale), static_shape)

        if reg.verbose
            println("Scale $scale_idx/$(length(reg.scales)): scale=$scale, shape=$shape, iters=$iters")
        end

        # Resample images to this scale
        x_small = _resample_to_size(moving, shape; align_corners=reg.align_corners)
        y_small = _resample_to_size(static, shape; align_corners=reg.align_corners)

        # Resample velocity fields to this scale
        v_xy = _resample_velocity(reg.v_xy, shape, moving)
        v_yx = _resample_velocity(reg.v_yx, shape, static)

        # Image smoothing
        if reg.sigma_img > zero(T)
            # Scale sigma based on shape
            sigma_scaled = reg.sigma_img * T(200) / T(minimum(shape))
            x_small = gauss_smoothing(x_small, sigma_scaled)
            y_small = gauss_smoothing(y_small, sigma_scaled)
        end

        # Adam optimizers
        adam_xy = _init_adam(v_xy)
        adam_yx = _init_adam(v_yx)

        # Gradient buffers
        d_v_xy = similar(v_xy)
        d_v_yx = similar(v_yx)

        # Optimization loop
        for iter in 1:iters
            fill!(d_v_xy, zero(T))
            fill!(d_v_yx, zero(T))

            # Apply flow smoothing
            v_xy_smooth = reg.sigma_flow > zero(T) ? gauss_smoothing(v_xy, reg.sigma_flow) : v_xy
            v_yx_smooth = reg.sigma_flow > zero(T) ? gauss_smoothing(v_yx, reg.sigma_flow) : v_yx

            # Forward pass
            images, flows = apply_flows(reg, x_small, y_small, v_xy_smooth, v_yx_smooth)

            # Compute losses
            dissim_1 = loss_fn(x_small, images[:yx_full])
            dissim_2 = loss_fn(y_small, images[:xy_full])
            dissim_3 = loss_fn(images[:yx_half], images[:xy_half])
            dissim_arr = dissim_1 .+ dissim_2 .+ dissim_3

            # Regularization
            reg_xy = linear_elasticity(flows[:xy_full])
            reg_yx = linear_elasticity(flows[:yx_full])
            reg_arr = reg_xy .+ reg_yx

            # Total loss
            loss_arr = dissim_arr .+ reg.lambda_ .* reg_arr
            loss_val = AK.reduce(+, loss_arr; init=zero(T))
            push!(reg.loss_history, loss_val)

            if reg.verbose && (iter % 10 == 0 || iter == 1 || iter == iters)
                dissim_val = AK.reduce(+, dissim_arr; init=zero(T))
                reg_val = AK.reduce(+, reg_arr; init=zero(T))
                println("  Iter $iter/$iters: loss=$(round(loss_val, digits=6)), dissim=$(round(dissim_val, digits=6)), reg=$(round(reg_val, digits=6))")
            end

            # Backward pass - compute gradients via finite differences for now
            # This is a simplified approach; full AD would require more complex rrule!! chains
            _compute_syn_gradients!(d_v_xy, d_v_yx, x_small, y_small, v_xy_smooth, v_yx_smooth,
                                     reg, loss_fn)

            # Adam update
            _adam_step!(v_xy, d_v_xy, adam_xy, lr)
            _adam_step!(v_yx, d_v_yx, adam_yx, lr)
        end

        # Apply final smoothing and store
        v_xy_smooth = reg.sigma_flow > zero(T) ? gauss_smoothing(v_xy, reg.sigma_flow) : v_xy
        v_yx_smooth = reg.sigma_flow > zero(T) ? gauss_smoothing(v_yx, reg.sigma_flow) : v_yx

        # Upsample velocity fields back to full resolution
        reg.v_xy = _resample_velocity(v_xy_smooth, static_shape, static)
        reg.v_yx = _resample_velocity(v_yx_smooth, static_shape, static)
    end

    return reg
end

"""
    _resample_velocity(v, target_shape, reference)

Resample velocity field to target spatial shape.
"""
function _resample_velocity(
    v::AbstractArray{T,5},
    target_shape::NTuple{3,Int},
    reference::AbstractArray{T}
) where T
    X_t, Y_t, Z_t = target_shape
    X_v, Y_v, Z_v, D, N = size(v)

    if (X_v, Y_v, Z_v) == target_shape
        return copy(v)
    end

    # Treat D as channels for resampling
    result = similar(reference, X_t, Y_t, Z_t, D, N)

    # Use grid_sample to resample
    theta_cpu = zeros(T, 3, 4, N)
    for n in 1:N
        theta_cpu[1, 1, n] = one(T)
        theta_cpu[2, 2, n] = one(T)
        theta_cpu[3, 3, n] = one(T)
    end
    theta = similar(reference, 3, 4, N)
    copyto!(theta, theta_cpu)

    grid = affine_grid(theta, target_shape; align_corners=true)
    result = grid_sample(v, grid; padding_mode=:border, align_corners=true)

    return result
end

"""
    _compute_syn_gradients!(d_v_xy, d_v_yx, x, y, v_xy, v_yx, reg, loss_fn)

Compute gradients for SyN velocity fields using numerical approximation.
This is a simplified version - full AD through apply_flows is complex.
"""
function _compute_syn_gradients!(
    d_v_xy::AbstractArray{T,5},
    d_v_yx::AbstractArray{T,5},
    x::AbstractArray{T,5},
    y::AbstractArray{T,5},
    v_xy::AbstractArray{T,5},
    v_yx::AbstractArray{T,5},
    reg::SyNRegistration{T},
    loss_fn::Function
) where T
    # Compute current loss
    images, flows = apply_flows(reg, x, y, v_xy, v_yx)

    # Extract arrays from Dict BEFORE closures (GPU kernels require bits types)
    yx_full_img = images[:yx_full]
    xy_full_img = images[:xy_full]
    yx_full_flow = flows[:yx_full]
    xy_full_flow = flows[:xy_full]

    # Get config values as locals (bits types)
    align_corners = reg.align_corners
    padding_mode = reg.padding_mode
    lambda_ = reg.lambda_

    # Gradient w.r.t. moved images -> velocity fields
    # Simplified: gradient of mse_loss w.r.t. first argument
    # Then propagate through spatial_transform

    X, Y, Z, D, N = size(v_xy)
    n_elements = T(length(x))
    scale = T(2) / n_elements

    # d_loss/d_images[:yx_full] = 2*(x - yx_full)/n  (for mse_loss(x, yx_full))
    d_yx_full = similar(yx_full_img)
    AK.foreachindex(d_yx_full) do idx
        @inbounds d_yx_full[idx] = -scale * (x[idx] - yx_full_img[idx])
    end

    # d_loss/d_images[:xy_full] = 2*(y - xy_full)/n
    d_xy_full = similar(xy_full_img)
    AK.foreachindex(d_xy_full) do idx
        @inbounds d_xy_full[idx] = -scale * (y[idx] - xy_full_img[idx])
    end

    # Propagate through spatial_transform to get d_v
    # This is an approximation - full gradient is more complex
    _∇spatial_transform_v!(d_v_yx, d_yx_full, y, yx_full_flow;
                           align_corners=align_corners, padding_mode=padding_mode)
    _∇spatial_transform_v!(d_v_xy, d_xy_full, x, xy_full_flow;
                           align_corners=align_corners, padding_mode=padding_mode)

    # Add regularization gradient
    d_reg_xy = similar(v_xy)
    d_reg_yx = similar(v_yx)
    fill!(d_reg_xy, zero(T))
    fill!(d_reg_yx, zero(T))

    d_reg_out = similar(xy_full_flow, 1)
    fill!(d_reg_out, one(T))

    _∇linear_elasticity!(d_reg_xy, d_reg_out, xy_full_flow, T(2), T(1))
    _∇linear_elasticity!(d_reg_yx, d_reg_out, yx_full_flow, T(2), T(1))

    d_v_xy .+= lambda_ .* d_reg_xy
    d_v_yx .+= lambda_ .* d_reg_yx

    return nothing
end

# ============================================================================
# User-Facing API
# ============================================================================

"""
    register(reg::SyNRegistration, moving, static; kwargs...)

Register moving image to static image using SyN diffeomorphic registration.

# Keyword Arguments
- `loss_fn`: Loss function (default: mse_loss)
- `reset_params`: Reset parameters before fitting (default: true)
- `final_interpolation`: Interpolation mode for final output (:bilinear default, :nearest for HU preservation).
  During optimization, trilinear is always used for smooth gradients.

# Returns
- (moved_xy, moved_yx, flow_xy, flow_yx): Forward and inverse moved images and flow fields
"""
function register(
    reg::SyNRegistration{T},
    moving::AbstractArray{T,5},
    static::AbstractArray{T,5};
    loss_fn::Function=mse_loss,
    reset_params::Bool=true,
    final_interpolation::Symbol=:bilinear
) where T
    if reset_params
        reset!(reg)
    end

    # Fit parameters (always uses trilinear for smooth gradients)
    fit!(reg, moving, static; loss_fn=loss_fn)

    # Apply final transformation with specified interpolation mode
    images, flows = apply_flows(reg, moving, static, reg.v_xy, reg.v_yx; interpolation=final_interpolation)

    return images[:xy_full], images[:yx_full], flows[:xy_full], flows[:yx_full]
end

"""
    transform(reg::SyNRegistration, image; direction=:forward, interpolation=:bilinear)

Transform an image using the current registration parameters.

# Arguments
- `image`: Image to transform (X, Y, Z, C, N)
- `direction`: :forward (moving→static) or :inverse (static→moving)
- `interpolation`: Interpolation mode (:bilinear/:trilinear default, :nearest for HU preservation)
"""
function transform(
    reg::SyNRegistration{T},
    image::AbstractArray{T,5};
    direction::Symbol=:forward,
    interpolation::Symbol=:bilinear
) where T
    @assert reg.v_xy !== nothing "Must fit registration first"

    v = direction === :forward ? reg.v_xy : reg.v_yx

    # Compute diffeomorphic flow
    flow = diffeomorphic_transform(v; time_steps=reg.time_steps,
                                    align_corners=reg.align_corners,
                                    padding_mode=reg.padding_mode)

    return spatial_transform(image, flow; align_corners=reg.align_corners, padding_mode=reg.padding_mode, interpolation=interpolation)
end
