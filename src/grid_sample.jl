# Pure Julia implementation of grid_sample with GPU acceleration
#
# This implementation is designed to:
# 1. Match PyTorch F.grid_sample output exactly (within rtol=1e-5)
# 2. Be fully differentiable with Mooncake AD (pure Julia, sequential loops on CPU)
# 3. Support bilinear (2D) and trilinear (3D) interpolation
# 4. Support padding_mode :zeros and :border
# 5. Use AcceleratedKernels.jl for GPU acceleration (CUDA, Metal, ROCm)
#
# Parallelization strategy:
# - CPU: Uses sequential loops for Mooncake AD compatibility
#   (AK.foreachindex on CPU uses Task spawning which breaks Mooncake)
# - GPU: Uses AK.foreachindex which dispatches to efficient GPU kernels
#   (no Task spawning on GPU backends)

import AcceleratedKernels as AK
import GPUArraysCore: AbstractGPUArray

# Dispatch helper: returns true for GPU arrays, false for CPU arrays
_is_gpu_array(::AbstractArray) = false
_is_gpu_array(::AbstractGPUArray) = true

"""
    grid_sample(input::AbstractArray{T, 4}, grid::AbstractArray{T, 4};
                padding_mode::Symbol=:zeros) where T

Sample from a 2D input using a sampling grid (bilinear interpolation).

# Arguments
- `input`: Input array of shape `(X, Y, C, N)` where X=width, Y=height, C=channels, N=batch
- `grid`: Sampling grid of shape `(2, X_out, Y_out, N)` where dim 1 contains (x, y) coords
- `padding_mode`: Either `:zeros` (default) or `:border`

# Returns
- Output array of shape `(X_out, Y_out, C, N)`

# Notes
- Grid coordinates are in normalized [-1, 1] range (align_corners=true semantics)
- Uses bilinear interpolation
- CPU: Sequential loops compatible with Mooncake AD
- GPU: AcceleratedKernels.jl for parallel execution
"""
function grid_sample(input::AbstractArray{T, 4}, grid::AbstractArray{T, 4};
                     padding_mode::Symbol=:zeros) where T
    X_in, Y_in, C, N = size(input)
    coords_dim, X_out, Y_out, N_grid = size(grid)

    @assert coords_dim == 2 "Grid must have 2 coordinates in first dimension, got $coords_dim"
    @assert N_grid == N "Grid batch size ($N_grid) must match input batch size ($N)"
    @assert padding_mode in (:zeros, :border) "padding_mode must be :zeros or :border, got $padding_mode"

    output = similar(input, T, X_out, Y_out, C, N)
    fill!(output, zero(T))

    if _is_gpu_array(input)
        # GPU path: Use AK.foreachindex for parallel GPU execution
        _grid_sample_2d_gpu!(output, input, grid, X_in, Y_in, C, N, X_out, Y_out, padding_mode)
    else
        # CPU path: Sequential loops for Mooncake AD compatibility
        _grid_sample_2d_cpu!(output, input, grid, X_in, Y_in, C, N, X_out, Y_out, padding_mode)
    end

    return output
end

# CPU implementation: sequential loops for AD compatibility
function _grid_sample_2d_cpu!(output, input::AbstractArray{T}, grid, X_in, Y_in, C, N, X_out, Y_out, padding_mode) where T
    @inbounds for n in 1:N
        for j_out in 1:Y_out
            for i_out in 1:X_out
                # Get normalized coordinates from grid
                x_norm = grid[1, i_out, j_out, n]
                y_norm = grid[2, i_out, j_out, n]

                # Convert from normalized [-1, 1] to pixel coordinates [1, size]
                x = (x_norm + one(T)) * T(0.5) * T(X_in - 1) + one(T)
                y = (y_norm + one(T)) * T(0.5) * T(Y_in - 1) + one(T)

                # Bilinear interpolation indices
                x0 = floor(Int, x)
                y0 = floor(Int, y)
                x1 = x0 + 1
                y1 = y0 + 1

                # Interpolation weights
                wx1 = x - T(x0)
                wx0 = one(T) - wx1
                wy1 = y - T(y0)
                wy0 = one(T) - wy1

                for c in 1:C
                    v00 = _sample_2d(input, x0, y0, c, n, X_in, Y_in, padding_mode)
                    v10 = _sample_2d(input, x1, y0, c, n, X_in, Y_in, padding_mode)
                    v01 = _sample_2d(input, x0, y1, c, n, X_in, Y_in, padding_mode)
                    v11 = _sample_2d(input, x1, y1, c, n, X_in, Y_in, padding_mode)

                    output[i_out, j_out, c, n] = wx0 * wy0 * v00 +
                                                  wx1 * wy0 * v10 +
                                                  wx0 * wy1 * v01 +
                                                  wx1 * wy1 * v11
                end
            end
        end
    end
end

# GPU implementation: AK.foreachindex for parallel execution
function _grid_sample_2d_gpu!(output, input::AbstractArray{T}, grid, X_in, Y_in, C, N, X_out, Y_out, padding_mode) where T
    AK.foreachindex(output) do idx
        # Convert linear index to (i_out, j_out, c, n) using column-major order
        i_out = mod1(idx, X_out)
        j_out = mod1(div(idx - 1, X_out) + 1, Y_out)
        c = mod1(div(idx - 1, X_out * Y_out) + 1, C)
        n = div(idx - 1, X_out * Y_out * C) + 1

        # Get normalized coordinates from grid
        x_norm = grid[1, i_out, j_out, n]
        y_norm = grid[2, i_out, j_out, n]

        # Convert from normalized [-1, 1] to pixel coordinates [1, size]
        x = (x_norm + one(T)) * T(0.5) * T(X_in - 1) + one(T)
        y = (y_norm + one(T)) * T(0.5) * T(Y_in - 1) + one(T)

        # Bilinear interpolation indices
        x0 = floor(Int, x)
        y0 = floor(Int, y)
        x1 = x0 + 1
        y1 = y0 + 1

        # Interpolation weights
        wx1 = x - T(x0)
        wx0 = one(T) - wx1
        wy1 = y - T(y0)
        wy0 = one(T) - wy1

        # Sample and interpolate
        v00 = _sample_2d(input, x0, y0, c, n, X_in, Y_in, padding_mode)
        v10 = _sample_2d(input, x1, y0, c, n, X_in, Y_in, padding_mode)
        v01 = _sample_2d(input, x0, y1, c, n, X_in, Y_in, padding_mode)
        v11 = _sample_2d(input, x1, y1, c, n, X_in, Y_in, padding_mode)

        @inbounds output[i_out, j_out, c, n] = wx0 * wy0 * v00 +
                                                wx1 * wy0 * v10 +
                                                wx0 * wy1 * v01 +
                                                wx1 * wy1 * v11
    end
end

"""
    grid_sample(input::AbstractArray{T, 5}, grid::AbstractArray{T, 5};
                padding_mode::Symbol=:zeros) where T

Sample from a 3D input using a sampling grid (trilinear interpolation).

# Arguments
- `input`: Input array of shape `(X, Y, Z, C, N)` where X=width, Y=height, Z=depth, C=channels, N=batch
- `grid`: Sampling grid of shape `(3, X_out, Y_out, Z_out, N)` where dim 1 contains (x, y, z) coords
- `padding_mode`: Either `:zeros` (default) or `:border`

# Returns
- Output array of shape `(X_out, Y_out, Z_out, C, N)`

# Notes
- Grid coordinates are in normalized [-1, 1] range (align_corners=true semantics)
- Uses trilinear interpolation
- CPU: Sequential loops compatible with Mooncake AD
- GPU: AcceleratedKernels.jl for parallel execution
"""
function grid_sample(input::AbstractArray{T, 5}, grid::AbstractArray{T, 5};
                     padding_mode::Symbol=:zeros) where T
    X_in, Y_in, Z_in, C, N = size(input)
    coords_dim, X_out, Y_out, Z_out, N_grid = size(grid)

    @assert coords_dim == 3 "Grid must have 3 coordinates in first dimension, got $coords_dim"
    @assert N_grid == N "Grid batch size ($N_grid) must match input batch size ($N)"
    @assert padding_mode in (:zeros, :border) "padding_mode must be :zeros or :border, got $padding_mode"

    output = similar(input, T, X_out, Y_out, Z_out, C, N)
    fill!(output, zero(T))

    if _is_gpu_array(input)
        # GPU path: Use AK.foreachindex for parallel GPU execution
        _grid_sample_3d_gpu!(output, input, grid, X_in, Y_in, Z_in, C, N, X_out, Y_out, Z_out, padding_mode)
    else
        # CPU path: Sequential loops for Mooncake AD compatibility
        _grid_sample_3d_cpu!(output, input, grid, X_in, Y_in, Z_in, C, N, X_out, Y_out, Z_out, padding_mode)
    end

    return output
end

# CPU implementation: sequential loops for AD compatibility
function _grid_sample_3d_cpu!(output, input::AbstractArray{T}, grid, X_in, Y_in, Z_in, C, N, X_out, Y_out, Z_out, padding_mode) where T
    @inbounds for n in 1:N
        for k_out in 1:Z_out
            for j_out in 1:Y_out
                for i_out in 1:X_out
                    # Get normalized coordinates from grid
                    x_norm = grid[1, i_out, j_out, k_out, n]
                    y_norm = grid[2, i_out, j_out, k_out, n]
                    z_norm = grid[3, i_out, j_out, k_out, n]

                    # Convert from normalized [-1, 1] to pixel coordinates [1, size]
                    x = (x_norm + one(T)) * T(0.5) * T(X_in - 1) + one(T)
                    y = (y_norm + one(T)) * T(0.5) * T(Y_in - 1) + one(T)
                    z = (z_norm + one(T)) * T(0.5) * T(Z_in - 1) + one(T)

                    # Trilinear interpolation indices
                    x0 = floor(Int, x)
                    y0 = floor(Int, y)
                    z0 = floor(Int, z)
                    x1 = x0 + 1
                    y1 = y0 + 1
                    z1 = z0 + 1

                    # Interpolation weights
                    wx1 = x - T(x0)
                    wx0 = one(T) - wx1
                    wy1 = y - T(y0)
                    wy0 = one(T) - wy1
                    wz1 = z - T(z0)
                    wz0 = one(T) - wz1

                    for c in 1:C
                        v000 = _sample_3d(input, x0, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v100 = _sample_3d(input, x1, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v010 = _sample_3d(input, x0, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v110 = _sample_3d(input, x1, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v001 = _sample_3d(input, x0, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        v101 = _sample_3d(input, x1, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        v011 = _sample_3d(input, x0, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        v111 = _sample_3d(input, x1, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)

                        output[i_out, j_out, k_out, c, n] =
                            wx0 * wy0 * wz0 * v000 +
                            wx1 * wy0 * wz0 * v100 +
                            wx0 * wy1 * wz0 * v010 +
                            wx1 * wy1 * wz0 * v110 +
                            wx0 * wy0 * wz1 * v001 +
                            wx1 * wy0 * wz1 * v101 +
                            wx0 * wy1 * wz1 * v011 +
                            wx1 * wy1 * wz1 * v111
                    end
                end
            end
        end
    end
end

# GPU implementation: AK.foreachindex for parallel execution
function _grid_sample_3d_gpu!(output, input::AbstractArray{T}, grid, X_in, Y_in, Z_in, C, N, X_out, Y_out, Z_out, padding_mode) where T
    AK.foreachindex(output) do idx
        # Convert linear index to (i_out, j_out, k_out, c, n) using column-major order
        i_out = mod1(idx, X_out)
        j_out = mod1(div(idx - 1, X_out) + 1, Y_out)
        k_out = mod1(div(idx - 1, X_out * Y_out) + 1, Z_out)
        c = mod1(div(idx - 1, X_out * Y_out * Z_out) + 1, C)
        n = div(idx - 1, X_out * Y_out * Z_out * C) + 1

        # Get normalized coordinates from grid
        x_norm = grid[1, i_out, j_out, k_out, n]
        y_norm = grid[2, i_out, j_out, k_out, n]
        z_norm = grid[3, i_out, j_out, k_out, n]

        # Convert from normalized [-1, 1] to pixel coordinates [1, size]
        x = (x_norm + one(T)) * T(0.5) * T(X_in - 1) + one(T)
        y = (y_norm + one(T)) * T(0.5) * T(Y_in - 1) + one(T)
        z = (z_norm + one(T)) * T(0.5) * T(Z_in - 1) + one(T)

        # Trilinear interpolation indices
        x0 = floor(Int, x)
        y0 = floor(Int, y)
        z0 = floor(Int, z)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Interpolation weights
        wx1 = x - T(x0)
        wx0 = one(T) - wx1
        wy1 = y - T(y0)
        wy0 = one(T) - wy1
        wz1 = z - T(z0)
        wz0 = one(T) - wz1

        # Sample and interpolate
        v000 = _sample_3d(input, x0, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
        v100 = _sample_3d(input, x1, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
        v010 = _sample_3d(input, x0, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
        v110 = _sample_3d(input, x1, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
        v001 = _sample_3d(input, x0, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
        v101 = _sample_3d(input, x1, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
        v011 = _sample_3d(input, x0, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)
        v111 = _sample_3d(input, x1, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)

        @inbounds output[i_out, j_out, k_out, c, n] =
            wx0 * wy0 * wz0 * v000 +
            wx1 * wy0 * wz0 * v100 +
            wx0 * wy1 * wz0 * v010 +
            wx1 * wy1 * wz0 * v110 +
            wx0 * wy0 * wz1 * v001 +
            wx1 * wy0 * wz1 * v101 +
            wx0 * wy1 * wz1 * v011 +
            wx1 * wy1 * wz1 * v111
    end
end

# ============================================================================
# Internal helper functions for sampling with boundary handling
# ============================================================================

"""
Sample from 2D input with boundary handling.
Returns 0 for :zeros padding or clamped value for :border padding.
"""
@inline function _sample_2d(input::AbstractArray{T}, x::Int, y::Int, c::Int, n::Int,
                            X::Int, Y::Int, padding_mode::Symbol) where T
    if padding_mode == :zeros
        if x < 1 || x > X || y < 1 || y > Y
            return zero(T)
        else
            return input[x, y, c, n]
        end
    else  # :border
        x_clamped = clamp(x, 1, X)
        y_clamped = clamp(y, 1, Y)
        return input[x_clamped, y_clamped, c, n]
    end
end

"""
Sample from 3D input with boundary handling.
Returns 0 for :zeros padding or clamped value for :border padding.
"""
@inline function _sample_3d(input::AbstractArray{T}, x::Int, y::Int, z::Int, c::Int, n::Int,
                            X::Int, Y::Int, Z::Int, padding_mode::Symbol) where T
    if padding_mode == :zeros
        if x < 1 || x > X || y < 1 || y > Y || z < 1 || z > Z
            return zero(T)
        else
            return input[x, y, z, c, n]
        end
    else  # :border
        x_clamped = clamp(x, 1, X)
        y_clamped = clamp(y, 1, Y)
        z_clamped = clamp(z, 1, Z)
        return input[x_clamped, y_clamped, z_clamped, c, n]
    end
end

# ============================================================================
# Gradient computation for grid_sample
# ============================================================================

"""
    ∇grid_sample(dy::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{T, 4};
                 padding_mode::Symbol=:zeros) where T

Compute gradients for 2D grid_sample.

# Arguments
- `dy`: Upstream gradient of shape `(X_out, Y_out, C, N)`
- `input`: Original input of shape `(X, Y, C, N)`
- `grid`: Original grid of shape `(2, X_out, Y_out, N)`
- `padding_mode`: Same as forward pass

# Returns
- `d_input`: Gradient w.r.t. input, shape `(X, Y, C, N)`
- `d_grid`: Gradient w.r.t. grid, shape `(2, X_out, Y_out, N)`
"""
function ∇grid_sample(dy::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{T, 4};
                      padding_mode::Symbol=:zeros) where T
    X_in, Y_in, C, N = size(input)
    _, X_out, Y_out, _ = size(grid)

    d_input = zeros(T, X_in, Y_in, C, N)
    d_grid = zeros(T, 2, X_out, Y_out, N)

    @inbounds for n in 1:N
        for j_out in 1:Y_out
            for i_out in 1:X_out
                # Get normalized coordinates
                x_norm = grid[1, i_out, j_out, n]
                y_norm = grid[2, i_out, j_out, n]

                # Convert to pixel coordinates
                x = (x_norm + one(T)) * T(0.5) * T(X_in - 1) + one(T)
                y = (y_norm + one(T)) * T(0.5) * T(Y_in - 1) + one(T)

                x0 = floor(Int, x)
                y0 = floor(Int, y)
                x1 = x0 + 1
                y1 = y0 + 1

                wx1 = x - T(x0)
                wx0 = one(T) - wx1
                wy1 = y - T(y0)
                wy0 = one(T) - wy1

                for c in 1:C
                    dout = dy[i_out, j_out, c, n]

                    # Gradient w.r.t. input (scatter)
                    _scatter_grad_2d!(d_input, dout * wx0 * wy0, x0, y0, c, n, X_in, Y_in, padding_mode)
                    _scatter_grad_2d!(d_input, dout * wx1 * wy0, x1, y0, c, n, X_in, Y_in, padding_mode)
                    _scatter_grad_2d!(d_input, dout * wx0 * wy1, x0, y1, c, n, X_in, Y_in, padding_mode)
                    _scatter_grad_2d!(d_input, dout * wx1 * wy1, x1, y1, c, n, X_in, Y_in, padding_mode)

                    # Gradient w.r.t. grid coordinates
                    v00 = _sample_2d(input, x0, y0, c, n, X_in, Y_in, padding_mode)
                    v10 = _sample_2d(input, x1, y0, c, n, X_in, Y_in, padding_mode)
                    v01 = _sample_2d(input, x0, y1, c, n, X_in, Y_in, padding_mode)
                    v11 = _sample_2d(input, x1, y1, c, n, X_in, Y_in, padding_mode)

                    # d/dx of bilinear: (-wy0*v00 + wy0*v10 - wy1*v01 + wy1*v11)
                    dx = dout * (-wy0 * v00 + wy0 * v10 - wy1 * v01 + wy1 * v11)
                    # d/dy of bilinear: (-wx0*v00 - wx1*v10 + wx0*v01 + wx1*v11)
                    d_y = dout * (-wx0 * v00 - wx1 * v10 + wx0 * v01 + wx1 * v11)

                    # Chain rule: d_x_norm = dx * d_x/d_x_norm = dx * 0.5 * (X_in - 1)
                    d_grid[1, i_out, j_out, n] += dx * T(0.5) * T(X_in - 1)
                    d_grid[2, i_out, j_out, n] += d_y * T(0.5) * T(Y_in - 1)
                end
            end
        end
    end

    return d_input, d_grid
end

"""
    ∇grid_sample(dy::AbstractArray{T, 5}, input::AbstractArray{T, 5}, grid::AbstractArray{T, 5};
                 padding_mode::Symbol=:zeros) where T

Compute gradients for 3D grid_sample.

# Arguments
- `dy`: Upstream gradient of shape `(X_out, Y_out, Z_out, C, N)`
- `input`: Original input of shape `(X, Y, Z, C, N)`
- `grid`: Original grid of shape `(3, X_out, Y_out, Z_out, N)`
- `padding_mode`: Same as forward pass

# Returns
- `d_input`: Gradient w.r.t. input, shape `(X, Y, Z, C, N)`
- `d_grid`: Gradient w.r.t. grid, shape `(3, X_out, Y_out, Z_out, N)`
"""
function ∇grid_sample(dy::AbstractArray{T, 5}, input::AbstractArray{T, 5}, grid::AbstractArray{T, 5};
                      padding_mode::Symbol=:zeros) where T
    X_in, Y_in, Z_in, C, N = size(input)
    _, X_out, Y_out, Z_out, _ = size(grid)

    d_input = zeros(T, X_in, Y_in, Z_in, C, N)
    d_grid = zeros(T, 3, X_out, Y_out, Z_out, N)

    @inbounds for n in 1:N
        for k_out in 1:Z_out
            for j_out in 1:Y_out
                for i_out in 1:X_out
                    # Get normalized coordinates
                    x_norm = grid[1, i_out, j_out, k_out, n]
                    y_norm = grid[2, i_out, j_out, k_out, n]
                    z_norm = grid[3, i_out, j_out, k_out, n]

                    # Convert to pixel coordinates
                    x = (x_norm + one(T)) * T(0.5) * T(X_in - 1) + one(T)
                    y = (y_norm + one(T)) * T(0.5) * T(Y_in - 1) + one(T)
                    z = (z_norm + one(T)) * T(0.5) * T(Z_in - 1) + one(T)

                    x0 = floor(Int, x)
                    y0 = floor(Int, y)
                    z0 = floor(Int, z)
                    x1 = x0 + 1
                    y1 = y0 + 1
                    z1 = z0 + 1

                    wx1 = x - T(x0)
                    wx0 = one(T) - wx1
                    wy1 = y - T(y0)
                    wy0 = one(T) - wy1
                    wz1 = z - T(z0)
                    wz0 = one(T) - wz1

                    for c in 1:C
                        dout = dy[i_out, j_out, k_out, c, n]

                        # Gradient w.r.t. input (scatter)
                        _scatter_grad_3d!(d_input, dout * wx0 * wy0 * wz0, x0, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        _scatter_grad_3d!(d_input, dout * wx1 * wy0 * wz0, x1, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        _scatter_grad_3d!(d_input, dout * wx0 * wy1 * wz0, x0, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        _scatter_grad_3d!(d_input, dout * wx1 * wy1 * wz0, x1, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        _scatter_grad_3d!(d_input, dout * wx0 * wy0 * wz1, x0, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        _scatter_grad_3d!(d_input, dout * wx1 * wy0 * wz1, x1, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        _scatter_grad_3d!(d_input, dout * wx0 * wy1 * wz1, x0, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        _scatter_grad_3d!(d_input, dout * wx1 * wy1 * wz1, x1, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)

                        # Gradient w.r.t. grid coordinates
                        v000 = _sample_3d(input, x0, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v100 = _sample_3d(input, x1, y0, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v010 = _sample_3d(input, x0, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v110 = _sample_3d(input, x1, y1, z0, c, n, X_in, Y_in, Z_in, padding_mode)
                        v001 = _sample_3d(input, x0, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        v101 = _sample_3d(input, x1, y0, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        v011 = _sample_3d(input, x0, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)
                        v111 = _sample_3d(input, x1, y1, z1, c, n, X_in, Y_in, Z_in, padding_mode)

                        # d/dx of trilinear
                        dx = dout * (
                            -wy0 * wz0 * v000 + wy0 * wz0 * v100 +
                            -wy1 * wz0 * v010 + wy1 * wz0 * v110 +
                            -wy0 * wz1 * v001 + wy0 * wz1 * v101 +
                            -wy1 * wz1 * v011 + wy1 * wz1 * v111
                        )

                        # d/dy of trilinear
                        d_y = dout * (
                            -wx0 * wz0 * v000 - wx1 * wz0 * v100 +
                            wx0 * wz0 * v010 + wx1 * wz0 * v110 +
                            -wx0 * wz1 * v001 - wx1 * wz1 * v101 +
                            wx0 * wz1 * v011 + wx1 * wz1 * v111
                        )

                        # d/dz of trilinear
                        dz = dout * (
                            -wx0 * wy0 * v000 - wx1 * wy0 * v100 +
                            -wx0 * wy1 * v010 - wx1 * wy1 * v110 +
                            wx0 * wy0 * v001 + wx1 * wy0 * v101 +
                            wx0 * wy1 * v011 + wx1 * wy1 * v111
                        )

                        # Chain rule
                        d_grid[1, i_out, j_out, k_out, n] += dx * T(0.5) * T(X_in - 1)
                        d_grid[2, i_out, j_out, k_out, n] += d_y * T(0.5) * T(Y_in - 1)
                        d_grid[3, i_out, j_out, k_out, n] += dz * T(0.5) * T(Z_in - 1)
                    end
                end
            end
        end
    end

    return d_input, d_grid
end

# Helper functions for gradient scatter

@inline function _scatter_grad_2d!(d_input::AbstractArray{T}, grad::T, x::Int, y::Int,
                                    c::Int, n::Int, X::Int, Y::Int, padding_mode::Symbol) where T
    if padding_mode == :zeros
        if 1 <= x <= X && 1 <= y <= Y
            d_input[x, y, c, n] += grad
        end
    else  # :border - gradient flows to clamped position
        x_clamped = clamp(x, 1, X)
        y_clamped = clamp(y, 1, Y)
        d_input[x_clamped, y_clamped, c, n] += grad
    end
end

@inline function _scatter_grad_3d!(d_input::AbstractArray{T}, grad::T, x::Int, y::Int, z::Int,
                                    c::Int, n::Int, X::Int, Y::Int, Z::Int, padding_mode::Symbol) where T
    if padding_mode == :zeros
        if 1 <= x <= X && 1 <= y <= Y && 1 <= z <= Z
            d_input[x, y, z, c, n] += grad
        end
    else  # :border
        x_clamped = clamp(x, 1, X)
        y_clamped = clamp(y, 1, Y)
        z_clamped = clamp(z, 1, Z)
        d_input[x_clamped, y_clamped, z_clamped, c, n] += grad
    end
end
