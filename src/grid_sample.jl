# GPU-accelerated grid_sample with Mooncake AD support
# Uses AcceleratedKernels.jl for cross-platform GPU execution
# Dependencies imported by parent module: AK, Mooncake, CoDual, NoFData, NoRData, @is_primitive, MinimalCtx, Atomix

# ============================================================================
# Julia Array Convention: (X, Y, [Z], C, N) - column-major
# PyTorch Convention: (N, C, [Z], Y, X) - row-major
#
# For 2D:
#   input:  (X_in, Y_in, C, N) - input image
#   grid:   (2, X_out, Y_out, N) - sampling locations (x, y coords)
#   output: (X_out, Y_out, C, N) - sampled output
#
# For 3D:
#   input:  (X_in, Y_in, Z_in, C, N) - input volume
#   grid:   (3, X_out, Y_out, Z_out, N) - sampling locations (x, y, z coords)
#   output: (X_out, Y_out, Z_out, C, N) - sampled output
#
# Grid coordinates are normalized to [-1, 1]:
#   - (-1, -1) = top-left corner (first pixel center when align_corners=true)
#   - (1, 1) = bottom-right corner (last pixel center when align_corners=true)
# ============================================================================

# Padding mode types for GPU-compatible dispatch
abstract type AbstractPaddingMode end
struct ZerosPadding <: AbstractPaddingMode end
struct BorderPadding <: AbstractPaddingMode end

"""
    grid_sample(input, grid; padding_mode=:zeros, align_corners=true)

Sample from `input` at locations specified by `grid` using bilinear (2D) or
trilinear (3D) interpolation.

# Arguments
- `input`: Input array of shape (X, Y, C, N) for 2D or (X, Y, Z, C, N) for 3D
- `grid`: Sampling grid of shape (2, X_out, Y_out, N) for 2D or (3, X_out, Y_out, Z_out, N) for 3D
  Grid values should be in [-1, 1] normalized coordinates.

# Keyword Arguments
- `padding_mode`: How to handle out-of-bounds samples
  - `:zeros` (default): Use zero for out-of-bounds
  - `:border`: Clamp to border values
- `align_corners`: If true (default), corner pixels are at -1 and 1.
  If false, the corners of the image are at -1 and 1 (not pixel centers).

# Returns
- Sampled output of shape (X_out, Y_out, C, N) for 2D or (X_out, Y_out, Z_out, C, N) for 3D
"""
function grid_sample(
    input::AbstractArray{T,4},
    grid::AbstractArray{T,4};
    padding_mode::Symbol=:zeros,
    align_corners::Bool=true
) where T
    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    return _grid_sample_2d(input, grid, pm, Val(align_corners))
end

function grid_sample(
    input::AbstractArray{T,5},
    grid::AbstractArray{T,5};
    padding_mode::Symbol=:zeros,
    align_corners::Bool=true
) where T
    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    return _grid_sample_3d(input, grid, pm, Val(align_corners))
end

# ============================================================================
# 2D Bilinear Grid Sample
# ============================================================================

function _grid_sample_2d(
    input::AbstractArray{T,4},
    grid::AbstractArray{T,4},
    ::PM,
    ::Val{AC}
) where {T, PM<:AbstractPaddingMode, AC}
    X_in, Y_in, C, N = size(input)
    coord_dim, X_out, Y_out, N_grid = size(grid)

    @assert coord_dim == 2 "Grid must have 2 coordinates for 2D input"
    @assert N == N_grid "Batch size mismatch between input ($N) and grid ($N_grid)"

    output = similar(input, X_out, Y_out, C, N)

    AK.foreachindex(output) do idx
        # Convert linear index to (i_out, j_out, c, n)
        i_out, j_out, c, n = _linear_to_cartesian_4d(idx, X_out, Y_out, C)

        # Get normalized coordinates from grid
        x_norm = @inbounds grid[1, i_out, j_out, n]
        y_norm = @inbounds grid[2, i_out, j_out, n]

        # Unnormalize to pixel coordinates
        x_pix, y_pix = _unnormalize_2d(x_norm, y_norm, X_in, Y_in, AC)

        # Bilinear interpolation
        @inbounds output[idx] = _bilinear_sample(input, x_pix, y_pix, c, n, X_in, Y_in, PM())
    end

    return output
end

# ============================================================================
# 3D Trilinear Grid Sample
# ============================================================================

function _grid_sample_3d(
    input::AbstractArray{T,5},
    grid::AbstractArray{T,5},
    ::PM,
    ::Val{AC}
) where {T, PM<:AbstractPaddingMode, AC}
    X_in, Y_in, Z_in, C, N = size(input)
    coord_dim, X_out, Y_out, Z_out, N_grid = size(grid)

    @assert coord_dim == 3 "Grid must have 3 coordinates for 3D input"
    @assert N == N_grid "Batch size mismatch between input ($N) and grid ($N_grid)"

    output = similar(input, X_out, Y_out, Z_out, C, N)

    AK.foreachindex(output) do idx
        # Convert linear index to (i_out, j_out, k_out, c, n)
        i_out, j_out, k_out, c, n = _linear_to_cartesian_5d(idx, X_out, Y_out, Z_out, C)

        # Get normalized coordinates from grid
        x_norm = @inbounds grid[1, i_out, j_out, k_out, n]
        y_norm = @inbounds grid[2, i_out, j_out, k_out, n]
        z_norm = @inbounds grid[3, i_out, j_out, k_out, n]

        # Unnormalize to pixel coordinates
        x_pix, y_pix, z_pix = _unnormalize_3d(x_norm, y_norm, z_norm, X_in, Y_in, Z_in, AC)

        # Trilinear interpolation
        @inbounds output[idx] = _trilinear_sample(input, x_pix, y_pix, z_pix, c, n, X_in, Y_in, Z_in, PM())
    end

    return output
end

# ============================================================================
# Index Conversion Helpers
# ============================================================================

@inline function _linear_to_cartesian_4d(idx::Int, X::Int, Y::Int, C::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    c = idx_0 % C + 1
    n = idx_0 ÷ C + 1
    return i, j, c, n
end

@inline function _linear_to_cartesian_5d(idx::Int, X::Int, Y::Int, Z::Int, C::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    idx_0 = idx_0 ÷ Z
    c = idx_0 % C + 1
    n = idx_0 ÷ C + 1
    return i, j, k, c, n
end

# ============================================================================
# Coordinate Unnormalization
# ============================================================================

@inline function _unnormalize_2d(x_norm::T, y_norm::T, X::Int, Y::Int, align_corners::Bool) where T
    # Convert normalized [-1, 1] coordinates to 1-indexed pixel coordinates
    # align_corners=true: -1 maps to index 1, +1 maps to index X (pixel centers at corners)
    # align_corners=false: -1 maps to index 0.5, +1 maps to index X+0.5 (pixel edges at corners)
    if align_corners
        x_pix = (x_norm + one(T)) / 2 * T(X - 1) + one(T)
        y_pix = (y_norm + one(T)) / 2 * T(Y - 1) + one(T)
    else
        x_pix = (x_norm + one(T)) / 2 * T(X) + T(0.5)
        y_pix = (y_norm + one(T)) / 2 * T(Y) + T(0.5)
    end
    return x_pix, y_pix
end

@inline function _unnormalize_3d(x_norm::T, y_norm::T, z_norm::T, X::Int, Y::Int, Z::Int, align_corners::Bool) where T
    # Convert normalized [-1, 1] coordinates to 1-indexed pixel coordinates
    if align_corners
        x_pix = (x_norm + one(T)) / 2 * T(X - 1) + one(T)
        y_pix = (y_norm + one(T)) / 2 * T(Y - 1) + one(T)
        z_pix = (z_norm + one(T)) / 2 * T(Z - 1) + one(T)
    else
        x_pix = (x_norm + one(T)) / 2 * T(X) + T(0.5)
        y_pix = (y_norm + one(T)) / 2 * T(Y) + T(0.5)
        z_pix = (z_norm + one(T)) / 2 * T(Z) + T(0.5)
    end
    return x_pix, y_pix, z_pix
end

# ============================================================================
# Bilinear Interpolation (2D)
# ============================================================================

@inline function _bilinear_sample(
    input::AbstractArray{T,4},
    x::T, y::T,
    c::Int, n::Int,
    X::Int, Y::Int,
    pm::PM
) where {T, PM<:AbstractPaddingMode}
    # Floor to get integer indices (convert to Int carefully)
    x0 = unsafe_trunc(Int, floor(x))
    y0 = unsafe_trunc(Int, floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # Interpolation weights
    wx1 = x - T(x0)
    wx0 = one(T) - wx1
    wy1 = y - T(y0)
    wy0 = one(T) - wy1

    # Sample the four corners
    v00 = _sample_pixel_2d(input, x0, y0, c, n, X, Y, pm)
    v10 = _sample_pixel_2d(input, x1, y0, c, n, X, Y, pm)
    v01 = _sample_pixel_2d(input, x0, y1, c, n, X, Y, pm)
    v11 = _sample_pixel_2d(input, x1, y1, c, n, X, Y, pm)

    # Bilinear interpolation
    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11
end

@inline function _sample_pixel_2d(
    input::AbstractArray{T,4},
    i::Int, j::Int, c::Int, n::Int,
    X::Int, Y::Int,
    ::ZerosPadding
) where T
    if i < 1 || i > X || j < 1 || j > Y
        return zero(T)
    else
        return @inbounds input[i, j, c, n]
    end
end

@inline function _sample_pixel_2d(
    input::AbstractArray{T,4},
    i::Int, j::Int, c::Int, n::Int,
    X::Int, Y::Int,
    ::BorderPadding
) where T
    i_clamped = clamp(i, 1, X)
    j_clamped = clamp(j, 1, Y)
    return @inbounds input[i_clamped, j_clamped, c, n]
end

# ============================================================================
# Trilinear Interpolation (3D)
# ============================================================================

@inline function _trilinear_sample(
    input::AbstractArray{T,5},
    x::T, y::T, z::T,
    c::Int, n::Int,
    X::Int, Y::Int, Z::Int,
    pm::PM
) where {T, PM<:AbstractPaddingMode}
    x0 = unsafe_trunc(Int, floor(x))
    y0 = unsafe_trunc(Int, floor(y))
    z0 = unsafe_trunc(Int, floor(z))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    wx1 = x - T(x0)
    wx0 = one(T) - wx1
    wy1 = y - T(y0)
    wy0 = one(T) - wy1
    wz1 = z - T(z0)
    wz0 = one(T) - wz1

    v000 = _sample_pixel_3d(input, x0, y0, z0, c, n, X, Y, Z, pm)
    v100 = _sample_pixel_3d(input, x1, y0, z0, c, n, X, Y, Z, pm)
    v010 = _sample_pixel_3d(input, x0, y1, z0, c, n, X, Y, Z, pm)
    v110 = _sample_pixel_3d(input, x1, y1, z0, c, n, X, Y, Z, pm)
    v001 = _sample_pixel_3d(input, x0, y0, z1, c, n, X, Y, Z, pm)
    v101 = _sample_pixel_3d(input, x1, y0, z1, c, n, X, Y, Z, pm)
    v011 = _sample_pixel_3d(input, x0, y1, z1, c, n, X, Y, Z, pm)
    v111 = _sample_pixel_3d(input, x1, y1, z1, c, n, X, Y, Z, pm)

    return (wx0 * wy0 * wz0 * v000 + wx1 * wy0 * wz0 * v100 +
            wx0 * wy1 * wz0 * v010 + wx1 * wy1 * wz0 * v110 +
            wx0 * wy0 * wz1 * v001 + wx1 * wy0 * wz1 * v101 +
            wx0 * wy1 * wz1 * v011 + wx1 * wy1 * wz1 * v111)
end

@inline function _sample_pixel_3d(
    input::AbstractArray{T,5},
    i::Int, j::Int, k::Int, c::Int, n::Int,
    X::Int, Y::Int, Z::Int,
    ::ZerosPadding
) where T
    if i < 1 || i > X || j < 1 || j > Y || k < 1 || k > Z
        return zero(T)
    else
        return @inbounds input[i, j, k, c, n]
    end
end

@inline function _sample_pixel_3d(
    input::AbstractArray{T,5},
    i::Int, j::Int, k::Int, c::Int, n::Int,
    X::Int, Y::Int, Z::Int,
    ::BorderPadding
) where T
    i_clamped = clamp(i, 1, X)
    j_clamped = clamp(j, 1, Y)
    k_clamped = clamp(k, 1, Z)
    return @inbounds input[i_clamped, j_clamped, k_clamped, c, n]
end

# ============================================================================
# Backward Pass (Gradients) - 2D
# ============================================================================

function _∇grid_sample_input_2d!(
    d_input::AbstractArray{T,4},
    d_output::AbstractArray{T,4},
    grid::AbstractArray{T,4},
    ::PM,
    ::Val{AC}
) where {T, PM<:AbstractPaddingMode, AC}
    X_in, Y_in = size(d_input, 1), size(d_input, 2)
    _, X_out, Y_out, N = size(grid)
    C = size(d_output, 3)

    AK.foreachindex(d_output) do idx
        i_out, j_out, c, n = _linear_to_cartesian_4d(idx, X_out, Y_out, C)

        d_out_val = @inbounds d_output[idx]

        if d_out_val == zero(T)
            return nothing
        end

        x_norm = @inbounds grid[1, i_out, j_out, n]
        y_norm = @inbounds grid[2, i_out, j_out, n]

        x_pix, y_pix = _unnormalize_2d(x_norm, y_norm, X_in, Y_in, AC)

        x0 = unsafe_trunc(Int, floor(x_pix))
        y0 = unsafe_trunc(Int, floor(y_pix))
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = x_pix - T(x0)
        wx0 = one(T) - wx1
        wy1 = y_pix - T(y0)
        wy0 = one(T) - wy1

        _scatter_grad_2d!(d_input, d_out_val * wx0 * wy0, x0, y0, c, n, X_in, Y_in, PM())
        _scatter_grad_2d!(d_input, d_out_val * wx1 * wy0, x1, y0, c, n, X_in, Y_in, PM())
        _scatter_grad_2d!(d_input, d_out_val * wx0 * wy1, x0, y1, c, n, X_in, Y_in, PM())
        _scatter_grad_2d!(d_input, d_out_val * wx1 * wy1, x1, y1, c, n, X_in, Y_in, PM())

        return nothing
    end

    return nothing
end

@inline function _scatter_grad_2d!(
    d_input::AbstractArray{T,4},
    grad::T,
    i::Int, j::Int, c::Int, n::Int,
    X::Int, Y::Int,
    ::ZerosPadding
) where T
    if i >= 1 && i <= X && j >= 1 && j <= Y
        Atomix.@atomic d_input[i, j, c, n] += grad
    end
    return nothing
end

@inline function _scatter_grad_2d!(
    d_input::AbstractArray{T,4},
    grad::T,
    i::Int, j::Int, c::Int, n::Int,
    X::Int, Y::Int,
    ::BorderPadding
) where T
    i_clamped = clamp(i, 1, X)
    j_clamped = clamp(j, 1, Y)
    Atomix.@atomic d_input[i_clamped, j_clamped, c, n] += grad
    return nothing
end

function _∇grid_sample_grid_2d!(
    d_grid::AbstractArray{T,4},
    d_output::AbstractArray{T,4},
    input::AbstractArray{T,4},
    grid::AbstractArray{T,4},
    ::PM,
    ::Val{AC}
) where {T, PM<:AbstractPaddingMode, AC}
    X_in, Y_in, C, N = size(input)
    _, X_out, Y_out, _ = size(grid)

    AK.foreachindex(d_output) do idx
        i_out, j_out, c, n = _linear_to_cartesian_4d(idx, X_out, Y_out, C)

        d_out_val = @inbounds d_output[idx]

        if d_out_val == zero(T)
            return nothing
        end

        x_norm = @inbounds grid[1, i_out, j_out, n]
        y_norm = @inbounds grid[2, i_out, j_out, n]

        x_pix, y_pix = _unnormalize_2d(x_norm, y_norm, X_in, Y_in, AC)

        x0 = unsafe_trunc(Int, floor(x_pix))
        y0 = unsafe_trunc(Int, floor(y_pix))
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = x_pix - T(x0)
        wx0 = one(T) - wx1
        wy1 = y_pix - T(y0)
        wy0 = one(T) - wy1

        v00 = _sample_pixel_2d(input, x0, y0, c, n, X_in, Y_in, PM())
        v10 = _sample_pixel_2d(input, x1, y0, c, n, X_in, Y_in, PM())
        v01 = _sample_pixel_2d(input, x0, y1, c, n, X_in, Y_in, PM())
        v11 = _sample_pixel_2d(input, x1, y1, c, n, X_in, Y_in, PM())

        d_x_pix = wy0 * (v10 - v00) + wy1 * (v11 - v01)
        d_y_pix = wx0 * (v01 - v00) + wx1 * (v11 - v10)

        if AC
            d_x_norm = d_x_pix * T(X_in - 1) / 2
            d_y_norm = d_y_pix * T(Y_in - 1) / 2
        else
            d_x_norm = d_x_pix * T(X_in) / 2
            d_y_norm = d_y_pix * T(Y_in) / 2
        end

        Atomix.@atomic d_grid[1, i_out, j_out, n] += d_out_val * d_x_norm
        Atomix.@atomic d_grid[2, i_out, j_out, n] += d_out_val * d_y_norm

        return nothing
    end

    return nothing
end

# ============================================================================
# Backward Pass (Gradients) - 3D
# ============================================================================

function _∇grid_sample_input_3d!(
    d_input::AbstractArray{T,5},
    d_output::AbstractArray{T,5},
    grid::AbstractArray{T,5},
    ::PM,
    ::Val{AC}
) where {T, PM<:AbstractPaddingMode, AC}
    X_in, Y_in, Z_in = size(d_input, 1), size(d_input, 2), size(d_input, 3)
    _, X_out, Y_out, Z_out, N = size(grid)
    C = size(d_output, 4)

    AK.foreachindex(d_output) do idx
        i_out, j_out, k_out, c, n = _linear_to_cartesian_5d(idx, X_out, Y_out, Z_out, C)

        d_out_val = @inbounds d_output[idx]

        if d_out_val == zero(T)
            return nothing
        end

        x_norm = @inbounds grid[1, i_out, j_out, k_out, n]
        y_norm = @inbounds grid[2, i_out, j_out, k_out, n]
        z_norm = @inbounds grid[3, i_out, j_out, k_out, n]

        x_pix, y_pix, z_pix = _unnormalize_3d(x_norm, y_norm, z_norm, X_in, Y_in, Z_in, AC)

        x0 = unsafe_trunc(Int, floor(x_pix))
        y0 = unsafe_trunc(Int, floor(y_pix))
        z0 = unsafe_trunc(Int, floor(z_pix))
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        wx1 = x_pix - T(x0)
        wx0 = one(T) - wx1
        wy1 = y_pix - T(y0)
        wy0 = one(T) - wy1
        wz1 = z_pix - T(z0)
        wz0 = one(T) - wz1

        _scatter_grad_3d!(d_input, d_out_val * wx0 * wy0 * wz0, x0, y0, z0, c, n, X_in, Y_in, Z_in, PM())
        _scatter_grad_3d!(d_input, d_out_val * wx1 * wy0 * wz0, x1, y0, z0, c, n, X_in, Y_in, Z_in, PM())
        _scatter_grad_3d!(d_input, d_out_val * wx0 * wy1 * wz0, x0, y1, z0, c, n, X_in, Y_in, Z_in, PM())
        _scatter_grad_3d!(d_input, d_out_val * wx1 * wy1 * wz0, x1, y1, z0, c, n, X_in, Y_in, Z_in, PM())
        _scatter_grad_3d!(d_input, d_out_val * wx0 * wy0 * wz1, x0, y0, z1, c, n, X_in, Y_in, Z_in, PM())
        _scatter_grad_3d!(d_input, d_out_val * wx1 * wy0 * wz1, x1, y0, z1, c, n, X_in, Y_in, Z_in, PM())
        _scatter_grad_3d!(d_input, d_out_val * wx0 * wy1 * wz1, x0, y1, z1, c, n, X_in, Y_in, Z_in, PM())
        _scatter_grad_3d!(d_input, d_out_val * wx1 * wy1 * wz1, x1, y1, z1, c, n, X_in, Y_in, Z_in, PM())

        return nothing
    end

    return nothing
end

@inline function _scatter_grad_3d!(
    d_input::AbstractArray{T,5},
    grad::T,
    i::Int, j::Int, k::Int, c::Int, n::Int,
    X::Int, Y::Int, Z::Int,
    ::ZerosPadding
) where T
    if i >= 1 && i <= X && j >= 1 && j <= Y && k >= 1 && k <= Z
        Atomix.@atomic d_input[i, j, k, c, n] += grad
    end
    return nothing
end

@inline function _scatter_grad_3d!(
    d_input::AbstractArray{T,5},
    grad::T,
    i::Int, j::Int, k::Int, c::Int, n::Int,
    X::Int, Y::Int, Z::Int,
    ::BorderPadding
) where T
    i_clamped = clamp(i, 1, X)
    j_clamped = clamp(j, 1, Y)
    k_clamped = clamp(k, 1, Z)
    Atomix.@atomic d_input[i_clamped, j_clamped, k_clamped, c, n] += grad
    return nothing
end

function _∇grid_sample_grid_3d!(
    d_grid::AbstractArray{T,5},
    d_output::AbstractArray{T,5},
    input::AbstractArray{T,5},
    grid::AbstractArray{T,5},
    ::PM,
    ::Val{AC}
) where {T, PM<:AbstractPaddingMode, AC}
    X_in, Y_in, Z_in, C, N = size(input)
    _, X_out, Y_out, Z_out, _ = size(grid)

    AK.foreachindex(d_output) do idx
        i_out, j_out, k_out, c, n = _linear_to_cartesian_5d(idx, X_out, Y_out, Z_out, C)

        d_out_val = @inbounds d_output[idx]

        if d_out_val == zero(T)
            return nothing
        end

        x_norm = @inbounds grid[1, i_out, j_out, k_out, n]
        y_norm = @inbounds grid[2, i_out, j_out, k_out, n]
        z_norm = @inbounds grid[3, i_out, j_out, k_out, n]

        x_pix, y_pix, z_pix = _unnormalize_3d(x_norm, y_norm, z_norm, X_in, Y_in, Z_in, AC)

        x0 = unsafe_trunc(Int, floor(x_pix))
        y0 = unsafe_trunc(Int, floor(y_pix))
        z0 = unsafe_trunc(Int, floor(z_pix))
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        wx1 = x_pix - T(x0)
        wx0 = one(T) - wx1
        wy1 = y_pix - T(y0)
        wy0 = one(T) - wy1
        wz1 = z_pix - T(z0)
        wz0 = one(T) - wz1

        v000 = _sample_pixel_3d(input, x0, y0, z0, c, n, X_in, Y_in, Z_in, PM())
        v100 = _sample_pixel_3d(input, x1, y0, z0, c, n, X_in, Y_in, Z_in, PM())
        v010 = _sample_pixel_3d(input, x0, y1, z0, c, n, X_in, Y_in, Z_in, PM())
        v110 = _sample_pixel_3d(input, x1, y1, z0, c, n, X_in, Y_in, Z_in, PM())
        v001 = _sample_pixel_3d(input, x0, y0, z1, c, n, X_in, Y_in, Z_in, PM())
        v101 = _sample_pixel_3d(input, x1, y0, z1, c, n, X_in, Y_in, Z_in, PM())
        v011 = _sample_pixel_3d(input, x0, y1, z1, c, n, X_in, Y_in, Z_in, PM())
        v111 = _sample_pixel_3d(input, x1, y1, z1, c, n, X_in, Y_in, Z_in, PM())

        d_x_pix = (wz0 * (wy0 * (v100 - v000) + wy1 * (v110 - v010)) +
                   wz1 * (wy0 * (v101 - v001) + wy1 * (v111 - v011)))

        d_y_pix = (wz0 * (wx0 * (v010 - v000) + wx1 * (v110 - v100)) +
                   wz1 * (wx0 * (v011 - v001) + wx1 * (v111 - v101)))

        d_z_pix = (wy0 * (wx0 * (v001 - v000) + wx1 * (v101 - v100)) +
                   wy1 * (wx0 * (v011 - v010) + wx1 * (v111 - v110)))

        if AC
            d_x_norm = d_x_pix * T(X_in - 1) / 2
            d_y_norm = d_y_pix * T(Y_in - 1) / 2
            d_z_norm = d_z_pix * T(Z_in - 1) / 2
        else
            d_x_norm = d_x_pix * T(X_in) / 2
            d_y_norm = d_y_pix * T(Y_in) / 2
            d_z_norm = d_z_pix * T(Z_in) / 2
        end

        Atomix.@atomic d_grid[1, i_out, j_out, k_out, n] += d_out_val * d_x_norm
        Atomix.@atomic d_grid[2, i_out, j_out, k_out, n] += d_out_val * d_y_norm
        Atomix.@atomic d_grid[3, i_out, j_out, k_out, n] += d_out_val * d_z_norm

        return nothing
    end

    return nothing
end

# ============================================================================
# Mooncake rrule!! Definitions
# ============================================================================

# Mark as primitives
@is_primitive MinimalCtx Tuple{typeof(grid_sample), AbstractArray{<:Any,4}, AbstractArray{<:Any,4}}
@is_primitive MinimalCtx Tuple{typeof(grid_sample), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}

# 2D rrule!!
function Mooncake.rrule!!(
    ::CoDual{typeof(grid_sample)},
    input::CoDual{A1, F1},
    grid::CoDual{A2, F2};
    padding_mode::Symbol=:zeros,
    align_corners::Bool=true
) where {A1<:AbstractArray{<:Any,4}, F1, A2<:AbstractArray{<:Any,4}, F2}
    input_primal = input.x
    input_fdata = input.dx
    grid_primal = grid.x
    grid_fdata = grid.dx

    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    ac = Val(align_corners)

    output = grid_sample(input_primal, grid_primal; padding_mode, align_corners)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function grid_sample_2d_pullback(_rdata)
        _∇grid_sample_input_2d!(input_fdata, output_fdata, grid_primal, pm, ac)
        _∇grid_sample_grid_2d!(grid_fdata, output_fdata, input_primal, grid_primal, pm, ac)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), grid_sample_2d_pullback
end

# 3D rrule!!
function Mooncake.rrule!!(
    ::CoDual{typeof(grid_sample)},
    input::CoDual{A1, F1},
    grid::CoDual{A2, F2};
    padding_mode::Symbol=:zeros,
    align_corners::Bool=true
) where {A1<:AbstractArray{<:Any,5}, F1, A2<:AbstractArray{<:Any,5}, F2}
    input_primal = input.x
    input_fdata = input.dx
    grid_primal = grid.x
    grid_fdata = grid.dx

    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    ac = Val(align_corners)

    output = grid_sample(input_primal, grid_primal; padding_mode, align_corners)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function grid_sample_3d_pullback(_rdata)
        _∇grid_sample_input_3d!(input_fdata, output_fdata, grid_primal, pm, ac)
        _∇grid_sample_grid_3d!(grid_fdata, output_fdata, input_primal, grid_primal, pm, ac)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), grid_sample_3d_pullback
end
