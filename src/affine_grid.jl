# GPU-accelerated affine_grid with Mooncake AD support
# Uses AcceleratedKernels.jl for cross-platform GPU execution
# Dependencies imported by parent module: AK, Mooncake, CoDual, NoFData, NoRData, @is_primitive, MinimalCtx

# ============================================================================
# Julia Array Convention: (dims, X, Y, [Z], N) - column-major
# PyTorch Convention: (N, H, W, dims) / (N, D, H, W, dims) - row-major
#
# For 2D:
#   theta:  (2, 3, N) - 2x3 affine matrix per batch (rotation+scale+shear, translation)
#   output: (2, X_out, Y_out, N) - grid of (x, y) coordinates in [-1, 1]
#
# For 3D:
#   theta:  (3, 4, N) - 3x4 affine matrix per batch
#   output: (3, X_out, Y_out, Z_out, N) - grid of (x, y, z) coordinates in [-1, 1]
#
# The affine transformation is:
#   [x_out]   [a b tx] [x_in]
#   [y_out] = [c d ty] [y_in]
#             where (x_in, y_in, 1) is the homogeneous normalized coordinate
# ============================================================================

"""
    affine_grid(theta, size; align_corners=true)

Generate a sampling grid from an affine transformation matrix.

# Arguments
- `theta`: Affine transformation matrix
  - 2D: shape (2, 3, N) - each batch has a 2x3 matrix [A | t] where A is 2x2, t is 2x1
  - 3D: shape (3, 4, N) - each batch has a 3x4 matrix [A | t] where A is 3x3, t is 3x1
- `size`: Output size tuple
  - 2D: (X_out, Y_out) or (X_out, Y_out, C, N)
  - 3D: (X_out, Y_out, Z_out) or (X_out, Y_out, Z_out, C, N)

# Keyword Arguments
- `align_corners`: If true (default), corner pixels are at -1 and 1.

# Returns
- Grid of normalized coordinates
  - 2D: shape (2, X_out, Y_out, N)
  - 3D: shape (3, X_out, Y_out, Z_out, N)

# Example
```julia
# Create identity transformation
theta = zeros(Float32, 2, 3, 1)
theta[1, 1, 1] = 1.0f0  # scale x
theta[2, 2, 1] = 1.0f0  # scale y

# Generate grid for 64x64 output
grid = affine_grid(theta, (64, 64))
```
"""
function affine_grid(
    theta::AbstractArray{T,3},
    size::NTuple{2,Int};
    align_corners::Bool=true
) where T
    return _affine_grid_2d(theta, size, Val(align_corners))
end

function affine_grid(
    theta::AbstractArray{T,3},
    size::NTuple{4,Int};
    align_corners::Bool=true
) where T
    return _affine_grid_2d(theta, (size[1], size[2]), Val(align_corners))
end

function affine_grid(
    theta::AbstractArray{T,3},
    size::NTuple{3,Int};
    align_corners::Bool=true
) where T
    return _affine_grid_3d(theta, size, Val(align_corners))
end

function affine_grid(
    theta::AbstractArray{T,3},
    size::NTuple{5,Int};
    align_corners::Bool=true
) where T
    return _affine_grid_3d(theta, (size[1], size[2], size[3]), Val(align_corners))
end

# ============================================================================
# 2D Affine Grid
# ============================================================================

function _affine_grid_2d(
    theta::AbstractArray{T,3},
    size::NTuple{2,Int},
    ::Val{AC}
) where {T, AC}
    @assert Base.size(theta, 1) == 2 "theta must have 2 rows for 2D"
    @assert Base.size(theta, 2) == 3 "theta must have 3 columns for 2D (2x3 matrix)"

    X_out, Y_out = size
    N = Base.size(theta, 3)

    # Output grid: (2, X_out, Y_out, N)
    grid = similar(theta, 2, X_out, Y_out, N)

    AK.foreachindex(grid) do idx
        # Convert linear index to (coord, i, j, n)
        coord, i, j, n = _linear_to_cartesian_4d_affine(idx, X_out, Y_out)

        # Generate normalized base coordinates
        x_base, y_base = _generate_base_coord_2d(i, j, X_out, Y_out, T, AC)

        # Apply affine transformation: [a b tx; c d ty] * [x; y; 1]
        if coord == 1
            # x_out = a*x + b*y + tx
            @inbounds grid[idx] = theta[1, 1, n] * x_base + theta[1, 2, n] * y_base + theta[1, 3, n]
        else
            # y_out = c*x + d*y + ty
            @inbounds grid[idx] = theta[2, 1, n] * x_base + theta[2, 2, n] * y_base + theta[2, 3, n]
        end
    end

    return grid
end

# ============================================================================
# 3D Affine Grid
# ============================================================================

function _affine_grid_3d(
    theta::AbstractArray{T,3},
    size::NTuple{3,Int},
    ::Val{AC}
) where {T, AC}
    @assert Base.size(theta, 1) == 3 "theta must have 3 rows for 3D"
    @assert Base.size(theta, 2) == 4 "theta must have 4 columns for 3D (3x4 matrix)"

    X_out, Y_out, Z_out = size
    N = Base.size(theta, 3)

    # Output grid: (3, X_out, Y_out, Z_out, N)
    grid = similar(theta, 3, X_out, Y_out, Z_out, N)

    AK.foreachindex(grid) do idx
        # Convert linear index to (coord, i, j, k, n)
        coord, i, j, k, n = _linear_to_cartesian_5d_affine(idx, X_out, Y_out, Z_out)

        # Generate normalized base coordinates
        x_base, y_base, z_base = _generate_base_coord_3d(i, j, k, X_out, Y_out, Z_out, T, AC)

        # Apply affine transformation
        if coord == 1
            # x_out
            @inbounds grid[idx] = theta[1, 1, n] * x_base + theta[1, 2, n] * y_base + theta[1, 3, n] * z_base + theta[1, 4, n]
        elseif coord == 2
            # y_out
            @inbounds grid[idx] = theta[2, 1, n] * x_base + theta[2, 2, n] * y_base + theta[2, 3, n] * z_base + theta[2, 4, n]
        else
            # z_out
            @inbounds grid[idx] = theta[3, 1, n] * x_base + theta[3, 2, n] * y_base + theta[3, 3, n] * z_base + theta[3, 4, n]
        end
    end

    return grid
end

# ============================================================================
# Index Conversion Helpers
# ============================================================================

@inline function _linear_to_cartesian_4d_affine(idx::Int, X::Int, Y::Int)
    # Grid shape: (2, X, Y, N)
    idx_0 = idx - 1
    coord = idx_0 % 2 + 1
    idx_0 = idx_0 ÷ 2
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    n = idx_0 ÷ Y + 1
    return coord, i, j, n
end

@inline function _linear_to_cartesian_5d_affine(idx::Int, X::Int, Y::Int, Z::Int)
    # Grid shape: (3, X, Y, Z, N)
    idx_0 = idx - 1
    coord = idx_0 % 3 + 1
    idx_0 = idx_0 ÷ 3
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    n = idx_0 ÷ Z + 1
    return coord, i, j, k, n
end

# ============================================================================
# Base Coordinate Generation
# ============================================================================

@inline function _generate_base_coord_2d(i::Int, j::Int, X::Int, Y::Int, ::Type{T}, align_corners::Bool) where T
    # Generate normalized coordinates in [-1, 1]
    if align_corners
        # -1 at index 1, +1 at index X (or Y)
        x = X > 1 ? T(2) * T(i - 1) / T(X - 1) - T(1) : T(0)
        y = Y > 1 ? T(2) * T(j - 1) / T(Y - 1) - T(1) : T(0)
    else
        # Pixel centers, -1 and +1 at edges (half pixel outside)
        x = T(2) * (T(i) - T(0.5)) / T(X) - T(1)
        y = T(2) * (T(j) - T(0.5)) / T(Y) - T(1)
    end
    return x, y
end

@inline function _generate_base_coord_3d(i::Int, j::Int, k::Int, X::Int, Y::Int, Z::Int, ::Type{T}, align_corners::Bool) where T
    if align_corners
        x = X > 1 ? T(2) * T(i - 1) / T(X - 1) - T(1) : T(0)
        y = Y > 1 ? T(2) * T(j - 1) / T(Y - 1) - T(1) : T(0)
        z = Z > 1 ? T(2) * T(k - 1) / T(Z - 1) - T(1) : T(0)
    else
        x = T(2) * (T(i) - T(0.5)) / T(X) - T(1)
        y = T(2) * (T(j) - T(0.5)) / T(Y) - T(1)
        z = T(2) * (T(k) - T(0.5)) / T(Z) - T(1)
    end
    return x, y, z
end

# ============================================================================
# Backward Pass (Gradients)
# ============================================================================

function _∇affine_grid_theta_2d!(
    d_theta::AbstractArray{T,3},
    d_grid::AbstractArray{T,4},
    ::Val{AC}
) where {T, AC}
    _, X_out, Y_out, N = Base.size(d_grid)

    # For each element in d_grid, accumulate gradient to corresponding theta elements
    # d_grid[1, i, j, n] contributes to theta[1, :, n]
    # d_grid[2, i, j, n] contributes to theta[2, :, n]

    AK.foreachindex(d_grid) do idx
        coord, i, j, n = _linear_to_cartesian_4d_affine(idx, X_out, Y_out)

        d_val = @inbounds d_grid[idx]
        if d_val == zero(T)
            return nothing
        end

        x_base, y_base = _generate_base_coord_2d(i, j, X_out, Y_out, T, AC)

        # Gradient w.r.t theta:
        # grid[coord, i, j, n] = theta[coord, 1, n] * x_base + theta[coord, 2, n] * y_base + theta[coord, 3, n]
        # d_theta[coord, 1, n] += d_grid[coord, i, j, n] * x_base
        # d_theta[coord, 2, n] += d_grid[coord, i, j, n] * y_base
        # d_theta[coord, 3, n] += d_grid[coord, i, j, n] * 1

        Atomix.@atomic d_theta[coord, 1, n] += d_val * x_base
        Atomix.@atomic d_theta[coord, 2, n] += d_val * y_base
        Atomix.@atomic d_theta[coord, 3, n] += d_val

        return nothing
    end

    return nothing
end

function _∇affine_grid_theta_3d!(
    d_theta::AbstractArray{T,3},
    d_grid::AbstractArray{T,5},
    ::Val{AC}
) where {T, AC}
    _, X_out, Y_out, Z_out, N = Base.size(d_grid)

    AK.foreachindex(d_grid) do idx
        coord, i, j, k, n = _linear_to_cartesian_5d_affine(idx, X_out, Y_out, Z_out)

        d_val = @inbounds d_grid[idx]
        if d_val == zero(T)
            return nothing
        end

        x_base, y_base, z_base = _generate_base_coord_3d(i, j, k, X_out, Y_out, Z_out, T, AC)

        Atomix.@atomic d_theta[coord, 1, n] += d_val * x_base
        Atomix.@atomic d_theta[coord, 2, n] += d_val * y_base
        Atomix.@atomic d_theta[coord, 3, n] += d_val * z_base
        Atomix.@atomic d_theta[coord, 4, n] += d_val

        return nothing
    end

    return nothing
end

# ============================================================================
# Mooncake rrule!! Definitions
# ============================================================================

# Mark as primitives - need to handle all size tuple variants
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{2,Int}}
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{3,Int}}
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{4,Int}}
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{5,Int}}

# 2D rrule!! (for NTuple{2} and NTuple{4} sizes)
function Mooncake.rrule!!(
    ::CoDual{typeof(affine_grid)},
    theta::CoDual{A, F},
    size::CoDual{S, NoFData};
    align_corners::Bool=true
) where {A<:AbstractArray{<:Any,3}, F, S<:Union{NTuple{2,Int}, NTuple{4,Int}}}
    theta_primal = theta.x
    theta_fdata = theta.dx
    size_primal = size.x

    ac = Val(align_corners)

    # Determine actual output size
    out_size = S <: NTuple{2,Int} ? size_primal : (size_primal[1], size_primal[2])

    output = affine_grid(theta_primal, size_primal; align_corners)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function affine_grid_2d_pullback(_rdata)
        _∇affine_grid_theta_2d!(theta_fdata, output_fdata, ac)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), affine_grid_2d_pullback
end

# 3D rrule!! (for NTuple{3} and NTuple{5} sizes)
function Mooncake.rrule!!(
    ::CoDual{typeof(affine_grid)},
    theta::CoDual{A, F},
    size::CoDual{S, NoFData};
    align_corners::Bool=true
) where {A<:AbstractArray{<:Any,3}, F, S<:Union{NTuple{3,Int}, NTuple{5,Int}}}
    theta_primal = theta.x
    theta_fdata = theta.dx
    size_primal = size.x

    ac = Val(align_corners)

    output = affine_grid(theta_primal, size_primal; align_corners)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function affine_grid_3d_pullback(_rdata)
        _∇affine_grid_theta_3d!(theta_fdata, output_fdata, ac)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), affine_grid_3d_pullback
end
