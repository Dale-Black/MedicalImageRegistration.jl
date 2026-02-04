# Displacement field resampling for multi-resolution registration workflow
# GPU-first architecture with AK.foreachindex + Mooncake rrule!!
#
# Key functions:
# - resample_displacement: Upsample/downsample displacement field with value scaling
# - resample_velocity: Resample SyN velocity fields
# - upsample_affine_transform: Adjust affine for resolution change
# - invert_displacement: Compute inverse displacement field

# ============================================================================
# Resample Displacement Field
# ============================================================================

"""
    resample_displacement(disp::AbstractArray{T,5}, target_size::NTuple{3,Int}; interpolation=:bilinear) where T

Resample a displacement field to a new spatial size, scaling displacement values
by the resolution ratio.

# Arguments
- `disp`: Displacement field of shape (X, Y, Z, 3, N)
- `target_size`: Target spatial size (X_out, Y_out, Z_out)
- `interpolation`: Interpolation mode (:bilinear default, :nearest)

# Returns
- Resampled displacement field of shape (X_out, Y_out, Z_out, 3, N)

# Why scaling matters
When upsampling from 2mm to 0.5mm resolution (4x), a displacement of 1 voxel
at 2mm resolution corresponds to 4 voxels at 0.5mm resolution. The displacement
VALUES must be scaled by (target_size - 1) / (source_size - 1) for each axis.

# Example
```julia
# Register at 2mm isotropic (32³), apply at 0.5mm (128³)
disp_lowres = register_at_lowres(moving, static)  # (32, 32, 32, 3, 1)
disp_highres = resample_displacement(disp_lowres, (128, 128, 128))  # (128, 128, 128, 3, 1)
moved = spatial_transform(moving_highres, disp_highres; interpolation=:nearest)
```
"""
function resample_displacement(
    disp::AbstractArray{T,5},
    target_size::NTuple{3,Int};
    interpolation::Symbol=:bilinear
) where T
    X_in, Y_in, Z_in, D, N = size(disp)
    @assert D == 3 "Displacement field must have 3 channels"

    X_out, Y_out, Z_out = target_size

    # If already at target size, just copy
    if (X_in, Y_in, Z_in) == target_size
        return copy(disp)
    end

    # Compute scale factors for displacement value scaling
    # Scale factor = (target_size - 1) / (source_size - 1)
    # This accounts for the fact that displacement is in normalized [-1, 1] coordinates
    scale_x = T(X_out - 1) / T(max(X_in - 1, 1))
    scale_y = T(Y_out - 1) / T(max(Y_in - 1, 1))
    scale_z = T(Z_out - 1) / T(max(Z_in - 1, 1))

    # Create identity grid at target resolution to sample from source
    # Grid values in [-1, 1] normalized coordinates
    grid = _create_resample_grid_for_disp(disp, target_size)

    # Sample displacement field using grid_sample
    # Treat 3 displacement channels as channels for sampling
    disp_resampled = grid_sample(disp, grid; padding_mode=:border, align_corners=true, interpolation=interpolation)

    # Scale the displacement values by resolution ratio
    output = _scale_displacement_values(disp_resampled, scale_x, scale_y, scale_z)

    return output
end

# 2D version
function resample_displacement(
    disp::AbstractArray{T,4},
    target_size::NTuple{2,Int};
    interpolation::Symbol=:bilinear
) where T
    X_in, Y_in, D, N = size(disp)
    @assert D == 2 "2D displacement field must have 2 channels"

    X_out, Y_out = target_size

    if (X_in, Y_in) == target_size
        return copy(disp)
    end

    scale_x = T(X_out - 1) / T(max(X_in - 1, 1))
    scale_y = T(Y_out - 1) / T(max(Y_in - 1, 1))

    grid = _create_resample_grid_for_disp_2d(disp, target_size)
    disp_resampled = grid_sample(disp, grid; padding_mode=:border, align_corners=true, interpolation=interpolation)
    output = _scale_displacement_values_2d(disp_resampled, scale_x, scale_y)

    return output
end

"""
    _create_resample_grid_for_disp(disp, target_size)

Create a sampling grid for resampling a 3D displacement field.
"""
function _create_resample_grid_for_disp(
    disp::AbstractArray{T,5},
    target_size::NTuple{3,Int}
) where T
    X_out, Y_out, Z_out = target_size
    _, _, _, _, N = size(disp)

    grid = similar(disp, 3, X_out, Y_out, Z_out, N)

    AK.foreachindex(grid) do idx
        d, i, j, k, n = _linear_to_cartesian_5d_resample(idx, X_out, Y_out, Z_out)

        # Normalized coordinate in [-1, 1]
        x_norm = T(2) * (T(i) - one(T)) / T(max(X_out - 1, 1)) - one(T)
        y_norm = T(2) * (T(j) - one(T)) / T(max(Y_out - 1, 1)) - one(T)
        z_norm = T(2) * (T(k) - one(T)) / T(max(Z_out - 1, 1)) - one(T)

        if d == 1
            @inbounds grid[idx] = x_norm
        elseif d == 2
            @inbounds grid[idx] = y_norm
        else
            @inbounds grid[idx] = z_norm
        end
    end

    return grid
end

function _create_resample_grid_for_disp_2d(
    disp::AbstractArray{T,4},
    target_size::NTuple{2,Int}
) where T
    X_out, Y_out = target_size
    _, _, _, N = size(disp)

    grid = similar(disp, 2, X_out, Y_out, N)

    AK.foreachindex(grid) do idx
        d, i, j, n = _linear_to_cartesian_4d_resample(idx, X_out, Y_out)

        x_norm = T(2) * (T(i) - one(T)) / T(max(X_out - 1, 1)) - one(T)
        y_norm = T(2) * (T(j) - one(T)) / T(max(Y_out - 1, 1)) - one(T)

        if d == 1
            @inbounds grid[idx] = x_norm
        else
            @inbounds grid[idx] = y_norm
        end
    end

    return grid
end

"""
    _scale_displacement_values(disp, scale_x, scale_y, scale_z)

Scale displacement values by resolution ratio (GPU-accelerated).
"""
function _scale_displacement_values(
    disp::AbstractArray{T,5},
    scale_x::T, scale_y::T, scale_z::T
) where T
    X, Y, Z, D, N = size(disp)
    output = similar(disp)

    AK.foreachindex(output) do idx
        i, j, k, d, n = _linear_to_cartesian_5d_disp(idx, X, Y, Z, D)

        @inbounds val = disp[idx]

        # Scale based on which displacement component
        if d == 1
            @inbounds output[idx] = val * scale_x
        elseif d == 2
            @inbounds output[idx] = val * scale_y
        else
            @inbounds output[idx] = val * scale_z
        end
    end

    return output
end

function _scale_displacement_values_2d(
    disp::AbstractArray{T,4},
    scale_x::T, scale_y::T
) where T
    X, Y, D, N = size(disp)
    output = similar(disp)

    AK.foreachindex(output) do idx
        i, j, d, n = _linear_to_cartesian_4d_disp(idx, X, Y, D)

        @inbounds val = disp[idx]

        if d == 1
            @inbounds output[idx] = val * scale_x
        else
            @inbounds output[idx] = val * scale_y
        end
    end

    return output
end

# Index conversion helpers
@inline function _linear_to_cartesian_5d_resample(idx::Int, X::Int, Y::Int, Z::Int)
    idx_0 = idx - 1
    d = idx_0 % 3 + 1
    idx_0 = idx_0 ÷ 3
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    n = idx_0 ÷ Z + 1
    return d, i, j, k, n
end

@inline function _linear_to_cartesian_4d_resample(idx::Int, X::Int, Y::Int)
    idx_0 = idx - 1
    d = idx_0 % 2 + 1
    idx_0 = idx_0 ÷ 2
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    n = idx_0 ÷ Y + 1
    return d, i, j, n
end

@inline function _linear_to_cartesian_5d_disp(idx::Int, X::Int, Y::Int, Z::Int, D::Int)
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

@inline function _linear_to_cartesian_4d_disp(idx::Int, X::Int, Y::Int, D::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    d = idx_0 % D + 1
    n = idx_0 ÷ D + 1
    return i, j, d, n
end

# ============================================================================
# Resample Velocity Field (for SyN)
# ============================================================================

"""
    resample_velocity(v::AbstractArray{T,5}, target_size::NTuple{3,Int}; interpolation=:bilinear) where T

Resample a SyN velocity field to a new spatial size, scaling values appropriately.

For velocity fields, the scaling is the same as displacement fields since
velocity represents rate of change in normalized coordinates.

# Arguments
- `v`: Velocity field of shape (X, Y, Z, 3, N)
- `target_size`: Target spatial size (X_out, Y_out, Z_out)
- `interpolation`: Interpolation mode (:bilinear default)

# Returns
- Resampled velocity field
"""
function resample_velocity(
    v::AbstractArray{T,5},
    target_size::NTuple{3,Int};
    interpolation::Symbol=:bilinear
) where T
    # Velocity fields have the same scaling requirements as displacement fields
    return resample_displacement(v, target_size; interpolation=interpolation)
end

function resample_velocity(
    v::AbstractArray{T,4},
    target_size::NTuple{2,Int};
    interpolation::Symbol=:bilinear
) where T
    return resample_displacement(v, target_size; interpolation=interpolation)
end

# ============================================================================
# Upsample Affine Transform
# ============================================================================

"""
    upsample_affine_transform(theta::AbstractArray{T,3}, old_size, new_size) where T

Adjust an affine transformation matrix for a change in image resolution.

For normalized coordinate systems ([-1, 1]), the affine matrix itself doesn't
need scaling because it operates in normalized space. However, if the affine
was computed at a lower resolution and there are edge effects, this function
can interpolate the transformation appropriately.

# Arguments
- `theta`: Affine matrix of shape (D, D+1, N) where D=2 or D=3
- `old_size`: Original image size (tuple)
- `new_size`: New image size (tuple)

# Returns
- Affine matrix (same shape, potentially with interpolated values)

# Note
For standard affine registration in normalized coordinates, the affine matrix
is resolution-independent. This function is provided for completeness and for
cases where edge handling differs between resolutions.
"""
function upsample_affine_transform(
    theta::AbstractArray{T,3},
    old_size::NTuple{2,Int},
    new_size::NTuple{2,Int}
) where T
    # For normalized coordinate systems, affine is resolution-independent
    # The matrix operates in [-1, 1] normalized space
    return copy(theta)
end

function upsample_affine_transform(
    theta::AbstractArray{T,3},
    old_size::NTuple{3,Int},
    new_size::NTuple{3,Int}
) where T
    # Same reasoning - affine is resolution-independent in normalized coordinates
    return copy(theta)
end

"""
    upsample_affine_transform_physical(theta, old_spacing, new_spacing)

Adjust an affine transformation for physical coordinate changes.

When working in physical (mm) coordinates, changing image spacing requires
adjusting the affine matrix to maintain the same physical transformation.

# Arguments
- `theta`: Affine matrix (D, D+1, N)
- `old_spacing`: Original voxel spacing (mm)
- `new_spacing`: New voxel spacing (mm)

# Returns
- Adjusted affine matrix
"""
function upsample_affine_transform_physical(
    theta::AbstractArray{T,3},
    old_spacing::NTuple{3,<:Real},
    new_spacing::NTuple{3,<:Real}
) where T
    # For physical coordinates, we need to adjust for spacing changes
    # S_new^-1 * theta * S_old where S is diagonal scaling matrix

    D = size(theta, 1)
    D_plus_1 = size(theta, 2)
    N = size(theta, 3)

    output = similar(theta)

    # Scale factors: new_spacing / old_spacing
    sx = T(new_spacing[1]) / T(old_spacing[1])
    sy = T(new_spacing[2]) / T(old_spacing[2])
    sz = T(new_spacing[3]) / T(old_spacing[3])

    # Inverse scale factors
    inv_sx = T(old_spacing[1]) / T(new_spacing[1])
    inv_sy = T(old_spacing[2]) / T(new_spacing[2])
    inv_sz = T(old_spacing[3]) / T(new_spacing[3])

    scales = (sx, sy, sz)
    inv_scales = (inv_sx, inv_sy, inv_sz)

    AK.foreachindex(output) do idx
        d, col, n = _linear_to_cartesian_3d_theta(idx, D, D_plus_1)

        @inbounds val = theta[idx]

        if col <= D
            # Linear part: apply inv_scale[d] * val * scale[col]
            @inbounds output[idx] = inv_scales[d] * val * scales[col]
        else
            # Translation part: scale by inv_scale[d]
            @inbounds output[idx] = inv_scales[d] * val
        end
    end

    return output
end

function upsample_affine_transform_physical(
    theta::AbstractArray{T,3},
    old_spacing::NTuple{2,<:Real},
    new_spacing::NTuple{2,<:Real}
) where T
    D = size(theta, 1)
    D_plus_1 = size(theta, 2)
    N = size(theta, 3)

    output = similar(theta)

    sx = T(new_spacing[1]) / T(old_spacing[1])
    sy = T(new_spacing[2]) / T(old_spacing[2])

    inv_sx = T(old_spacing[1]) / T(new_spacing[1])
    inv_sy = T(old_spacing[2]) / T(new_spacing[2])

    scales = (sx, sy)
    inv_scales = (inv_sx, inv_sy)

    AK.foreachindex(output) do idx
        d, col, n = _linear_to_cartesian_3d_theta(idx, D, D_plus_1)

        @inbounds val = theta[idx]

        if col <= D
            @inbounds output[idx] = inv_scales[d] * val * scales[col]
        else
            @inbounds output[idx] = inv_scales[d] * val
        end
    end

    return output
end

@inline function _linear_to_cartesian_3d_theta(idx::Int, D::Int, D_plus_1::Int)
    idx_0 = idx - 1
    d = idx_0 % D + 1
    idx_0 = idx_0 ÷ D
    col = idx_0 % D_plus_1 + 1
    n = idx_0 ÷ D_plus_1 + 1
    return d, col, n
end

# ============================================================================
# Invert Displacement Field
# ============================================================================

"""
    invert_displacement(disp::AbstractArray{T,5}; iterations::Int=10, tolerance::T=T(1e-6)) where T

Compute the inverse of a displacement field using fixed-point iteration.

For a displacement field φ where: y = x + φ(x)
The inverse ψ satisfies: x = y + ψ(y)  i.e., ψ = -φ(x+ψ)

# Arguments
- `disp`: Displacement field of shape (X, Y, Z, 3, N)
- `iterations`: Number of fixed-point iterations (default: 10)
- `tolerance`: Convergence tolerance (currently unused, for future early stopping)

# Returns
- Inverse displacement field of same shape

# Algorithm
Uses fixed-point iteration: ψ_{n+1} = -φ(id + ψ_n)
Starting from ψ_0 = -φ

# Example
```julia
# Compute inverse for bidirectional registration
flow_forward = diffeomorphic_transform(v_xy)
flow_inverse = invert_displacement(flow_forward)

# Verify: x + forward + inverse(x + forward) ≈ x
```

# Note
This method works well for smooth, relatively small displacements.
For large displacements, more iterations may be needed.
"""
function invert_displacement(
    disp::AbstractArray{T,5};
    iterations::Int=10,
    tolerance::T=T(1e-6),
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    X, Y, Z, D, N = size(disp)
    @assert D == 3 "Displacement field must have 3 channels"

    # Initialize with negative of input
    inv_disp = similar(disp)
    AK.foreachindex(inv_disp) do idx
        @inbounds inv_disp[idx] = -disp[idx]
    end

    # Fixed-point iteration: ψ_{n+1} = -φ(id + ψ_n)
    for iter in 1:iterations
        # Sample disp at (identity + current inverse)
        # This requires creating a grid and sampling
        sampled = _sample_at_displaced(disp, inv_disp; align_corners=align_corners, padding_mode=padding_mode)

        # Update: ψ = -φ(id + ψ)
        AK.foreachindex(inv_disp) do idx
            @inbounds inv_disp[idx] = -sampled[idx]
        end
    end

    return inv_disp
end

# 2D version
function invert_displacement(
    disp::AbstractArray{T,4};
    iterations::Int=10,
    tolerance::T=T(1e-6),
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    X, Y, D, N = size(disp)
    @assert D == 2 "2D displacement field must have 2 channels"

    inv_disp = similar(disp)
    AK.foreachindex(inv_disp) do idx
        @inbounds inv_disp[idx] = -disp[idx]
    end

    for iter in 1:iterations
        sampled = _sample_at_displaced_2d(disp, inv_disp; align_corners=align_corners, padding_mode=padding_mode)

        AK.foreachindex(inv_disp) do idx
            @inbounds inv_disp[idx] = -sampled[idx]
        end
    end

    return inv_disp
end

"""
    _sample_at_displaced(disp, offset_disp)

Sample displacement field at positions (identity + offset_disp).
"""
function _sample_at_displaced(
    disp::AbstractArray{T,5},
    offset_disp::AbstractArray{T,5};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    X, Y, Z, D, N = size(disp)

    # Create grid: identity + offset
    grid = _create_displaced_grid_3d(offset_disp)

    # Sample disp at grid positions
    return grid_sample(disp, grid; padding_mode=padding_mode, align_corners=align_corners)
end

function _sample_at_displaced_2d(
    disp::AbstractArray{T,4},
    offset_disp::AbstractArray{T,4};
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    grid = _create_displaced_grid_2d(offset_disp)
    return grid_sample(disp, grid; padding_mode=padding_mode, align_corners=align_corners)
end

"""
    _create_displaced_grid_3d(offset_disp)

Create a sampling grid: identity + offset_disp.
offset_disp is (X, Y, Z, 3, N), output grid is (3, X, Y, Z, N).
"""
function _create_displaced_grid_3d(offset_disp::AbstractArray{T,5}) where T
    X, Y, Z, D, N = size(offset_disp)

    grid = similar(offset_disp, 3, X, Y, Z, N)

    # Pre-compute normalization factors
    norm_x = T(2) / T(max(X - 1, 1))
    norm_y = T(2) / T(max(Y - 1, 1))
    norm_z = T(2) / T(max(Z - 1, 1))

    AK.foreachindex(grid) do idx
        d, i, j, k, n = _linear_to_cartesian_5d_resample(idx, X, Y, Z)

        # Identity coordinate
        if d == 1
            id_coord = (T(i) - one(T)) * norm_x - one(T)
            @inbounds grid[idx] = id_coord + offset_disp[i, j, k, 1, n]
        elseif d == 2
            id_coord = (T(j) - one(T)) * norm_y - one(T)
            @inbounds grid[idx] = id_coord + offset_disp[i, j, k, 2, n]
        else
            id_coord = (T(k) - one(T)) * norm_z - one(T)
            @inbounds grid[idx] = id_coord + offset_disp[i, j, k, 3, n]
        end
    end

    return grid
end

function _create_displaced_grid_2d(offset_disp::AbstractArray{T,4}) where T
    X, Y, D, N = size(offset_disp)

    grid = similar(offset_disp, 2, X, Y, N)

    norm_x = T(2) / T(max(X - 1, 1))
    norm_y = T(2) / T(max(Y - 1, 1))

    AK.foreachindex(grid) do idx
        d, i, j, n = _linear_to_cartesian_4d_resample(idx, X, Y)

        if d == 1
            id_coord = (T(i) - one(T)) * norm_x - one(T)
            @inbounds grid[idx] = id_coord + offset_disp[i, j, 1, n]
        else
            id_coord = (T(j) - one(T)) * norm_y - one(T)
            @inbounds grid[idx] = id_coord + offset_disp[i, j, 2, n]
        end
    end

    return grid
end

# ============================================================================
# Backward Passes for Mooncake rrule!!
# ============================================================================

# Gradient for resample_displacement
function _∇resample_displacement_3d!(
    d_disp::AbstractArray{T,5},
    d_output::AbstractArray{T,5},
    scale_x::T, scale_y::T, scale_z::T,
    grid::AbstractArray{T,5},
    disp_shape::NTuple{5,Int};
    padding_mode::Symbol=:border,
    align_corners::Bool=true
) where T
    X_in, Y_in, Z_in, D, N = disp_shape
    X_out, Y_out, Z_out = size(d_output, 1), size(d_output, 2), size(d_output, 3)

    # Unscale the output gradient (reverse the scaling)
    d_output_unscaled = similar(d_output)
    AK.foreachindex(d_output_unscaled) do idx
        i, j, k, d, n = _linear_to_cartesian_5d_disp(idx, X_out, Y_out, Z_out, D)

        @inbounds val = d_output[idx]

        if d == 1
            @inbounds d_output_unscaled[idx] = val * scale_x
        elseif d == 2
            @inbounds d_output_unscaled[idx] = val * scale_y
        else
            @inbounds d_output_unscaled[idx] = val * scale_z
        end
    end

    # Backprop through grid_sample
    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    ac = Val(align_corners)
    _∇grid_sample_input_3d!(d_disp, d_output_unscaled, grid, pm, ac)

    return nothing
end

function _∇resample_displacement_2d!(
    d_disp::AbstractArray{T,4},
    d_output::AbstractArray{T,4},
    scale_x::T, scale_y::T,
    grid::AbstractArray{T,4},
    disp_shape::NTuple{4,Int};
    padding_mode::Symbol=:border,
    align_corners::Bool=true
) where T
    X_in, Y_in, D, N = disp_shape
    X_out, Y_out = size(d_output, 1), size(d_output, 2)

    d_output_unscaled = similar(d_output)
    AK.foreachindex(d_output_unscaled) do idx
        i, j, d, n = _linear_to_cartesian_4d_disp(idx, X_out, Y_out, D)

        @inbounds val = d_output[idx]

        if d == 1
            @inbounds d_output_unscaled[idx] = val * scale_x
        else
            @inbounds d_output_unscaled[idx] = val * scale_y
        end
    end

    pm = padding_mode === :zeros ? ZerosPadding() : BorderPadding()
    ac = Val(align_corners)
    _∇grid_sample_input_2d!(d_disp, d_output_unscaled, grid, pm, ac)

    return nothing
end

# Gradient for invert_displacement (simplified - unrolled iterations)
function _∇invert_displacement_3d!(
    d_disp::AbstractArray{T,5},
    d_inv::AbstractArray{T,5},
    disp::AbstractArray{T,5},
    inv_disp::AbstractArray{T,5},
    iterations::Int;
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where T
    # For fixed-point iteration: ψ_{n+1} = -φ(id + ψ_n)
    # Gradient is complex due to unrolling
    # Simplified: d_disp ≈ -d_inv (first-order approximation)
    AK.foreachindex(d_disp) do idx
        @inbounds d_disp[idx] += -d_inv[idx]
    end

    return nothing
end

# ============================================================================
# Mooncake rrule!! Definitions
# ============================================================================

# Mark as primitives
@is_primitive MinimalCtx Tuple{typeof(resample_displacement), AbstractArray{<:Any,5}, NTuple{3,Int}}
@is_primitive MinimalCtx Tuple{typeof(resample_displacement), AbstractArray{<:Any,4}, NTuple{2,Int}}
@is_primitive MinimalCtx Tuple{typeof(resample_velocity), AbstractArray{<:Any,5}, NTuple{3,Int}}
@is_primitive MinimalCtx Tuple{typeof(resample_velocity), AbstractArray{<:Any,4}, NTuple{2,Int}}
@is_primitive MinimalCtx Tuple{typeof(invert_displacement), AbstractArray{<:Any,5}}
@is_primitive MinimalCtx Tuple{typeof(invert_displacement), AbstractArray{<:Any,4}}

# rrule!! for resample_displacement 3D
function Mooncake.rrule!!(
    ::CoDual{typeof(resample_displacement)},
    disp::CoDual{A, F},
    target_size::CoDual{S, NoFData};
    interpolation::Symbol=:bilinear
) where {A<:AbstractArray{<:Any,5}, F, S<:NTuple{3,Int}}
    disp_primal = disp.x
    disp_fdata = disp.dx
    target_size_primal = target_size.x

    X_in, Y_in, Z_in, D, N = size(disp_primal)
    X_out, Y_out, Z_out = target_size_primal
    T = eltype(disp_primal)

    # Compute scale factors
    scale_x = T(X_out - 1) / T(max(X_in - 1, 1))
    scale_y = T(Y_out - 1) / T(max(Y_in - 1, 1))
    scale_z = T(Z_out - 1) / T(max(Z_in - 1, 1))

    # Forward pass
    grid = _create_resample_grid_for_disp(disp_primal, target_size_primal)
    output = resample_displacement(disp_primal, target_size_primal; interpolation=interpolation)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function resample_displacement_3d_pullback(_rdata)
        _∇resample_displacement_3d!(
            disp_fdata, output_fdata, scale_x, scale_y, scale_z, grid,
            size(disp_primal); padding_mode=:border, align_corners=true
        )
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), resample_displacement_3d_pullback
end

# rrule!! for resample_displacement 2D
function Mooncake.rrule!!(
    ::CoDual{typeof(resample_displacement)},
    disp::CoDual{A, F},
    target_size::CoDual{S, NoFData};
    interpolation::Symbol=:bilinear
) where {A<:AbstractArray{<:Any,4}, F, S<:NTuple{2,Int}}
    disp_primal = disp.x
    disp_fdata = disp.dx
    target_size_primal = target_size.x

    X_in, Y_in, D, N = size(disp_primal)
    X_out, Y_out = target_size_primal
    T = eltype(disp_primal)

    scale_x = T(X_out - 1) / T(max(X_in - 1, 1))
    scale_y = T(Y_out - 1) / T(max(Y_in - 1, 1))

    grid = _create_resample_grid_for_disp_2d(disp_primal, target_size_primal)
    output = resample_displacement(disp_primal, target_size_primal; interpolation=interpolation)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function resample_displacement_2d_pullback(_rdata)
        _∇resample_displacement_2d!(
            disp_fdata, output_fdata, scale_x, scale_y, grid,
            size(disp_primal); padding_mode=:border, align_corners=true
        )
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), resample_displacement_2d_pullback
end

# rrule!! for resample_velocity (same as resample_displacement)
function Mooncake.rrule!!(
    ::CoDual{typeof(resample_velocity)},
    v::CoDual{A, F},
    target_size::CoDual{S, NoFData};
    interpolation::Symbol=:bilinear
) where {A<:AbstractArray{<:Any,5}, F, S<:NTuple{3,Int}}
    # Delegate to resample_displacement
    return Mooncake.rrule!!(
        CoDual(resample_displacement, NoFData()),
        v, target_size;
        interpolation=interpolation
    )
end

function Mooncake.rrule!!(
    ::CoDual{typeof(resample_velocity)},
    v::CoDual{A, F},
    target_size::CoDual{S, NoFData};
    interpolation::Symbol=:bilinear
) where {A<:AbstractArray{<:Any,4}, F, S<:NTuple{2,Int}}
    return Mooncake.rrule!!(
        CoDual(resample_displacement, NoFData()),
        v, target_size;
        interpolation=interpolation
    )
end

# rrule!! for invert_displacement 3D
function Mooncake.rrule!!(
    ::CoDual{typeof(invert_displacement)},
    disp::CoDual{A, F};
    iterations::Int=10,
    tolerance::Real=1e-6,
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where {A<:AbstractArray{<:Any,5}, F}
    disp_primal = disp.x
    disp_fdata = disp.dx
    T = eltype(disp_primal)

    output = invert_displacement(disp_primal; iterations=iterations,
                                  tolerance=T(tolerance), align_corners=align_corners,
                                  padding_mode=padding_mode)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function invert_displacement_3d_pullback(_rdata)
        _∇invert_displacement_3d!(disp_fdata, output_fdata, disp_primal, output, iterations;
                                   align_corners=align_corners, padding_mode=padding_mode)
        return NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), invert_displacement_3d_pullback
end

# rrule!! for invert_displacement 2D
function Mooncake.rrule!!(
    ::CoDual{typeof(invert_displacement)},
    disp::CoDual{A, F};
    iterations::Int=10,
    tolerance::Real=1e-6,
    align_corners::Bool=true,
    padding_mode::Symbol=:border
) where {A<:AbstractArray{<:Any,4}, F}
    disp_primal = disp.x
    disp_fdata = disp.dx
    T = eltype(disp_primal)

    output = invert_displacement(disp_primal; iterations=iterations,
                                  tolerance=T(tolerance), align_corners=align_corners,
                                  padding_mode=padding_mode)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function invert_displacement_2d_pullback(_rdata)
        # Simplified gradient
        AK.foreachindex(disp_fdata) do idx
            @inbounds disp_fdata[idx] += -output_fdata[idx]
        end
        return NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), invert_displacement_2d_pullback
end
