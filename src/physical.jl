# Physical coordinate support for MedicalImageRegistration.jl
# GPU-first architecture with AK.foreachindex + Mooncake rrule!!
#
# Handles anisotropic voxels (e.g., 3mm vs 0.5mm slice thickness in cardiac CT)
# by working in physical (mm) coordinates instead of normalized [-1, 1] coordinates.

# ============================================================================
# PhysicalImage Type
# ============================================================================

"""
    PhysicalImage{T, N, A<:AbstractArray{T}}

Image with physical coordinate information for proper handling of anisotropic voxels.

# Fields
- `data`: The image array (X, Y, [Z], C, N) where C=channels, N=batch
- `spacing`: Voxel spacing in mm - tuple of (sx, sy, [sz]) for spatial dimensions
- `origin`: Physical position of first voxel center in mm

# Constructor
```julia
# 3D image with 0.5mm x 0.5mm x 3mm voxels
img = PhysicalImage(data; spacing=(0.5, 0.5, 3.0), origin=(0.0, 0.0, 0.0))

# 2D image with default 1mm isotropic spacing
img = PhysicalImage(data_2d)
```

# Physical coordinates
- Voxel (i, j, k) at 1-indexed position maps to physical position:
  - x_mm = origin[1] + (i - 1) * spacing[1]
  - y_mm = origin[2] + (j - 1) * spacing[2]
  - z_mm = origin[3] + (k - 1) * spacing[3]

# Why this matters
For cardiac CT with 3mm static and 0.5mm moving:
- Without spacing: 10 voxels = same displacement (WRONG!)
- With spacing: 10 voxels in 3mm = 30mm, 10 voxels in 0.5mm = 5mm (CORRECT!)
"""
struct PhysicalImage{T, N, A<:AbstractArray{T,N}}
    data::A
    spacing::NTuple{3, T}  # Always 3D spacing (2D images use sz=1)
    origin::NTuple{3, T}   # Always 3D origin (2D images use oz=0)
end

# ============================================================================
# Constructors
# ============================================================================

"""
    PhysicalImage(data::AbstractArray{T,4}; spacing=(1,1), origin=(0,0)) where T

Create a 2D PhysicalImage (X, Y, C, N) with given spacing.
"""
function PhysicalImage(
    data::AbstractArray{T,4};
    spacing::NTuple{2,<:Real}=(one(T), one(T)),
    origin::NTuple{2,<:Real}=(zero(T), zero(T))
) where T
    spacing_3d = (T(spacing[1]), T(spacing[2]), one(T))
    origin_3d = (T(origin[1]), T(origin[2]), zero(T))
    return PhysicalImage{T, 4, typeof(data)}(data, spacing_3d, origin_3d)
end

"""
    PhysicalImage(data::AbstractArray{T,5}; spacing=(1,1,1), origin=(0,0,0)) where T

Create a 3D PhysicalImage (X, Y, Z, C, N) with given spacing.
"""
function PhysicalImage(
    data::AbstractArray{T,5};
    spacing::NTuple{3,<:Real}=(one(T), one(T), one(T)),
    origin::NTuple{3,<:Real}=(zero(T), zero(T), zero(T))
) where T
    spacing_t = (T(spacing[1]), T(spacing[2]), T(spacing[3]))
    origin_t = (T(origin[1]), T(origin[2]), T(origin[3]))
    return PhysicalImage{T, 5, typeof(data)}(data, spacing_t, origin_t)
end

# ============================================================================
# Basic Accessors and Properties
# ============================================================================

"""Get the underlying array data."""
Base.parent(img::PhysicalImage) = img.data

"""Get array element type."""
Base.eltype(::PhysicalImage{T}) where T = T

"""Get array size."""
Base.size(img::PhysicalImage) = size(img.data)
Base.size(img::PhysicalImage, d::Int) = size(img.data, d)

"""Get number of dimensions."""
Base.ndims(img::PhysicalImage) = ndims(img.data)

"""Check if 3D (5-dimensional array) or 2D (4-dimensional array)."""
is_3d(img::PhysicalImage{T, 5}) where T = true
is_3d(img::PhysicalImage{T, 4}) where T = false

"""Get spatial dimensions (excluding channel and batch)."""
function spatial_size(img::PhysicalImage{T, 4}) where T
    return (size(img.data, 1), size(img.data, 2))
end

function spatial_size(img::PhysicalImage{T, 5}) where T
    return (size(img.data, 1), size(img.data, 2), size(img.data, 3))
end

"""Get spatial spacing (excluding channel and batch dimensions)."""
function spatial_spacing(img::PhysicalImage{T, 4}) where T
    return (img.spacing[1], img.spacing[2])
end

function spatial_spacing(img::PhysicalImage{T, 5}) where T
    return img.spacing
end

"""
    physical_extent(img::PhysicalImage)

Get the physical extent (size in mm) of the image.
"""
function physical_extent(img::PhysicalImage{T, 4}) where T
    sx, sy, _ = img.spacing
    X, Y = spatial_size(img)
    return ((X - 1) * sx, (Y - 1) * sy)
end

function physical_extent(img::PhysicalImage{T, 5}) where T
    sx, sy, sz = img.spacing
    X, Y, Z = spatial_size(img)
    return ((X - 1) * sx, (Y - 1) * sy, (Z - 1) * sz)
end

"""
    physical_bounds(img::PhysicalImage)

Get (min, max) physical coordinates for each spatial dimension.
"""
function physical_bounds(img::PhysicalImage{T, 4}) where T
    ox, oy, _ = img.origin
    ex, ey = physical_extent(img)
    return ((ox, ox + ex), (oy, oy + ey))
end

function physical_bounds(img::PhysicalImage{T, 5}) where T
    ox, oy, oz = img.origin
    ex, ey, ez = physical_extent(img)
    return ((ox, ox + ex), (oy, oy + ey), (oz, oz + ez))
end

# ============================================================================
# Coordinate Transformations
# ============================================================================

"""
    voxel_to_physical(img::PhysicalImage, i, j[, k])

Convert 1-indexed voxel coordinates to physical coordinates (mm).
"""
function voxel_to_physical(img::PhysicalImage{T, 4}, i::Real, j::Real) where T
    ox, oy, _ = img.origin
    sx, sy, _ = img.spacing
    x_mm = ox + (i - 1) * sx
    y_mm = oy + (j - 1) * sy
    return (T(x_mm), T(y_mm))
end

function voxel_to_physical(img::PhysicalImage{T, 5}, i::Real, j::Real, k::Real) where T
    ox, oy, oz = img.origin
    sx, sy, sz = img.spacing
    x_mm = ox + (i - 1) * sx
    y_mm = oy + (j - 1) * sy
    z_mm = oz + (k - 1) * sz
    return (T(x_mm), T(y_mm), T(z_mm))
end

"""
    physical_to_voxel(img::PhysicalImage, x_mm, y_mm[, z_mm])

Convert physical coordinates (mm) to 1-indexed voxel coordinates.
Returns fractional coordinates (for interpolation).
"""
function physical_to_voxel(img::PhysicalImage{T, 4}, x_mm::Real, y_mm::Real) where T
    ox, oy, _ = img.origin
    sx, sy, _ = img.spacing
    i = (x_mm - ox) / sx + 1
    j = (y_mm - oy) / sy + 1
    return (T(i), T(j))
end

function physical_to_voxel(img::PhysicalImage{T, 5}, x_mm::Real, y_mm::Real, z_mm::Real) where T
    ox, oy, oz = img.origin
    sx, sy, sz = img.spacing
    i = (x_mm - ox) / sx + 1
    j = (y_mm - oy) / sy + 1
    k = (z_mm - oz) / sz + 1
    return (T(i), T(j), T(k))
end

"""
    voxel_to_normalized(img::PhysicalImage, i, j[, k])

Convert 1-indexed voxel coordinates to normalized [-1, 1] coordinates.
Assumes align_corners=true (corners at -1 and +1).
"""
function voxel_to_normalized(img::PhysicalImage{T, 4}, i::Real, j::Real) where T
    X, Y = spatial_size(img)
    x_norm = X > 1 ? T(2) * (i - 1) / (X - 1) - one(T) : zero(T)
    y_norm = Y > 1 ? T(2) * (j - 1) / (Y - 1) - one(T) : zero(T)
    return (x_norm, y_norm)
end

function voxel_to_normalized(img::PhysicalImage{T, 5}, i::Real, j::Real, k::Real) where T
    X, Y, Z = spatial_size(img)
    x_norm = X > 1 ? T(2) * (i - 1) / (X - 1) - one(T) : zero(T)
    y_norm = Y > 1 ? T(2) * (j - 1) / (Y - 1) - one(T) : zero(T)
    z_norm = Z > 1 ? T(2) * (k - 1) / (Z - 1) - one(T) : zero(T)
    return (x_norm, y_norm, z_norm)
end

"""
    normalized_to_voxel(img::PhysicalImage, x_norm, y_norm[, z_norm])

Convert normalized [-1, 1] coordinates to 1-indexed voxel coordinates.
"""
function normalized_to_voxel(img::PhysicalImage{T, 4}, x_norm::Real, y_norm::Real) where T
    X, Y = spatial_size(img)
    i = (x_norm + one(T)) / 2 * (X - 1) + 1
    j = (y_norm + one(T)) / 2 * (Y - 1) + 1
    return (T(i), T(j))
end

function normalized_to_voxel(img::PhysicalImage{T, 5}, x_norm::Real, y_norm::Real, z_norm::Real) where T
    X, Y, Z = spatial_size(img)
    i = (x_norm + one(T)) / 2 * (X - 1) + 1
    j = (y_norm + one(T)) / 2 * (Y - 1) + 1
    k = (z_norm + one(T)) / 2 * (Z - 1) + 1
    return (T(i), T(j), T(k))
end

# ============================================================================
# Spacing-Aware Affine Grid Generation
# ============================================================================

"""
    affine_grid_physical(theta, size, spacing; align_corners=true)

Generate a sampling grid from an affine transformation matrix, accounting for
anisotropic voxel spacing. The affine transformation is applied in physical
coordinates (mm), then converted back to normalized coordinates for grid_sample.

# Arguments
- `theta`: Affine transformation matrix (D, D+1, N) in physical units (mm)
- `size`: Output size tuple (X, Y[, Z]) or (X, Y[, Z], C, N)
- `spacing`: Voxel spacing in mm (sx, sy[, sz])

# Returns
- Grid of normalized coordinates for use with grid_sample

# Example
```julia
# 3D image with anisotropic spacing (0.5mm × 0.5mm × 3mm)
spacing = (0.5f0, 0.5f0, 3.0f0)
theta = create_identity_affine(3, 1)  # Identity transform
grid = affine_grid_physical(theta, (128, 128, 50), spacing)
```

# Why this matters
Standard affine_grid assumes isotropic voxels. A 10° rotation looks different
in physical space when voxels are 0.5mm vs 3mm. This function ensures the
affine transformation is geometrically correct in physical (mm) coordinates.
"""
function affine_grid_physical(
    theta::AbstractArray{T,3},
    size::NTuple{2,Int},
    spacing::NTuple{2,<:Real};
    align_corners::Bool=true
) where T
    return _affine_grid_physical_2d(theta, size, (T(spacing[1]), T(spacing[2])), Val(align_corners))
end

function affine_grid_physical(
    theta::AbstractArray{T,3},
    size::NTuple{4,Int},
    spacing::NTuple{2,<:Real};
    align_corners::Bool=true
) where T
    return _affine_grid_physical_2d(theta, (size[1], size[2]), (T(spacing[1]), T(spacing[2])), Val(align_corners))
end

function affine_grid_physical(
    theta::AbstractArray{T,3},
    size::NTuple{3,Int},
    spacing::NTuple{3,<:Real};
    align_corners::Bool=true
) where T
    return _affine_grid_physical_3d(theta, size, (T(spacing[1]), T(spacing[2]), T(spacing[3])), Val(align_corners))
end

function affine_grid_physical(
    theta::AbstractArray{T,3},
    size::NTuple{5,Int},
    spacing::NTuple{3,<:Real};
    align_corners::Bool=true
) where T
    return _affine_grid_physical_3d(theta, (size[1], size[2], size[3]), (T(spacing[1]), T(spacing[2]), T(spacing[3])), Val(align_corners))
end

# Convenience: accept PhysicalImage directly
function affine_grid_physical(
    theta::AbstractArray{T,3},
    img::PhysicalImage{T, 4};
    align_corners::Bool=true
) where T
    return _affine_grid_physical_2d(theta, spatial_size(img), spatial_spacing(img), Val(align_corners))
end

function affine_grid_physical(
    theta::AbstractArray{T,3},
    img::PhysicalImage{T, 5};
    align_corners::Bool=true
) where T
    return _affine_grid_physical_3d(theta, spatial_size(img), spatial_spacing(img), Val(align_corners))
end

# ============================================================================
# 2D Physical Affine Grid (GPU-accelerated)
# ============================================================================

function _affine_grid_physical_2d(
    theta::AbstractArray{T,3},
    size::NTuple{2,Int},
    spacing::NTuple{2,T},
    ::Val{AC}
) where {T, AC}
    @assert Base.size(theta, 1) == 2 "theta must have 2 rows for 2D"
    @assert Base.size(theta, 2) == 3 "theta must have 3 columns for 2D (2x3 matrix)"

    X_out, Y_out = size
    N = Base.size(theta, 3)
    sx, sy = spacing

    # Compute physical extent for normalization
    extent_x = T(X_out - 1) * sx
    extent_y = T(Y_out - 1) * sy
    max_extent_raw = max(extent_x, extent_y)

    # Avoid division by zero for single-voxel dimensions (use max for type stability)
    max_extent = max(max_extent_raw, one(T))

    # Pre-compute normalization factors for the kernel
    half_ex = extent_x / 2
    half_ey = extent_y / 2
    norm_scale = T(2) / max_extent

    grid = similar(theta, 2, X_out, Y_out, N)

    AK.foreachindex(grid) do idx
        coord, i, j, n = _linear_to_cartesian_4d_phys(idx, X_out, Y_out)

        # Generate physical coordinates centered at image center
        # Range: [-extent/2, +extent/2] in mm, then normalize
        if AC
            phys_x = ((T(i) - one(T)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - one(T)) * sy - half_ey) * norm_scale
        else
            phys_x = ((T(i) - T(0.5)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - T(0.5)) * sy - half_ey) * norm_scale
        end

        # Apply affine transformation in normalized physical space
        if coord == 1
            @inbounds grid[idx] = theta[1, 1, n] * phys_x + theta[1, 2, n] * phys_y + theta[1, 3, n]
        else
            @inbounds grid[idx] = theta[2, 1, n] * phys_x + theta[2, 2, n] * phys_y + theta[2, 3, n]
        end
    end

    return grid
end

# ============================================================================
# 3D Physical Affine Grid (GPU-accelerated)
# ============================================================================

function _affine_grid_physical_3d(
    theta::AbstractArray{T,3},
    size::NTuple{3,Int},
    spacing::NTuple{3,T},
    ::Val{AC}
) where {T, AC}
    @assert Base.size(theta, 1) == 3 "theta must have 3 rows for 3D"
    @assert Base.size(theta, 2) == 4 "theta must have 4 columns for 3D (3x4 matrix)"

    X_out, Y_out, Z_out = size
    N = Base.size(theta, 3)
    sx, sy, sz = spacing

    # Compute physical extent for normalization
    extent_x = T(X_out - 1) * sx
    extent_y = T(Y_out - 1) * sy
    extent_z = T(Z_out - 1) * sz
    max_extent_raw = max(extent_x, extent_y, extent_z)

    # Avoid division by zero (use max for type stability)
    max_extent = max(max_extent_raw, one(T))

    # Pre-compute normalization factors for the kernel
    half_ex = extent_x / 2
    half_ey = extent_y / 2
    half_ez = extent_z / 2
    norm_scale = T(2) / max_extent

    grid = similar(theta, 3, X_out, Y_out, Z_out, N)

    AK.foreachindex(grid) do idx
        coord, i, j, k, n = _linear_to_cartesian_5d_phys(idx, X_out, Y_out, Z_out)

        # Generate physical coordinates centered at image center
        if AC
            phys_x = ((T(i) - one(T)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - one(T)) * sy - half_ey) * norm_scale
            phys_z = ((T(k) - one(T)) * sz - half_ez) * norm_scale
        else
            phys_x = ((T(i) - T(0.5)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - T(0.5)) * sy - half_ey) * norm_scale
            phys_z = ((T(k) - T(0.5)) * sz - half_ez) * norm_scale
        end

        # Apply affine transformation
        if coord == 1
            @inbounds grid[idx] = theta[1, 1, n] * phys_x + theta[1, 2, n] * phys_y + theta[1, 3, n] * phys_z + theta[1, 4, n]
        elseif coord == 2
            @inbounds grid[idx] = theta[2, 1, n] * phys_x + theta[2, 2, n] * phys_y + theta[2, 3, n] * phys_z + theta[2, 4, n]
        else
            @inbounds grid[idx] = theta[3, 1, n] * phys_x + theta[3, 2, n] * phys_y + theta[3, 3, n] * phys_z + theta[3, 4, n]
        end
    end

    return grid
end

# ============================================================================
# Index Conversion Helpers
# ============================================================================

@inline function _linear_to_cartesian_4d_phys(idx::Int, X::Int, Y::Int)
    idx_0 = idx - 1
    coord = idx_0 % 2 + 1
    idx_0 = idx_0 ÷ 2
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    n = idx_0 ÷ Y + 1
    return coord, i, j, n
end

@inline function _linear_to_cartesian_5d_phys(idx::Int, X::Int, Y::Int, Z::Int)
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
# Backward Pass for Physical Affine Grid
# ============================================================================

function _∇affine_grid_physical_theta_2d!(
    d_theta::AbstractArray{T,3},
    d_grid::AbstractArray{T,4},
    spacing::NTuple{2,T},
    ::Val{AC}
) where {T, AC}
    _, X_out, Y_out, N = Base.size(d_grid)
    sx, sy = spacing

    extent_x = T(X_out - 1) * sx
    extent_y = T(Y_out - 1) * sy
    max_extent = max(max(extent_x, extent_y), one(T))

    # Pre-compute for kernel
    half_ex = extent_x / 2
    half_ey = extent_y / 2
    norm_scale = T(2) / max_extent

    AK.foreachindex(d_grid) do idx
        coord, i, j, n = _linear_to_cartesian_4d_phys(idx, X_out, Y_out)

        d_val = @inbounds d_grid[idx]
        if d_val == zero(T)
            return nothing
        end

        if AC
            phys_x = ((T(i) - one(T)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - one(T)) * sy - half_ey) * norm_scale
        else
            phys_x = ((T(i) - T(0.5)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - T(0.5)) * sy - half_ey) * norm_scale
        end

        Atomix.@atomic d_theta[coord, 1, n] += d_val * phys_x
        Atomix.@atomic d_theta[coord, 2, n] += d_val * phys_y
        Atomix.@atomic d_theta[coord, 3, n] += d_val

        return nothing
    end

    return nothing
end

function _∇affine_grid_physical_theta_3d!(
    d_theta::AbstractArray{T,3},
    d_grid::AbstractArray{T,5},
    spacing::NTuple{3,T},
    ::Val{AC}
) where {T, AC}
    _, X_out, Y_out, Z_out, N = Base.size(d_grid)
    sx, sy, sz = spacing

    extent_x = T(X_out - 1) * sx
    extent_y = T(Y_out - 1) * sy
    extent_z = T(Z_out - 1) * sz
    max_extent = max(max(extent_x, extent_y, extent_z), one(T))

    # Pre-compute for kernel
    half_ex = extent_x / 2
    half_ey = extent_y / 2
    half_ez = extent_z / 2
    norm_scale = T(2) / max_extent

    AK.foreachindex(d_grid) do idx
        coord, i, j, k, n = _linear_to_cartesian_5d_phys(idx, X_out, Y_out, Z_out)

        d_val = @inbounds d_grid[idx]
        if d_val == zero(T)
            return nothing
        end

        if AC
            phys_x = ((T(i) - one(T)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - one(T)) * sy - half_ey) * norm_scale
            phys_z = ((T(k) - one(T)) * sz - half_ez) * norm_scale
        else
            phys_x = ((T(i) - T(0.5)) * sx - half_ex) * norm_scale
            phys_y = ((T(j) - T(0.5)) * sy - half_ey) * norm_scale
            phys_z = ((T(k) - T(0.5)) * sz - half_ez) * norm_scale
        end

        Atomix.@atomic d_theta[coord, 1, n] += d_val * phys_x
        Atomix.@atomic d_theta[coord, 2, n] += d_val * phys_y
        Atomix.@atomic d_theta[coord, 3, n] += d_val * phys_z
        Atomix.@atomic d_theta[coord, 4, n] += d_val

        return nothing
    end

    return nothing
end

# ============================================================================
# Mooncake rrule!! for Physical Affine Grid
# ============================================================================

# We need a wrapper function that captures spacing for Mooncake primitives
# Using a struct to hold the spacing as part of the function signature

struct AffineGridPhysical2D{T}
    spacing::NTuple{2, T}
end

struct AffineGridPhysical3D{T}
    spacing::NTuple{3, T}
end

function (f::AffineGridPhysical2D{T})(theta::AbstractArray{T,3}, size::NTuple{2,Int}; align_corners::Bool=true) where T
    return _affine_grid_physical_2d(theta, size, f.spacing, Val(align_corners))
end

function (f::AffineGridPhysical3D{T})(theta::AbstractArray{T,3}, size::NTuple{3,Int}; align_corners::Bool=true) where T
    return _affine_grid_physical_3d(theta, size, f.spacing, Val(align_corners))
end

# Mark as primitives
@is_primitive MinimalCtx Tuple{AffineGridPhysical2D, AbstractArray{<:Any,3}, NTuple{2,Int}}
@is_primitive MinimalCtx Tuple{AffineGridPhysical3D, AbstractArray{<:Any,3}, NTuple{3,Int}}

# 2D rrule!!
function Mooncake.rrule!!(
    func::CoDual{AffineGridPhysical2D{T}},
    theta::CoDual{A, F},
    size::CoDual{S, NoFData};
    align_corners::Bool=true
) where {T, A<:AbstractArray{T,3}, F, S<:NTuple{2,Int}}
    func_primal = func.x
    theta_primal = theta.x
    theta_fdata = theta.dx
    size_primal = size.x

    ac = Val(align_corners)
    spacing = func_primal.spacing

    output = func_primal(theta_primal, size_primal; align_corners)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function affine_grid_physical_2d_pullback(_rdata)
        _∇affine_grid_physical_theta_2d!(theta_fdata, output_fdata, spacing, ac)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), affine_grid_physical_2d_pullback
end

# 3D rrule!!
function Mooncake.rrule!!(
    func::CoDual{AffineGridPhysical3D{T}},
    theta::CoDual{A, F},
    size::CoDual{S, NoFData};
    align_corners::Bool=true
) where {T, A<:AbstractArray{T,3}, F, S<:NTuple{3,Int}}
    func_primal = func.x
    theta_primal = theta.x
    theta_fdata = theta.dx
    size_primal = size.x

    ac = Val(align_corners)
    spacing = func_primal.spacing

    output = func_primal(theta_primal, size_primal; align_corners)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function affine_grid_physical_3d_pullback(_rdata)
        _∇affine_grid_physical_theta_3d!(theta_fdata, output_fdata, spacing, ac)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), affine_grid_physical_3d_pullback
end

# ============================================================================
# Resampling to Different Spacing
# ============================================================================

"""
    resample(img::PhysicalImage, target_spacing; interpolation=:bilinear)

Resample a PhysicalImage to a new voxel spacing.

# Arguments
- `img`: Input PhysicalImage
- `target_spacing`: Desired output spacing in mm (tuple)
- `interpolation`: `:bilinear` (default) or `:nearest`

# Returns
- New PhysicalImage with resampled data at target spacing

# Example
```julia
# Resample 0.5mm image to 2mm isotropic for registration
img_lowres = resample(img_highres, (2.0, 2.0, 2.0))
```
"""
function resample(
    img::PhysicalImage{T, 5},
    target_spacing::NTuple{3,<:Real};
    interpolation::Symbol=:bilinear
) where T
    target_sp = (T(target_spacing[1]), T(target_spacing[2]), T(target_spacing[3]))
    return _resample_3d(img, target_sp, interpolation)
end

function resample(
    img::PhysicalImage{T, 4},
    target_spacing::NTuple{2,<:Real};
    interpolation::Symbol=:bilinear
) where T
    target_sp = (T(target_spacing[1]), T(target_spacing[2]))
    return _resample_2d(img, target_sp, interpolation)
end

function _resample_3d(
    img::PhysicalImage{T, 5},
    target_spacing::NTuple{3,T},
    interpolation::Symbol
) where T
    X, Y, Z, C, N = size(img.data)
    sx, sy, sz = img.spacing
    target_sx, target_sy, target_sz = target_spacing

    # Compute output size based on physical extent
    extent_x = (X - 1) * sx
    extent_y = (Y - 1) * sy
    extent_z = (Z - 1) * sz

    X_out = max(1, round(Int, extent_x / target_sx) + 1)
    Y_out = max(1, round(Int, extent_y / target_sy) + 1)
    Z_out = max(1, round(Int, extent_z / target_sz) + 1)

    # Create identity affine (no transformation, just resampling)
    theta = similar(img.data, 3, 4, N)
    _fill_identity_affine_3d!(theta)

    # Generate grid that maps from target spacing to source spacing
    # This grid will have normalized coordinates that sample the input correctly
    grid = _create_resample_grid_3d(
        img.data, img.spacing, img.origin,
        (X_out, Y_out, Z_out), target_spacing, img.origin
    )

    # Sample using grid_sample
    output_data = grid_sample(img.data, grid; interpolation=interpolation, padding_mode=:border)

    return PhysicalImage(output_data; spacing=target_spacing, origin=img.origin)
end

function _resample_2d(
    img::PhysicalImage{T, 4},
    target_spacing::NTuple{2,T},
    interpolation::Symbol
) where T
    X, Y, C, N = size(img.data)
    sx, sy = spatial_spacing(img)
    target_sx, target_sy = target_spacing

    # Compute output size based on physical extent
    extent_x = (X - 1) * sx
    extent_y = (Y - 1) * sy

    X_out = max(1, round(Int, extent_x / target_sx) + 1)
    Y_out = max(1, round(Int, extent_y / target_sy) + 1)

    # Create resampling grid
    grid = _create_resample_grid_2d(
        img.data, spatial_spacing(img), (img.origin[1], img.origin[2]),
        (X_out, Y_out), target_spacing, (img.origin[1], img.origin[2])
    )

    # Sample
    output_data = grid_sample(img.data, grid; interpolation=interpolation, padding_mode=:border)

    return PhysicalImage(output_data; spacing=target_spacing, origin=(img.origin[1], img.origin[2]))
end

# Helper to create resampling grid (maps target voxels to source normalized coords)
function _create_resample_grid_3d(
    input::AbstractArray{T,5},
    source_spacing::NTuple{3,T},
    source_origin::NTuple{3,T},
    target_size::NTuple{3,Int},
    target_spacing::NTuple{3,T},
    target_origin::NTuple{3,T}
) where T
    X_in, Y_in, Z_in, C, N = size(input)
    X_out, Y_out, Z_out = target_size

    # Pre-compute normalization factors (use max(X-1, 1) to avoid division by zero)
    norm_x = T(2) / T(max(X_in - 1, 1))
    norm_y = T(2) / T(max(Y_in - 1, 1))
    norm_z = T(2) / T(max(Z_in - 1, 1))

    # Extract tuple elements for GPU kernel
    src_sp_x, src_sp_y, src_sp_z = source_spacing
    src_or_x, src_or_y, src_or_z = source_origin
    tgt_sp_x, tgt_sp_y, tgt_sp_z = target_spacing
    tgt_or_x, tgt_or_y, tgt_or_z = target_origin

    grid = similar(input, 3, X_out, Y_out, Z_out, N)

    AK.foreachindex(grid) do idx
        coord, i, j, k, n = _linear_to_cartesian_5d_phys(idx, X_out, Y_out, Z_out)

        # Physical position in target space
        phys_x = tgt_or_x + (T(i) - one(T)) * tgt_sp_x
        phys_y = tgt_or_y + (T(j) - one(T)) * tgt_sp_y
        phys_z = tgt_or_z + (T(k) - one(T)) * tgt_sp_z

        # Convert to source voxel coordinates
        src_i = (phys_x - src_or_x) / src_sp_x + one(T)
        src_j = (phys_y - src_or_y) / src_sp_y + one(T)
        src_k = (phys_z - src_or_z) / src_sp_z + one(T)

        # Convert to normalized [-1, 1]
        x_norm = (src_i - one(T)) * norm_x - one(T)
        y_norm = (src_j - one(T)) * norm_y - one(T)
        z_norm = (src_k - one(T)) * norm_z - one(T)

        if coord == 1
            @inbounds grid[idx] = x_norm
        elseif coord == 2
            @inbounds grid[idx] = y_norm
        else
            @inbounds grid[idx] = z_norm
        end
    end

    return grid
end

function _create_resample_grid_2d(
    input::AbstractArray{T,4},
    source_spacing::NTuple{2,T},
    source_origin::NTuple{2,T},
    target_size::NTuple{2,Int},
    target_spacing::NTuple{2,T},
    target_origin::NTuple{2,T}
) where T
    X_in, Y_in, C, N = size(input)
    X_out, Y_out = target_size

    # Pre-compute normalization factors
    norm_x = T(2) / T(max(X_in - 1, 1))
    norm_y = T(2) / T(max(Y_in - 1, 1))

    # Extract tuple elements for GPU kernel
    src_sp_x, src_sp_y = source_spacing
    src_or_x, src_or_y = source_origin
    tgt_sp_x, tgt_sp_y = target_spacing
    tgt_or_x, tgt_or_y = target_origin

    grid = similar(input, 2, X_out, Y_out, N)

    AK.foreachindex(grid) do idx
        coord, i, j, n = _linear_to_cartesian_4d_phys(idx, X_out, Y_out)

        # Physical position in target space
        phys_x = tgt_or_x + (T(i) - one(T)) * tgt_sp_x
        phys_y = tgt_or_y + (T(j) - one(T)) * tgt_sp_y

        # Convert to source voxel coordinates
        src_i = (phys_x - src_or_x) / src_sp_x + one(T)
        src_j = (phys_y - src_or_y) / src_sp_y + one(T)

        # Convert to normalized [-1, 1]
        x_norm = (src_i - one(T)) * norm_x - one(T)
        y_norm = (src_j - one(T)) * norm_y - one(T)

        if coord == 1
            @inbounds grid[idx] = x_norm
        else
            @inbounds grid[idx] = y_norm
        end
    end

    return grid
end

# Helper to fill identity affine matrix
function _fill_identity_affine_3d!(theta::AbstractArray{T,3}) where T
    fill!(theta, zero(T))
    N = size(theta, 3)
    # Set diagonal to 1 on CPU then copy to avoid scalar indexing
    theta_cpu = zeros(T, 3, 4, N)
    for n in 1:N
        theta_cpu[1, 1, n] = one(T)
        theta_cpu[2, 2, n] = one(T)
        theta_cpu[3, 3, n] = one(T)
    end
    copyto!(theta, theta_cpu)
    return nothing
end

# ============================================================================
# Resample rrule!! (for gradients through resampling if needed)
# ============================================================================

# Resampling is just grid_sample under the hood, so gradients flow through grid_sample
# No additional rrule!! needed - Mooncake will handle composition

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    with_spacing(img::PhysicalImage, spacing)

Create a new PhysicalImage with different spacing metadata (same data).
"""
function with_spacing(img::PhysicalImage{T, N, A}, spacing::NTuple) where {T, N, A}
    sp = if N == 4
        (T(spacing[1]), T(spacing[2]), one(T))
    else
        (T(spacing[1]), T(spacing[2]), T(spacing[3]))
    end
    return PhysicalImage{T, N, A}(img.data, sp, img.origin)
end

"""
    with_origin(img::PhysicalImage, origin)

Create a new PhysicalImage with different origin metadata (same data).
"""
function with_origin(img::PhysicalImage{T, N, A}, origin::NTuple) where {T, N, A}
    org = if N == 4
        (T(origin[1]), T(origin[2]), zero(T))
    else
        (T(origin[1]), T(origin[2]), T(origin[3]))
    end
    return PhysicalImage{T, N, A}(img.data, img.spacing, org)
end

"""
    similar_physical(img::PhysicalImage, data::AbstractArray)

Create a new PhysicalImage with same spacing/origin but different data.
"""
function similar_physical(img::PhysicalImage{T, N}, data::AbstractArray{T, N}) where {T, N}
    return PhysicalImage{T, N, typeof(data)}(data, img.spacing, img.origin)
end
