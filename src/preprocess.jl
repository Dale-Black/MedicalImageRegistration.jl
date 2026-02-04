# Preprocessing pipeline for clinical CT registration
# GPU-first architecture with AK.foreachindex
#
# Functions for initial alignment before optimization:
# - center_of_mass: intensity-weighted COM in physical coordinates
# - align_centers: translate moving image to align COMs
# - compute_overlap_region: find overlapping FOV
# - crop_to_overlap: crop to overlapping region
# - window_intensity: HU windowing for CT
# - preprocess_for_registration: main pipeline

# ============================================================================
# Center of Mass Computation
# ============================================================================

"""
    center_of_mass(image::PhysicalImage; threshold=-200f0)

Compute intensity-weighted center of mass in physical coordinates (mm).

# Arguments
- `image`: PhysicalImage with data, spacing, origin
- `threshold`: Minimum HU to include (excludes air/lung below this value)

# Returns
- `(x_mm, y_mm, z_mm)`: Center of mass in physical coordinates

# Algorithm
For CT images, uses threshold to exclude air (typically -1000 HU) and lung tissue.
Recommended threshold of -200 HU excludes air/lung but includes soft tissue.

```math
COM_x = \\frac{\\sum_{i,j,k} (I_{i,j,k} - threshold) \\cdot x_{i,j,k}}{\\sum_{i,j,k} (I_{i,j,k} - threshold)}
```

# Example
```julia
# Compute COM for cardiac CT, excluding air/lung
com = center_of_mass(ccta_image; threshold=-200f0)
# Returns something like (0.0, 50.0, 70.0) mm near heart center
```
"""
function center_of_mass(
    image::PhysicalImage{T, 5};
    threshold::Real = T(-200)
) where T
    return _center_of_mass_3d(image, T(threshold))
end

function center_of_mass(
    image::PhysicalImage{T, 4};
    threshold::Real = T(-200)
) where T
    return _center_of_mass_2d(image, T(threshold))
end

# 3D implementation using two-pass approach (avoid scalar indexing)
function _center_of_mass_3d(image::PhysicalImage{T, 5}, threshold::T) where T
    data = image.data
    X, Y, Z, C, N = size(data)
    sx, sy, sz = image.spacing
    ox, oy, oz = image.origin

    # Allocate accumulators (4 values: sum_weight, sum_x, sum_y, sum_z)
    # One set per batch element
    accum = similar(data, 4, N)
    fill!(accum, zero(T))

    # First pass: accumulate weighted positions using atomic operations
    AK.foreachindex(data) do idx
        # Convert linear index to (i, j, k, c, n)
        idx_0 = idx - 1
        i = idx_0 % X + 1
        idx_0 = idx_0 ÷ X
        j = idx_0 % Y + 1
        idx_0 = idx_0 ÷ Y
        k = idx_0 % Z + 1
        idx_0 = idx_0 ÷ Z
        c = idx_0 % C + 1
        n = idx_0 ÷ C + 1

        # Only use first channel for COM
        if c != 1
            return nothing
        end

        @inbounds val = data[i, j, k, c, n]

        # Apply threshold: weight = max(0, val - threshold)
        weight = max(val - threshold, zero(T))

        if weight > zero(T)
            # Physical coordinates of this voxel
            phys_x = ox + (T(i) - one(T)) * sx
            phys_y = oy + (T(j) - one(T)) * sy
            phys_z = oz + (T(k) - one(T)) * sz

            # Accumulate
            Atomix.@atomic accum[1, n] += weight
            Atomix.@atomic accum[2, n] += weight * phys_x
            Atomix.@atomic accum[3, n] += weight * phys_y
            Atomix.@atomic accum[4, n] += weight * phys_z
        end

        return nothing
    end

    # Copy to CPU for final division (small array, OK to index)
    accum_cpu = Array(accum)

    # Compute COM for first batch element (typical use case)
    total_weight = accum_cpu[1, 1]
    if total_weight > zero(T)
        com_x = accum_cpu[2, 1] / total_weight
        com_y = accum_cpu[3, 1] / total_weight
        com_z = accum_cpu[4, 1] / total_weight
    else
        # Fallback to geometric center if no voxels above threshold
        com_x = ox + (T(X) - one(T)) * sx / T(2)
        com_y = oy + (T(Y) - one(T)) * sy / T(2)
        com_z = oz + (T(Z) - one(T)) * sz / T(2)
    end

    return (T(com_x), T(com_y), T(com_z))
end

# 2D implementation
function _center_of_mass_2d(image::PhysicalImage{T, 4}, threshold::T) where T
    data = image.data
    X, Y, C, N = size(data)
    sx, sy = spatial_spacing(image)
    ox, oy = image.origin[1], image.origin[2]

    # Allocate accumulators (3 values: sum_weight, sum_x, sum_y)
    accum = similar(data, 3, N)
    fill!(accum, zero(T))

    AK.foreachindex(data) do idx
        idx_0 = idx - 1
        i = idx_0 % X + 1
        idx_0 = idx_0 ÷ X
        j = idx_0 % Y + 1
        idx_0 = idx_0 ÷ Y
        c = idx_0 % C + 1
        n = idx_0 ÷ C + 1

        if c != 1
            return nothing
        end

        @inbounds val = data[i, j, c, n]
        weight = max(val - threshold, zero(T))

        if weight > zero(T)
            phys_x = ox + (T(i) - one(T)) * sx
            phys_y = oy + (T(j) - one(T)) * sy

            Atomix.@atomic accum[1, n] += weight
            Atomix.@atomic accum[2, n] += weight * phys_x
            Atomix.@atomic accum[3, n] += weight * phys_y
        end

        return nothing
    end

    accum_cpu = Array(accum)

    total_weight = accum_cpu[1, 1]
    if total_weight > zero(T)
        com_x = accum_cpu[2, 1] / total_weight
        com_y = accum_cpu[3, 1] / total_weight
    else
        com_x = ox + (T(X) - one(T)) * sx / T(2)
        com_y = oy + (T(Y) - one(T)) * sy / T(2)
    end

    return (T(com_x), T(com_y))
end

# ============================================================================
# Center Alignment
# ============================================================================

"""
    align_centers(moving::PhysicalImage, static::PhysicalImage; threshold=-200f0)

Compute translation to align centers of mass.

# Arguments
- `moving`: Image to be aligned
- `static`: Reference image
- `threshold`: HU threshold for COM computation

# Returns
- `translated_moving`: PhysicalImage with updated origin (data unchanged)
- `translation`: The (dx, dy, dz) translation in mm

# Notes
This does NOT resample the data - it only adjusts the origin metadata.
The translation represents how much the moving image needs to shift to
align its COM with the static image's COM.

# Example
```julia
nc_aligned, translation = align_centers(nc_physical, ccta_physical)
# translation might be (5.0, 5.0, -15.0) mm
```
"""
function align_centers(
    moving::PhysicalImage{T, 5},
    static::PhysicalImage{T, 5};
    threshold::Real = T(-200)
) where T
    thresh = T(threshold)

    # Compute centers of mass
    com_static = center_of_mass(static; threshold=thresh)
    com_moving = center_of_mass(moving; threshold=thresh)

    # Translation to align moving COM to static COM
    translation = (
        com_static[1] - com_moving[1],
        com_static[2] - com_moving[2],
        com_static[3] - com_moving[3]
    )

    # Create new PhysicalImage with adjusted origin
    # Adding translation to origin moves the image in physical space
    new_origin = (
        moving.origin[1] + translation[1],
        moving.origin[2] + translation[2],
        moving.origin[3] + translation[3]
    )

    translated = PhysicalImage(moving.data; spacing=moving.spacing, origin=new_origin)

    return translated, translation
end

function align_centers(
    moving::PhysicalImage{T, 4},
    static::PhysicalImage{T, 4};
    threshold::Real = T(-200)
) where T
    thresh = T(threshold)

    com_static = center_of_mass(static; threshold=thresh)
    com_moving = center_of_mass(moving; threshold=thresh)

    translation = (
        com_static[1] - com_moving[1],
        com_static[2] - com_moving[2]
    )

    new_origin = (
        moving.origin[1] + translation[1],
        moving.origin[2] + translation[2]
    )

    translated = PhysicalImage(moving.data; spacing=(moving.spacing[1], moving.spacing[2]), origin=new_origin)

    return translated, translation
end

# ============================================================================
# Overlap Region Detection
# ============================================================================

"""
    compute_overlap_region(img1::PhysicalImage, img2::PhysicalImage)

Find physical bounding box of overlapping region between two images.

# Arguments
- `img1`: First PhysicalImage
- `img2`: Second PhysicalImage

# Returns
- `(min_corner, max_corner)`: Physical coordinates (mm) of overlap region
- `nothing`: If images don't overlap at all

# Example
```julia
overlap = compute_overlap_region(ccta_physical, nc_aligned)
if isnothing(overlap)
    error("Images do not overlap!")
end
# overlap = ((min_x, min_y, min_z), (max_x, max_y, max_z))
```
"""
function compute_overlap_region(
    img1::PhysicalImage{T, 5},
    img2::PhysicalImage{T, 5}
) where T
    # Get bounding boxes
    bb1 = _get_bounding_box_3d(img1)
    bb2 = _get_bounding_box_3d(img2)

    # Compute intersection
    min_corner = (
        max(bb1.min[1], bb2.min[1]),
        max(bb1.min[2], bb2.min[2]),
        max(bb1.min[3], bb2.min[3])
    )
    max_corner = (
        min(bb1.max[1], bb2.max[1]),
        min(bb1.max[2], bb2.max[2]),
        min(bb1.max[3], bb2.max[3])
    )

    # Check if valid overlap exists
    if min_corner[1] >= max_corner[1] ||
       min_corner[2] >= max_corner[2] ||
       min_corner[3] >= max_corner[3]
        return nothing
    end

    return (min=min_corner, max=max_corner)
end

function compute_overlap_region(
    img1::PhysicalImage{T, 4},
    img2::PhysicalImage{T, 4}
) where T
    bb1 = _get_bounding_box_2d(img1)
    bb2 = _get_bounding_box_2d(img2)

    min_corner = (
        max(bb1.min[1], bb2.min[1]),
        max(bb1.min[2], bb2.min[2])
    )
    max_corner = (
        min(bb1.max[1], bb2.max[1]),
        min(bb1.max[2], bb2.max[2])
    )

    if min_corner[1] >= max_corner[1] || min_corner[2] >= max_corner[2]
        return nothing
    end

    return (min=min_corner, max=max_corner)
end

# Helper to get bounding box in physical coordinates
function _get_bounding_box_3d(img::PhysicalImage{T, 5}) where T
    X, Y, Z, C, N = size(img.data)
    ox, oy, oz = img.origin
    sx, sy, sz = img.spacing

    min_corner = (ox, oy, oz)
    max_corner = (
        ox + (X - 1) * sx,
        oy + (Y - 1) * sy,
        oz + (Z - 1) * sz
    )

    return (min=min_corner, max=max_corner)
end

function _get_bounding_box_2d(img::PhysicalImage{T, 4}) where T
    X, Y, C, N = size(img.data)
    ox, oy = img.origin[1], img.origin[2]
    sx, sy = spatial_spacing(img)

    min_corner = (ox, oy)
    max_corner = (
        ox + (X - 1) * sx,
        oy + (Y - 1) * sy
    )

    return (min=min_corner, max=max_corner)
end

# ============================================================================
# Crop to Region
# ============================================================================

"""
    crop_to_overlap(image::PhysicalImage, region)

Crop image to specified physical region.

# Arguments
- `image`: PhysicalImage to crop
- `region`: From compute_overlap_region(), or (min=(...), max=(...)) tuple

# Returns
- PhysicalImage with cropped data and updated origin

# Notes
Uses nearest-voxel boundaries (no interpolation for crop).
The cropped image's origin is updated to match the crop region.

# Example
```julia
overlap = compute_overlap_region(img1, img2)
cropped = crop_to_overlap(img1, overlap)
```
"""
function crop_to_overlap(
    image::PhysicalImage{T, 5},
    region
) where T
    X, Y, Z, C, N = size(image.data)
    ox, oy, oz = image.origin
    sx, sy, sz = image.spacing

    # Convert physical region to voxel indices (1-based)
    i_start = round(Int, (region.min[1] - ox) / sx) + 1
    i_end = round(Int, (region.max[1] - ox) / sx) + 1
    j_start = round(Int, (region.min[2] - oy) / sy) + 1
    j_end = round(Int, (region.max[2] - oy) / sy) + 1
    k_start = round(Int, (region.min[3] - oz) / sz) + 1
    k_end = round(Int, (region.max[3] - oz) / sz) + 1

    # Clamp to valid range
    i_start = clamp(i_start, 1, X)
    i_end = clamp(i_end, 1, X)
    j_start = clamp(j_start, 1, Y)
    j_end = clamp(j_end, 1, Y)
    k_start = clamp(k_start, 1, Z)
    k_end = clamp(k_end, 1, Z)

    # Extract cropped region
    # Note: We need to use a GPU-compatible approach
    cropped_data = _crop_3d(image.data, i_start, i_end, j_start, j_end, k_start, k_end)

    # Compute new origin (physical position of first voxel in cropped image)
    new_origin = (
        ox + (i_start - 1) * sx,
        oy + (j_start - 1) * sy,
        oz + (k_start - 1) * sz
    )

    return PhysicalImage(cropped_data; spacing=image.spacing, origin=new_origin)
end

function crop_to_overlap(
    image::PhysicalImage{T, 4},
    region
) where T
    X, Y, C, N = size(image.data)
    ox, oy = image.origin[1], image.origin[2]
    sx, sy = spatial_spacing(image)

    i_start = round(Int, (region.min[1] - ox) / sx) + 1
    i_end = round(Int, (region.max[1] - ox) / sx) + 1
    j_start = round(Int, (region.min[2] - oy) / sy) + 1
    j_end = round(Int, (region.max[2] - oy) / sy) + 1

    i_start = clamp(i_start, 1, X)
    i_end = clamp(i_end, 1, X)
    j_start = clamp(j_start, 1, Y)
    j_end = clamp(j_end, 1, Y)

    cropped_data = _crop_2d(image.data, i_start, i_end, j_start, j_end)

    new_origin = (
        ox + (i_start - 1) * sx,
        oy + (j_start - 1) * sy
    )

    return PhysicalImage(cropped_data; spacing=(image.spacing[1], image.spacing[2]), origin=new_origin)
end

# GPU-compatible cropping using AK.foreachindex
function _crop_3d(
    data::AbstractArray{T, 5},
    i_start::Int, i_end::Int,
    j_start::Int, j_end::Int,
    k_start::Int, k_end::Int
) where T
    X_in, Y_in, Z_in, C, N = size(data)
    X_out = i_end - i_start + 1
    Y_out = j_end - j_start + 1
    Z_out = k_end - k_start + 1

    output = similar(data, X_out, Y_out, Z_out, C, N)

    AK.foreachindex(output) do idx
        idx_0 = idx - 1
        i_out = idx_0 % X_out + 1
        idx_0 = idx_0 ÷ X_out
        j_out = idx_0 % Y_out + 1
        idx_0 = idx_0 ÷ Y_out
        k_out = idx_0 % Z_out + 1
        idx_0 = idx_0 ÷ Z_out
        c = idx_0 % C + 1
        n = idx_0 ÷ C + 1

        # Map output index to input index
        i_in = i_start + i_out - 1
        j_in = j_start + j_out - 1
        k_in = k_start + k_out - 1

        @inbounds output[i_out, j_out, k_out, c, n] = data[i_in, j_in, k_in, c, n]

        return nothing
    end

    return output
end

function _crop_2d(
    data::AbstractArray{T, 4},
    i_start::Int, i_end::Int,
    j_start::Int, j_end::Int
) where T
    X_in, Y_in, C, N = size(data)
    X_out = i_end - i_start + 1
    Y_out = j_end - j_start + 1

    output = similar(data, X_out, Y_out, C, N)

    AK.foreachindex(output) do idx
        idx_0 = idx - 1
        i_out = idx_0 % X_out + 1
        idx_0 = idx_0 ÷ X_out
        j_out = idx_0 % Y_out + 1
        idx_0 = idx_0 ÷ Y_out
        c = idx_0 % C + 1
        n = idx_0 ÷ C + 1

        i_in = i_start + i_out - 1
        j_in = j_start + j_out - 1

        @inbounds output[i_out, j_out, c, n] = data[i_in, j_in, c, n]

        return nothing
    end

    return output
end

# ============================================================================
# Intensity Windowing
# ============================================================================

"""
    window_intensity(image; min_hu=-200f0, max_hu=1000f0)
    window_intensity(image::PhysicalImage; min_hu=-200f0, max_hu=1000f0)

Clip intensity values to specified range.

# Arguments
- `image`: Array or PhysicalImage to window
- `min_hu`: Minimum intensity value (default -200 for soft tissue)
- `max_hu`: Maximum intensity value (default 1000 for bone)

# Returns
- New array/PhysicalImage with windowed values

# Notes
For CT registration, windowing reduces the influence of extreme values
(very dense bone, metal artifacts) and focuses on soft tissue alignment.

# Example
```julia
# Window to soft tissue range
windowed = window_intensity(ct_image; min_hu=-100f0, max_hu=300f0)
```
"""
function window_intensity(
    data::AbstractArray{T};
    min_hu::Real = T(-200),
    max_hu::Real = T(1000)
) where T
    return _window_intensity_impl(data, T(min_hu), T(max_hu))
end

function window_intensity(
    image::PhysicalImage{T, N};
    min_hu::Real = T(-200),
    max_hu::Real = T(1000)
) where {T, N}
    windowed_data = _window_intensity_impl(image.data, T(min_hu), T(max_hu))
    return PhysicalImage{T, N, typeof(windowed_data)}(windowed_data, image.spacing, image.origin)
end

function _window_intensity_impl(data::AbstractArray{T}, min_val::T, max_val::T) where T
    output = similar(data)

    AK.foreachindex(output) do idx
        @inbounds val = data[idx]
        @inbounds output[idx] = clamp(val, min_val, max_val)
        return nothing
    end

    return output
end

# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

"""
    PreprocessInfo{T}

Contains information about preprocessing steps applied.

# Fields
- `translation`: COM alignment translation (dx, dy, dz) in mm
- `overlap_region`: Physical bounds of overlap region
- `com_moving`: Center of mass of moving image
- `com_static`: Center of mass of static image
"""
struct PreprocessInfo{T}
    translation::NTuple{3, T}
    overlap_region::Union{Nothing, NamedTuple{(:min, :max), Tuple{NTuple{3, T}, NTuple{3, T}}}}
    com_moving::NTuple{3, T}
    com_static::NTuple{3, T}
end

"""
    preprocess_for_registration(moving::PhysicalImage, static::PhysicalImage; kwargs...)

Full preprocessing pipeline for clinical CT registration.

# Arguments
- `moving`: Moving image (to be registered to static)
- `static`: Static/reference image

# Keyword Arguments
- `registration_resolution::Real = 2.0f0`: Target spacing in mm for registration
- `align_com::Bool = true`: Align centers of mass before registration
- `crop_to_overlap::Bool = true`: Crop both images to overlapping region
- `window_hu::Bool = true`: Apply HU windowing
- `min_hu::Real = -200f0`: Minimum HU for windowing
- `max_hu::Real = 1000f0`: Maximum HU for windowing
- `com_threshold::Real = -200f0`: Threshold for COM computation

# Returns
- `preprocessed_moving`: PhysicalImage ready for registration
- `preprocessed_static`: PhysicalImage ready for registration
- `preprocess_info::PreprocessInfo`: Information about applied preprocessing

# Pipeline Steps
1. Compute centers of mass for both images
2. Align moving image COM to static image COM (if align_com=true)
3. Detect overlapping FOV region
4. Crop both images to overlap (if crop_to_overlap=true)
5. Resample both to registration_resolution
6. Apply HU windowing (if window_hu=true)

# Example
```julia
# Full preprocessing for cardiac CT
moving_prep, static_prep, info = preprocess_for_registration(
    ccta_physical, nc_physical;
    registration_resolution=2.0f0,
    align_com=true,
    crop_to_overlap=true,
    window_hu=true
)

# Check preprocessing info
println("Translation: \$(info.translation) mm")
println("Overlap extent: \$(info.overlap_region.max .- info.overlap_region.min) mm")
```
"""
function preprocess_for_registration(
    moving::PhysicalImage{T, 5},
    static::PhysicalImage{T, 5};
    registration_resolution::Real = T(2.0),
    align_com::Bool = true,
    do_crop_to_overlap::Bool = true,
    window_hu::Bool = true,
    min_hu::Real = T(-200),
    max_hu::Real = T(1000),
    com_threshold::Real = T(-200)
) where T
    target_res = T(registration_resolution)
    thresh = T(com_threshold)

    # Step 1: Compute initial centers of mass
    com_moving_orig = center_of_mass(moving; threshold=thresh)
    com_static_orig = center_of_mass(static; threshold=thresh)

    # Step 2: Align centers of mass (if requested)
    translation = (zero(T), zero(T), zero(T))
    moving_aligned = moving

    if align_com
        moving_aligned, translation = align_centers(moving, static; threshold=thresh)
    end

    # Step 3: Compute overlap region
    overlap = compute_overlap_region(moving_aligned, static)

    if isnothing(overlap)
        error("Images do not overlap in physical space after COM alignment! " *
              "Check that both images cover the same anatomical region.")
    end

    # Step 4: Crop to overlap (if requested)
    moving_cropped = moving_aligned
    static_cropped = static

    if do_crop_to_overlap
        moving_cropped = crop_to_overlap(moving_aligned, overlap)
        static_cropped = crop_to_overlap(static, overlap)
    end

    # Step 5: Resample to registration resolution
    target_spacing = (target_res, target_res, target_res)
    moving_resampled = resample(moving_cropped, target_spacing; interpolation=:bilinear)
    static_resampled = resample(static_cropped, target_spacing; interpolation=:bilinear)

    # Step 6: Window intensities (if requested)
    moving_final = moving_resampled
    static_final = static_resampled

    if window_hu
        moving_final = window_intensity(moving_resampled; min_hu=T(min_hu), max_hu=T(max_hu))
        static_final = window_intensity(static_resampled; min_hu=T(min_hu), max_hu=T(max_hu))
    end

    # Build preprocessing info
    # Ensure 3D translation for info struct
    translation_3d = (T(translation[1]), T(translation[2]), T(translation[3]))
    com_moving_3d = (T(com_moving_orig[1]), T(com_moving_orig[2]), T(com_moving_orig[3]))
    com_static_3d = (T(com_static_orig[1]), T(com_static_orig[2]), T(com_static_orig[3]))

    overlap_typed = if !isnothing(overlap)
        (min=(T(overlap.min[1]), T(overlap.min[2]), T(overlap.min[3])),
         max=(T(overlap.max[1]), T(overlap.max[2]), T(overlap.max[3])))
    else
        nothing
    end

    info = PreprocessInfo{T}(translation_3d, overlap_typed, com_moving_3d, com_static_3d)

    return moving_final, static_final, info
end

# 2D version
function preprocess_for_registration(
    moving::PhysicalImage{T, 4},
    static::PhysicalImage{T, 4};
    registration_resolution::Real = T(2.0),
    align_com::Bool = true,
    do_crop_to_overlap::Bool = true,
    window_hu::Bool = true,
    min_hu::Real = T(-200),
    max_hu::Real = T(1000),
    com_threshold::Real = T(-200)
) where T
    target_res = T(registration_resolution)
    thresh = T(com_threshold)

    # Step 1: Compute initial centers of mass
    com_moving_orig = center_of_mass(moving; threshold=thresh)
    com_static_orig = center_of_mass(static; threshold=thresh)

    # Step 2: Align centers of mass (if requested)
    translation = (zero(T), zero(T))
    moving_aligned = moving

    if align_com
        moving_aligned, translation = align_centers(moving, static; threshold=thresh)
    end

    # Step 3: Compute overlap region
    overlap = compute_overlap_region(moving_aligned, static)

    if isnothing(overlap)
        error("Images do not overlap in physical space after COM alignment!")
    end

    # Step 4: Crop to overlap (if requested)
    moving_cropped = moving_aligned
    static_cropped = static

    if do_crop_to_overlap
        moving_cropped = crop_to_overlap(moving_aligned, overlap)
        static_cropped = crop_to_overlap(static, overlap)
    end

    # Step 5: Resample to registration resolution
    target_spacing = (target_res, target_res)
    moving_resampled = resample(moving_cropped, target_spacing; interpolation=:bilinear)
    static_resampled = resample(static_cropped, target_spacing; interpolation=:bilinear)

    # Step 6: Window intensities (if requested)
    moving_final = moving_resampled
    static_final = static_resampled

    if window_hu
        moving_final = window_intensity(moving_resampled; min_hu=T(min_hu), max_hu=T(max_hu))
        static_final = window_intensity(static_resampled; min_hu=T(min_hu), max_hu=T(max_hu))
    end

    # Build preprocessing info (pad to 3D for consistency)
    translation_3d = (T(translation[1]), T(translation[2]), zero(T))
    com_moving_3d = (T(com_moving_orig[1]), T(com_moving_orig[2]), zero(T))
    com_static_3d = (T(com_static_orig[1]), T(com_static_orig[2]), zero(T))

    overlap_typed = if !isnothing(overlap)
        (min=(T(overlap.min[1]), T(overlap.min[2]), zero(T)),
         max=(T(overlap.max[1]), T(overlap.max[2]), zero(T)))
    else
        nothing
    end

    info = PreprocessInfo{T}(translation_3d, overlap_typed, com_moving_3d, com_static_3d)

    return moving_final, static_final, info
end
