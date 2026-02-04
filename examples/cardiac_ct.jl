### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ a987dd55-059d-4a21-9bcc-9440b1899ed1
begin
	import Pkg
	Pkg.activate(@__DIR__)
end

# ╔═╡ c41f6864-09ef-4e45-a62a-4d63dc6cd9b2
import MedicalImageRegistration as MIR

# ╔═╡ 4df0ee4c-cdc1-4a92-8878-3508fe2378ff
import DICOM

# ╔═╡ c569ab7c-67c4-46bd-8d8b-4f56a77990e5
import CairoMakie as CM

# ╔═╡ 605f29e5-aeac-40a4-a0f9-cfa9882ce096
import PlutoUI as UI

# ╔═╡ 8de86531-fada-421f-8ee1-dbfd4952b0da
UI.TableOfContents()

# ╔═╡ ecf6bdc9-109a-4a8c-b5b9-34424f2880a7
non_contrast_dir = joinpath(@__DIR__, "DICOMs/ca score VMI 70")

# ╔═╡ 351c65de-b533-4ebb-a055-744bbe94dd4b
ccta_dir = joinpath(@__DIR__, "DICOMs/ccta mono 70")

# ╔═╡ 7cb187a4-c0c7-465f-a58f-ba5cb51d84fb
nc_dcms = DICOM.dcmdir_parse(non_contrast_dir)

# ╔═╡ 48f6d4e5-08c8-4e13-afab-c37d71cd774b
nc_dcms[1].meta

# ╔═╡ 33f0b630-e3a3-4b03-a1b7-fec51489b37d
ccta_dcms = DICOM.dcmdir_parse(ccta_dir)

# ╔═╡ 3407d97d-485c-4cdf-bd46-9f9e051728bd
md"""
## Helper Functions for DICOM Loading

These functions handle:
1. Extracting metadata (spacing, position, rescale parameters)
2. Sorting slices by z-position
3. Converting to HU values using RescaleSlope and RescaleIntercept
4. Creating 3D volumes in the format expected by MedicalImageRegistration: (X, Y, Z, C, N)
"""

# ╔═╡ ceea726d-db0f-4050-a00f-3e679dfe5c68
begin
	# DICOM tag hex codes (more reliable than tag string macro)
	const TAG_RESCALE_INTERCEPT = (0x0028, 0x1052)
	const TAG_RESCALE_SLOPE = (0x0028, 0x1053)
	const TAG_PIXEL_SPACING = (0x0028, 0x0030)
	const TAG_SLICE_THICKNESS = (0x0018, 0x0050)
	const TAG_IMAGE_POSITION = (0x0020, 0x0032)
	const TAG_IMAGE_ORIENTATION = (0x0020, 0x0037)
	const TAG_ROWS = (0x0028, 0x0010)
	const TAG_COLUMNS = (0x0028, 0x0011)
	const TAG_PIXEL_DATA = (0x7FE0, 0x0010)
end

# ╔═╡ e670750f-b885-401d-ba4c-30f18a0442af
"""
Extract metadata from a single DICOM slice.
Returns rescale parameters, spatial information, and image dimensions.
"""
function get_dicom_metadata(dcm)
	# Rescale parameters (required for HU conversion)
	rescale_slope = get(dcm, TAG_RESCALE_SLOPE, 1.0)
	rescale_intercept = get(dcm, TAG_RESCALE_INTERCEPT, 0.0)

	# Spatial information
	pixel_spacing = get(dcm, TAG_PIXEL_SPACING, [1.0, 1.0])  # [row, col] in mm
	slice_thickness = get(dcm, TAG_SLICE_THICKNESS, 1.0)  # mm

	# Position and orientation
	image_position = get(dcm, TAG_IMAGE_POSITION, [0.0, 0.0, 0.0])
	image_orientation = get(dcm, TAG_IMAGE_ORIENTATION, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

	# Image dimensions
	rows = dcm[TAG_ROWS]
	cols = dcm[TAG_COLUMNS]

	return (
		rescale_slope = Float32(rescale_slope),
		rescale_intercept = Float32(rescale_intercept),
		pixel_spacing = Float32.(pixel_spacing),
		slice_thickness = Float32(slice_thickness),
		image_position = Float32.(image_position),
		image_orientation = Float32.(image_orientation),
		rows = rows,
		cols = cols
	)
end

# ╔═╡ 046e3d1d-cd6f-4b06-bb80-65ff7ea60fcf
"""
Sort DICOM slices by z-position (Image Position Patient).
"""
function sort_slices_by_position(dcms)
	sorted = sort(dcms, by = dcm -> dcm[TAG_IMAGE_POSITION][3])
	return sorted
end

# ╔═╡ 4c66ad49-3fdf-408b-807b-c0cbe786937f
"""
Load a DICOM series as a 3D volume with proper HU values.

Returns a NamedTuple with:
- volume: (X, Y, Z, 1, 1) array in Float32 with HU values
- spacing: (x, y, z) voxel spacing in mm
- origin: (x, y, z) position of first voxel in mm
- orientation: DICOM orientation cosines
- size_voxels: (nx, ny, nz) number of voxels
- size_mm: (sx, sy, sz) physical size in mm
"""
function load_dicom_volume(dcms; dtype=Float32)
	# Sort slices by z-position
	sorted_dcms = sort_slices_by_position(dcms)

	# Get metadata from first slice
	meta = get_dicom_metadata(sorted_dcms[1])

	# Get number of slices
	n_slices = length(sorted_dcms)

	# Pre-allocate volume: (X, Y, Z, C, N) for MedicalImageRegistration
	# DICOM stores as (rows, cols) which is (Y, X), so we transpose
	volume = zeros(dtype, meta.cols, meta.rows, n_slices, 1, 1)

	# Load each slice and apply rescale
	for (i, dcm) in enumerate(sorted_dcms)
		# Get pixel data (stored as rows × cols)
		pixel_data = dcm[TAG_PIXEL_DATA]

		# Get rescale parameters (may vary per slice, though usually constant)
		slope = get(dcm, TAG_RESCALE_SLOPE, 1.0)
		intercept = get(dcm, TAG_RESCALE_INTERCEPT, 0.0)

		# Apply rescale to get HU values: HU = pixel * slope + intercept
		hu_slice = dtype.(pixel_data) .* dtype(slope) .+ dtype(intercept)

		# Store in volume (transpose to go from row-major to col-major)
		# DICOM is (row, col) = (Y, X), Julia convention is (X, Y)
		volume[:, :, i, 1, 1] = permutedims(hu_slice, (2, 1))
	end

	# Calculate z-spacing from actual slice positions
	if n_slices > 1
		z_pos_first = sorted_dcms[1][TAG_IMAGE_POSITION][3]
		z_pos_last = sorted_dcms[end][TAG_IMAGE_POSITION][3]
		z_spacing = abs(z_pos_last - z_pos_first) / (n_slices - 1)
	else
		z_spacing = meta.slice_thickness
	end

	# Return volume and complete metadata
	return (
		volume = volume,
		spacing = (meta.pixel_spacing[2], meta.pixel_spacing[1], z_spacing),  # (x, y, z) in mm
		origin = Tuple(meta.image_position),
		orientation = meta.image_orientation,
		size_voxels = size(volume)[1:3],
		size_mm = (meta.pixel_spacing[2] * meta.cols, meta.pixel_spacing[1] * meta.rows, z_spacing * n_slices)
	)
end

# ╔═╡ 14f2d0b4-89b0-4b8c-b9cd-3c637d6a6fd4
md"""
## Load Both Volumes
"""

# ╔═╡ bb5d236f-41cc-46d4-b6ce-2ffa10cf33ff
nc_data = load_dicom_volume(nc_dcms)

# ╔═╡ 3c8bbb15-13cd-4531-be85-d163958debad
ccta_data = load_dicom_volume(ccta_dcms)

# ╔═╡ 149d9e89-c765-4ab0-bd64-f300f968b9cf
md"""
## Metadata Comparison

Compare the two scans to understand the registration challenge:
- Resolution mismatch (especially z-spacing)
- FOV differences
- HU range differences (contrast vs non-contrast)
"""

# ╔═╡ 3dc458fd-9957-41b5-b2fe-8b20173806ae
begin
	println("=" ^ 60)
	println("NON-CONTRAST (Ca Score) Volume:")
	println("  Size (voxels): ", nc_data.size_voxels)
	println("  Spacing (mm):  ", round.(nc_data.spacing, digits=3))
	println("  Size (mm):     ", round.(nc_data.size_mm, digits=1))
	println("  HU range:      ", extrema(nc_data.volume))
	println()
	println("CCTA (Contrast) Volume:")
	println("  Size (voxels): ", ccta_data.size_voxels)
	println("  Spacing (mm):  ", round.(ccta_data.spacing, digits=3))
	println("  Size (mm):     ", round.(ccta_data.size_mm, digits=1))
	println("  HU range:      ", extrema(ccta_data.volume))
	println("=" ^ 60)
end

# ╔═╡ cbafe59d-6f8a-4e25-a8f8-f3e86d7cf4d0
md"""
## Resolution Analysis
"""

# ╔═╡ a40e6d1d-f1a8-4601-bf7a-f597f0788ac5
begin
	z_ratio = nc_data.spacing[3] / ccta_data.spacing[3]
	println("\nResolution Analysis:")
	println("  Non-contrast z-spacing: ", round(nc_data.spacing[3], digits=2), " mm")
	println("  CCTA z-spacing: ", round(ccta_data.spacing[3], digits=2), " mm")
	println("  Z-spacing ratio: ", round(z_ratio, digits=1), "x")

	if z_ratio > 2
		println("  ⚠️  Significant resolution mismatch - will need careful handling!")
	end
end

# ╔═╡ 138f598d-0399-45f7-8c61-5336d460caaf
md"""
## HU Sanity Check

Expected HU values:
- Air: ~-1000 HU
- Water/soft tissue: ~0-60 HU
- Bone: ~400-1000+ HU
- Contrast-enhanced blood: ~200-400+ HU
"""

# ╔═╡ 1b8f0793-5728-4b17-93fe-7249c1cd3638
begin
	nc_unique = length(unique(nc_data.volume))
	ccta_unique = length(unique(ccta_data.volume))
	println("Unique HU values:")
	println("  Non-contrast: ", nc_unique)
	println("  CCTA: ", ccta_unique)
end

# ╔═╡ fe28b08c-f926-404e-b273-5fad4f4734d1
md"""
## Visualize Middle Slices
"""

# ╔═╡ 778ef6f5-501d-4c85-9dbd-5432aa4bed21
let
	# Get middle slice indices
	nc_mid_z = nc_data.size_voxels[3] ÷ 2
	ccta_mid_z = ccta_data.size_voxels[3] ÷ 2

	# Extract middle slices
	nc_slice = nc_data.volume[:, :, nc_mid_z, 1, 1]
	ccta_slice = ccta_data.volume[:, :, ccta_mid_z, 1, 1]

	# Create figure
	fig = CM.Figure(size=(1000, 500))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast (slice $nc_mid_z)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays, colorrange=(-200, 400))

	ax2 = CM.Axis(fig[1, 2], title="CCTA Contrast (slice $ccta_mid_z)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_slice', colormap=:grays, colorrange=(-200, 400))

	fig
end

# ╔═╡ e536974f-8e81-4ac6-b8cc-c1c0d28c36d2
md"""
## Volumes Ready for Registration

The volumes are now in the format expected by MedicalImageRegistration: `(X, Y, Z, C, N)`

**Note:** Before registration, you may need to:
1. Resample to a common grid (for optimization)
2. Handle the contrast vs non-contrast intensity difference (Mutual Information loss)
3. Apply the final transform with `interpolation=:nearest` to preserve HU values
"""

# ╔═╡ 1f83def0-ee53-4c2c-979e-dd36d4197746
nc_volume = nc_data.volume  # Ready for registration

# ╔═╡ 170936b5-e78e-431d-8687-a860d95769ac
ccta_volume = ccta_data.volume  # Ready for registration

# ╔═╡ 8a3f2d1e-5b6c-4d8a-9e7f-1a2b3c4d5e6f
md"""
## Create PhysicalImage Objects

The `PhysicalImage` type wraps the volume data with physical spacing information.
This is critical for handling the resolution mismatch between the two scans:
- Non-contrast: thicker slices (larger z-spacing)
- CCTA: thinner slices (smaller z-spacing)
"""

# ╔═╡ 9b4e5f6a-7c8d-4e9a-0f1a-2b3c4d5e6f7a
begin
	# Create PhysicalImage from non-contrast volume
	# Ensure all spacing values are Float32 for consistency
	nc_spacing = Float32.(nc_data.spacing)  # (x, y, z) in mm
	nc_origin = Float32.(nc_data.origin)
	nc_physical = MIR.PhysicalImage(nc_data.volume; spacing=nc_spacing, origin=nc_origin)
	println("Non-contrast PhysicalImage created:")
	println("  Spacing: $(round.(MIR.spatial_spacing(nc_physical), digits=2)) mm")
	println("  Size: $(MIR.spatial_size(nc_physical)) voxels")
end

# ╔═╡ 0c5d6e7b-8f9a-4b0c-1d2e-3f4a5b6c7d8e
begin
	# Create PhysicalImage from CCTA contrast volume
	# Ensure all spacing values are Float32 for consistency
	ccta_spacing = Float32.(ccta_data.spacing)  # (x, y, z) in mm
	ccta_origin = Float32.(ccta_data.origin)
	ccta_physical = MIR.PhysicalImage(ccta_data.volume; spacing=ccta_spacing, origin=ccta_origin)
	println("CCTA PhysicalImage created:")
	println("  Spacing: $(round.(MIR.spatial_spacing(ccta_physical), digits=2)) mm")
	println("  Size: $(MIR.spatial_size(ccta_physical)) voxels")
end

# ╔═╡ 1d6e7f8c-9a0b-4c1d-2e3f-4a5b6c7d8e9f
md"""
## Registration with Mutual Information

Now we run the clinical registration workflow:

1. **Resample to common resolution** (2mm isotropic for speed)
2. **Register with MI loss** (handles contrast vs non-contrast intensity difference)
3. **Upsample transform** to original high resolution
4. **Apply with nearest-neighbor** to preserve exact HU values

**Why Mutual Information (MI)?**
- Blood in non-contrast: ~40 HU
- Blood in contrast: ~300+ HU
- MSE/NCC would PENALIZE correct alignment!
- MI measures statistical dependence - learns that 40 HU ↔ 300 HU

**Note:** Registration parameters can be adjusted:
- `registration_resolution`: Lower = faster but less accurate
- `affine_iterations`: More iterations = better alignment but slower
- `preserve_hu`: Set to `true` for quantitative analysis (calcium scoring, dose calc)
"""

# ╔═╡ 2e7f8a9d-0b1c-4d2e-3f4a-5b6c7d8e9f0a
begin
	# Run clinical registration
	# Moving: CCTA (contrast, 0.5mm-ish slices)
	# Static: Non-contrast (3mm-ish slices)
	println("Starting registration...")
	println("This will take a few moments...")

	registration_result = MIR.register_clinical(
		ccta_physical, nc_physical;
		registration_resolution=2.0f0,    # 2mm isotropic for optimization
		loss_fn=MIR.mi_loss,              # Mutual Information for contrast mismatch
		preserve_hu=true,                  # Nearest-neighbor final interpolation
		registration_type=:affine,         # Affine registration (global transform)
		affine_scales=(4, 2, 1),          # Multi-resolution pyramid
		affine_iterations=(50, 25, 10),   # Iterations per scale
		learning_rate=0.01f0,             # Optimizer learning rate
		verbose=true                       # Print progress
	)
end

# ╔═╡ 3f8a9b0c-1d2e-4f3a-5b6c-7d8e9f0a1b2c
md"""
## Registration Metrics

The registration result contains detailed metrics showing how well the alignment worked:
- **MI before**: Mutual Information before registration
- **MI after**: Mutual Information after registration
- **MI improvement**: The increase in MI (higher = better alignment)

A positive MI improvement indicates the registration successfully aligned the images.
"""

# ╔═╡ 4a9b0c1d-2e3f-4a5b-6c7d-8e9f0a1b2c3d
let
	println("=" ^ 60)
	println("REGISTRATION METRICS")
	println("=" ^ 60)
	println("MI Before:      $(round(registration_result.metrics[:mi_before], digits=4))")
	println("MI After:       $(round(registration_result.metrics[:mi_after], digits=4))")
	println("MI Improvement: $(round(registration_result.metrics[:mi_improvement], digits=4))")
	println()
	println("Metadata:")
	println("  Moving spacing: $(registration_result.metadata[:moving_spacing]) mm")
	println("  Static spacing: $(registration_result.metadata[:static_spacing]) mm")
	println("  Registration resolution: $(registration_result.metadata[:registration_resolution]) mm")
	println("  Preserve HU: $(registration_result.metadata[:preserve_hu])")
	println("=" ^ 60)
end

# ╔═╡ 5b0c1d2e-3f4a-5b6c-7d8e-9f0a1b2c3d4e
md"""
## HU Preservation Validation

Since we used `preserve_hu=true`, the output image should contain ONLY values that existed in the original CCTA scan (nearest-neighbor interpolation).

This is critical for:
- **Calcium scoring** (130 HU threshold)
- **Tissue density measurement**
- **Dose calculation** (HU → electron density)

Bilinear interpolation would create new values by averaging neighbors, corrupting quantitative analysis.
"""

# ╔═╡ 6c1d2e3f-4a5b-6c7d-8e9f-0a1b2c3d4e5f
let
	# Get unique HU values before and after
	original_values = Set(vec(Array(ccta_physical.data)))
	moved_values = Set(vec(Array(registration_result.moved_image.data)))

	n_original = length(original_values)
	n_moved = length(moved_values)

	hu_preserved = moved_values ⊆ original_values

	println("=" ^ 60)
	println("HU PRESERVATION VALIDATION")
	println("=" ^ 60)
	println("Original CCTA unique values: $(n_original)")
	println("Registered output unique values: $(n_moved)")
	println()

	if hu_preserved
		println("✓ HU PRESERVATION VERIFIED")
		println("  All output values exist in original input")
		println("  Safe for quantitative analysis!")
	else
		new_values = setdiff(moved_values, original_values)
		println("⚠ WARNING: $(length(new_values)) new values created")
		println("  Check interpolation settings")
	end

	println()
	println("Original HU range: $(extrema(Array(ccta_physical.data)))")
	println("Output HU range:   $(extrema(Array(registration_result.moved_image.data)))")
	println("=" ^ 60)
end

# ╔═╡ 7d2e3f4a-5b6c-7d8e-9f0a-1b2c3d4e5f6a
md"""
## Visualize Before/After Alignment

Compare the alignment before and after registration using:
1. **Difference images** - should be smaller after registration
2. **Checkerboard overlay** - should show smooth transitions at edges
"""

# ╔═╡ 8e3f4a5b-6c7d-8e9f-0a1b-2c3d4e5f6a7b
"""
Create a checkerboard overlay of two images for visual comparison.
Helps visualize alignment - good registration shows smooth transitions at checkerboard edges.
"""
function checkerboard_overlay(img1, img2; block_size=16)
	result = similar(img1)
	X, Y = size(img1, 1), size(img1, 2)
	for j in 1:Y, i in 1:X
		block_x = div(i - 1, block_size)
		block_y = div(j - 1, block_size)
		if mod(block_x + block_y, 2) == 0
			result[i, j] = img1[i, j]
		else
			result[i, j] = img2[i, j]
		end
	end
	return result
end

# ╔═╡ 9f4a5b6c-7d8e-9f0a-1b2c-3d4e5f6a7b8c
let
	# Get middle slice for visualization
	mid_z_nc = nc_data.size_voxels[3] ÷ 2
	mid_z_ccta = ccta_data.size_voxels[3] ÷ 2

	# Extract slices
	nc_slice = nc_data.volume[:, :, mid_z_nc, 1, 1]
	ccta_slice_before = ccta_data.volume[:, :, mid_z_ccta, 1, 1]

	# Get the moved CCTA slice at the same z position
	# Note: The moved image is now in the static (non-contrast) coordinate space
	moved_slice = registration_result.moved_image.data[:, :, mid_z_nc, 1, 1]

	# Normalize for visualization
	normalize_for_vis(img) = clamp.((img .- (-200)) ./ (400 - (-200)), 0, 1)

	nc_norm = normalize_for_vis(nc_slice)
	ccta_norm = normalize_for_vis(ccta_slice_before)
	moved_norm = normalize_for_vis(moved_slice)

	# Create checkerboards
	checker_before = checkerboard_overlay(nc_norm, ccta_norm; block_size=32)
	checker_after = checkerboard_overlay(nc_norm, moved_norm; block_size=32)

	# Create figure
	fig = CM.Figure(size=(1200, 800))

	# Top row: Original images
	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast (Static)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_norm', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="CCTA Before Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_norm', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="CCTA After Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax3, moved_norm', colormap=:grays)
	CM.hidedecorations!(ax3)

	# Bottom row: Checkerboard comparisons
	ax4 = CM.Axis(fig[2, 1], title="Checkerboard: NC + CCTA (Before)", aspect=CM.DataAspect())
	CM.heatmap!(ax4, checker_before', colormap=:grays)
	CM.hidedecorations!(ax4)

	ax5 = CM.Axis(fig[2, 2], title="Checkerboard: NC + CCTA (After)", aspect=CM.DataAspect())
	CM.heatmap!(ax5, checker_after', colormap=:grays)
	CM.hidedecorations!(ax5)

	# Difference image
	ax6 = CM.Axis(fig[2, 3], title="Abs Difference (After)", aspect=CM.DataAspect())
	diff_img = abs.(nc_norm .- moved_norm)
	CM.heatmap!(ax6, diff_img', colormap=:hot, colorrange=(0, 0.5))
	CM.hidedecorations!(ax6)

	CM.Label(fig[0, :], "Registration Results (Middle Slice)", fontsize=20)

	fig
end

# ╔═╡ 0a5b6c7d-8e9f-0a1b-2c3d-4e5f6a7b8c9d
md"""
## Slice-by-Slice Comparison

Browse through different slices to inspect the registration quality throughout the volume.
"""

# ╔═╡ 1b6c7d8e-9f0a-1b2c-3d4e-5f6a7b8c9d0e
@bind slice_idx UI.Slider(1:nc_data.size_voxels[3], default=nc_data.size_voxels[3]÷2, show_value=true)

# ╔═╡ 2c7d8e9f-0a1b-2c3d-4e5f-6a7b8c9d0e1f
let
	# Get slices at current index
	nc_slice = nc_data.volume[:, :, slice_idx, 1, 1]
	moved_slice = registration_result.moved_image.data[:, :, slice_idx, 1, 1]

	# Normalize for visualization
	normalize_for_vis(img) = clamp.((img .- (-200)) ./ (400 - (-200)), 0, 1)

	nc_norm = normalize_for_vis(nc_slice)
	moved_norm = normalize_for_vis(moved_slice)

	# Checkerboard
	checker = checkerboard_overlay(nc_norm, moved_norm; block_size=32)

	# Create figure
	fig = CM.Figure(size=(1000, 350))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_norm', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="Registered CCTA", aspect=CM.DataAspect())
	CM.heatmap!(ax2, moved_norm', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="Checkerboard Overlay", aspect=CM.DataAspect())
	CM.heatmap!(ax3, checker', colormap=:grays)
	CM.hidedecorations!(ax3)

	CM.Label(fig[0, :], "Slice $slice_idx of $(nc_data.size_voxels[3])", fontsize=16)

	fig
end

# ╔═╡ 3d8e9f0a-1b2c-3d4e-5f6a-7b8c9d0e1f2a
md"""
## Summary

This notebook demonstrated the complete clinical CT registration workflow:

1. **Loaded DICOM series** using DICOM.jl and converted to HU values
2. **Created PhysicalImage objects** with correct spacing metadata
3. **Registered with Mutual Information** to handle contrast vs non-contrast intensity mismatch
4. **Preserved HU values** using nearest-neighbor interpolation for the final output
5. **Validated registration** with metrics and visualizations

### Key Points for Clinical CT Registration

| Challenge | Solution |
|-----------|----------|
| **Resolution mismatch** (3mm vs 0.5mm) | Register at common resolution (2mm), upsample transform |
| **Contrast intensity difference** (40 HU vs 300 HU blood) | Mutual Information loss |
| **HU preservation for quantitative analysis** | `preserve_hu=true` (nearest-neighbor) |
| **FOV mismatch** | Automatic handling via PhysicalImage |

### When to Use Different Settings

| Use Case | Loss Function | Preserve HU |
|----------|--------------|-------------|
| Same-modality (CT to CT, no contrast) | `mse_loss` | Depends on analysis |
| Multi-modal (contrast vs non-contrast) | `mi_loss` | Yes for quantitative |
| Visual alignment only | `mse_loss` or `mi_loss` | No (bilinear is smoother) |
| Calcium scoring, dose calculation | `mi_loss` | **Always yes** |
"""

# ╔═╡ Cell order:
# ╠═a987dd55-059d-4a21-9bcc-9440b1899ed1
# ╠═c41f6864-09ef-4e45-a62a-4d63dc6cd9b2
# ╠═4df0ee4c-cdc1-4a92-8878-3508fe2378ff
# ╠═c569ab7c-67c4-46bd-8d8b-4f56a77990e5
# ╠═605f29e5-aeac-40a4-a0f9-cfa9882ce096
# ╠═8de86531-fada-421f-8ee1-dbfd4952b0da
# ╠═ecf6bdc9-109a-4a8c-b5b9-34424f2880a7
# ╠═351c65de-b533-4ebb-a055-744bbe94dd4b
# ╠═7cb187a4-c0c7-465f-a58f-ba5cb51d84fb
# ╠═48f6d4e5-08c8-4e13-afab-c37d71cd774b
# ╠═33f0b630-e3a3-4b03-a1b7-fec51489b37d
# ╟─3407d97d-485c-4cdf-bd46-9f9e051728bd
# ╠═ceea726d-db0f-4050-a00f-3e679dfe5c68
# ╠═e670750f-b885-401d-ba4c-30f18a0442af
# ╠═046e3d1d-cd6f-4b06-bb80-65ff7ea60fcf
# ╠═4c66ad49-3fdf-408b-807b-c0cbe786937f
# ╟─14f2d0b4-89b0-4b8c-b9cd-3c637d6a6fd4
# ╠═bb5d236f-41cc-46d4-b6ce-2ffa10cf33ff
# ╠═3c8bbb15-13cd-4531-be85-d163958debad
# ╟─149d9e89-c765-4ab0-bd64-f300f968b9cf
# ╠═3dc458fd-9957-41b5-b2fe-8b20173806ae
# ╟─cbafe59d-6f8a-4e25-a8f8-f3e86d7cf4d0
# ╠═a40e6d1d-f1a8-4601-bf7a-f597f0788ac5
# ╟─138f598d-0399-45f7-8c61-5336d460caaf
# ╠═1b8f0793-5728-4b17-93fe-7249c1cd3638
# ╟─fe28b08c-f926-404e-b273-5fad4f4734d1
# ╟─778ef6f5-501d-4c85-9dbd-5432aa4bed21
# ╟─e536974f-8e81-4ac6-b8cc-c1c0d28c36d2
# ╠═1f83def0-ee53-4c2c-979e-dd36d4197746
# ╠═170936b5-e78e-431d-8687-a860d95769ac
# ╟─8a3f2d1e-5b6c-4d8a-9e7f-1a2b3c4d5e6f
# ╠═9b4e5f6a-7c8d-4e9a-0f1a-2b3c4d5e6f7a
# ╠═0c5d6e7b-8f9a-4b0c-1d2e-3f4a5b6c7d8e
# ╟─1d6e7f8c-9a0b-4c1d-2e3f-4a5b6c7d8e9f
# ╠═2e7f8a9d-0b1c-4d2e-3f4a-5b6c7d8e9f0a
# ╟─3f8a9b0c-1d2e-4f3a-5b6c-7d8e9f0a1b2c
# ╠═4a9b0c1d-2e3f-4a5b-6c7d-8e9f0a1b2c3d
# ╟─5b0c1d2e-3f4a-5b6c-7d8e-9f0a1b2c3d4e
# ╠═6c1d2e3f-4a5b-6c7d-8e9f-0a1b2c3d4e5f
# ╟─7d2e3f4a-5b6c-7d8e-9f0a-1b2c3d4e5f6a
# ╠═8e3f4a5b-6c7d-8e9f-0a1b-2c3d4e5f6a7b
# ╠═9f4a5b6c-7d8e-9f0a-1b2c-3d4e5f6a7b8c
# ╟─0a5b6c7d-8e9f-0a1b-2c3d-4e5f6a7b8c9d
# ╠═1b6c7d8e-9f0a-1b2c-3d4e-5f6a7b8c9d0e
# ╠═2c7d8e9f-0a1b-2c3d-4e5f-6a7b8c9d0e1f
# ╟─3d8e9f0a-1b2c-3d4e-5f6a-7b8c9d0e1f2a
