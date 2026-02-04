### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

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
	ccta_spacing = Float32.(ccta_data.spacing)  # (x, y, z) in mm
	ccta_origin = Float32.(ccta_data.origin)
	ccta_physical = MIR.PhysicalImage(ccta_data.volume; spacing=ccta_spacing, origin=ccta_origin)
	println("CCTA PhysicalImage created:")
	println("  Spacing: $(round.(MIR.spatial_spacing(ccta_physical), digits=2)) mm")
	println("  Size: $(MIR.spatial_size(ccta_physical)) voxels")
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889900
md"""
# Preprocessing Pipeline

Before registration can work, we need to preprocess the images. Without preprocessing, gradient descent cannot find a good solution because:

1. **FOV mismatch**: CCTA has a tight FOV (heart only), non-contrast has wide FOV (whole chest)
2. **Resolution mismatch**: 0.5mm vs 3mm z-spacing (6x difference!)
3. **Center misalignment**: The images are centered differently in physical space

We will apply each preprocessing step individually and visualize the result.
"""

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889901
md"""
## Step 1: Center of Mass Alignment

The center of mass (COM) provides a robust estimate of where the "body" is in each image. By aligning the COMs, we get a good initial translation that brings the two images into rough alignment. This is critical because gradient-based optimization needs a reasonable starting point.
"""

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889902
begin
	# Compute center of mass for both images
	com_nc = MIR.center_of_mass(nc_physical; threshold=-200f0)
	com_ccta = MIR.center_of_mass(ccta_physical; threshold=-200f0)

	println("Center of Mass (physical coordinates in mm):")
	println("  Non-contrast: $(round.(com_nc, digits=1))")
	println("  CCTA:         $(round.(com_ccta, digits=1))")
	println()
	println("COM difference (translation needed):")
	com_diff = round.(com_nc .- com_ccta, digits=1)
	println("  dx=$(com_diff[1]) mm, dy=$(com_diff[2]) mm, dz=$(com_diff[3]) mm")
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889903
begin
	# Align CCTA center of mass to non-contrast COM
	ccta_aligned, com_translation = MIR.align_centers(ccta_physical, nc_physical; threshold=-200f0)

	println("COM alignment applied:")
	println("  Translation: $(round.(com_translation, digits=1)) mm")
	println("  CCTA origin before: $(round.(ccta_physical.origin, digits=1))")
	println("  CCTA origin after:  $(round.(ccta_aligned.origin, digits=1))")
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889904
md"""
## Step 2: FOV Overlap Detection

The CCTA scan has a tight FOV (covers mostly the heart region) while the non-contrast scan has a wide FOV (covers the whole chest). We need to find the overlapping region where both scans have valid data.
"""

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889905
begin
	# Compute physical bounds of each image
	nc_bounds = MIR.physical_bounds(nc_physical)
	ccta_bounds_orig = MIR.physical_bounds(ccta_physical)
	ccta_bounds_aligned = MIR.physical_bounds(ccta_aligned)

	println("Physical bounds (mm):")
	println()
	println("Non-contrast (static):")
	println("  X: $(round.(nc_bounds[1], digits=1))")
	println("  Y: $(round.(nc_bounds[2], digits=1))")
	println("  Z: $(round.(nc_bounds[3], digits=1))")
	println()
	println("CCTA before alignment:")
	println("  X: $(round.(ccta_bounds_orig[1], digits=1))")
	println("  Y: $(round.(ccta_bounds_orig[2], digits=1))")
	println("  Z: $(round.(ccta_bounds_orig[3], digits=1))")
	println()
	println("CCTA after COM alignment:")
	println("  X: $(round.(ccta_bounds_aligned[1], digits=1))")
	println("  Y: $(round.(ccta_bounds_aligned[2], digits=1))")
	println("  Z: $(round.(ccta_bounds_aligned[3], digits=1))")
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889906
begin
	# Detect overlapping region
	overlap_region = MIR.compute_overlap_region(ccta_aligned, nc_physical)

	if isnothing(overlap_region)
		println("WARNING: No overlap detected! Check image positions.")
	else
		overlap_extent = overlap_region.max .- overlap_region.min
		println("Overlap region detected:")
		println("  Min corner: $(round.(overlap_region.min, digits=1)) mm")
		println("  Max corner: $(round.(overlap_region.max, digits=1)) mm")
		println("  Extent: $(round.(overlap_extent, digits=1)) mm")
		println()

		# Compute what percentage of each image is in the overlap
		nc_extent = MIR.physical_extent(nc_physical)
		ccta_extent = MIR.physical_extent(ccta_aligned)

		nc_overlap_pct = round.(overlap_extent ./ nc_extent .* 100, digits=0)
		ccta_overlap_pct = round.(overlap_extent ./ ccta_extent .* 100, digits=0)
		println("Overlap as % of each image:")
		println("  Non-contrast: X=$(nc_overlap_pct[1])%, Y=$(nc_overlap_pct[2])%, Z=$(nc_overlap_pct[3])%")
		println("  CCTA:         X=$(ccta_overlap_pct[1])%, Y=$(ccta_overlap_pct[2])%, Z=$(ccta_overlap_pct[3])%")
	end
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889907
begin
	# Crop both images to the overlapping region
	nc_cropped = MIR.crop_to_overlap(nc_physical, overlap_region)
	ccta_cropped = MIR.crop_to_overlap(ccta_aligned, overlap_region)

	println("Cropped images:")
	println("  Non-contrast: $(MIR.spatial_size(nc_physical)) -> $(MIR.spatial_size(nc_cropped))")
	println("  CCTA:         $(MIR.spatial_size(ccta_aligned)) -> $(MIR.spatial_size(ccta_cropped))")
	println()
	println("Cropped physical extents:")
	println("  Non-contrast: $(round.(MIR.physical_extent(nc_cropped), digits=1)) mm")
	println("  CCTA:         $(round.(MIR.physical_extent(ccta_cropped), digits=1)) mm")
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889908
let
	# Visualize cropped images side-by-side
	nc_mid = MIR.spatial_size(nc_cropped)[3] ÷ 2
	ccta_mid = MIR.spatial_size(ccta_cropped)[3] ÷ 2

	nc_slice = nc_cropped.data[:, :, nc_mid, 1, 1]
	ccta_slice = ccta_cropped.data[:, :, ccta_mid, 1, 1]

	fig = CM.Figure(size=(1000, 500))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast Cropped (slice $nc_mid)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays, colorrange=(-200, 400))

	ax2 = CM.Axis(fig[1, 2], title="CCTA Cropped (slice $ccta_mid)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_slice', colormap=:grays, colorrange=(-200, 400))

	CM.Label(fig[0, :], "After FOV Overlap Crop", fontsize=18)
	fig
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889909
md"""
## Step 3: Resample to Common Resolution

The two images have very different resolutions, especially in z. We resample both to a common isotropic resolution (2mm) for registration. This makes the optimization tractable and ensures both images occupy the same voxel grid.
"""

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff667788990a
begin
	# Resample both to 2mm isotropic
	target_spacing = (2.0f0, 2.0f0, 2.0f0)
	nc_resampled = MIR.resample(nc_cropped, target_spacing; interpolation=:bilinear)
	ccta_resampled = MIR.resample(ccta_cropped, target_spacing; interpolation=:bilinear)

	println("Resampled images (2mm isotropic):")
	println("  Non-contrast: $(MIR.spatial_size(nc_cropped)) @ $(round.(MIR.spatial_spacing(nc_cropped), digits=2))mm")
	println("             -> $(MIR.spatial_size(nc_resampled)) @ $(round.(MIR.spatial_spacing(nc_resampled), digits=2))mm")
	println()
	println("  CCTA:         $(MIR.spatial_size(ccta_cropped)) @ $(round.(MIR.spatial_spacing(ccta_cropped), digits=2))mm")
	println("             -> $(MIR.spatial_size(ccta_resampled)) @ $(round.(MIR.spatial_spacing(ccta_resampled), digits=2))mm")
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff667788990b
let
	# Visualize resampled images side-by-side
	nc_mid = MIR.spatial_size(nc_resampled)[3] ÷ 2
	ccta_mid = MIR.spatial_size(ccta_resampled)[3] ÷ 2

	nc_slice = nc_resampled.data[:, :, nc_mid, 1, 1]
	ccta_slice = ccta_resampled.data[:, :, ccta_mid, 1, 1]

	fig = CM.Figure(size=(1000, 500))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast Resampled (slice $nc_mid)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays, colorrange=(-200, 400))

	ax2 = CM.Axis(fig[1, 2], title="CCTA Resampled (slice $ccta_mid)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_slice', colormap=:grays, colorrange=(-200, 400))

	CM.Label(fig[0, :], "After Resampling to 2mm Isotropic", fontsize=18)
	fig
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff667788990c
md"""
## Step 4: Intensity Windowing

Intensity windowing clips extreme HU values (very dense bone, metal artifacts, extreme air). This focuses the registration on the soft tissue range where alignment matters most, and reduces the influence of outliers on the loss function.
"""

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff667788990d
begin
	# Apply HU windowing
	nc_windowed = MIR.window_intensity(nc_resampled; min_hu=-200f0, max_hu=1000f0)
	ccta_windowed = MIR.window_intensity(ccta_resampled; min_hu=-200f0, max_hu=1000f0)

	println("Intensity windowing applied: [-200, 1000] HU")
	println("  Non-contrast HU range: $(extrema(nc_resampled.data)) -> $(extrema(nc_windowed.data))")
	println("  CCTA HU range:         $(extrema(ccta_resampled.data)) -> $(extrema(ccta_windowed.data))")
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff667788990e
let
	# Visualize intensity histograms before/after windowing
	fig = CM.Figure(size=(1000, 400))

	# Before windowing
	ax1 = CM.Axis(fig[1, 1], title="NC Before Windowing", xlabel="HU", ylabel="Count")
	nc_vals_before = vec(nc_resampled.data)
	CM.hist!(ax1, nc_vals_before, bins=100, color=(:blue, 0.5))

	ax2 = CM.Axis(fig[1, 2], title="NC After Windowing [-200, 1000]", xlabel="HU", ylabel="Count")
	nc_vals_after = vec(nc_windowed.data)
	CM.hist!(ax2, nc_vals_after, bins=100, color=(:blue, 0.5))

	ax3 = CM.Axis(fig[2, 1], title="CCTA Before Windowing", xlabel="HU", ylabel="Count")
	ccta_vals_before = vec(ccta_resampled.data)
	CM.hist!(ax3, ccta_vals_before, bins=100, color=(:red, 0.5))

	ax4 = CM.Axis(fig[2, 2], title="CCTA After Windowing [-200, 1000]", xlabel="HU", ylabel="Count")
	ccta_vals_after = vec(ccta_windowed.data)
	CM.hist!(ax4, ccta_vals_after, bins=100, color=(:red, 0.5))

	fig
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff667788990f
md"""
## Preprocessing Summary

Overview of all preprocessing steps applied to bring the images into a common space for registration.
"""

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889910
begin
	"""
	Create a checkerboard overlay of two images for visual comparison.
	Helps visualize alignment - good registration shows smooth transitions at edges.
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

	# Normalization helper
	normalize_for_vis(img) = clamp.((img .- (-200)) ./ (400 - (-200)), 0, 1)
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889911
let
	# Show preprocessing progression: Original -> Cropped -> Resampled for both images
	# Use middle slices at each stage

	fig = CM.Figure(size=(1200, 800))

	# Row labels
	CM.Label(fig[1, 0], "NC\n(Static)", fontsize=14, rotation=pi/2, tellheight=false)
	CM.Label(fig[2, 0], "CCTA\n(Moving)", fontsize=14, rotation=pi/2, tellheight=false)

	# Column labels
	CM.Label(fig[0, 1], "Original", fontsize=16)
	CM.Label(fig[0, 2], "Cropped to Overlap", fontsize=16)
	CM.Label(fig[0, 3], "Resampled (2mm)", fontsize=16)

	# NC original
	nc_orig_mid = nc_data.size_voxels[3] ÷ 2
	nc_orig_slice = nc_data.volume[:, :, nc_orig_mid, 1, 1]
	ax1 = CM.Axis(fig[1, 1], aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_orig_slice', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax1)

	# NC cropped
	nc_crop_mid = MIR.spatial_size(nc_cropped)[3] ÷ 2
	nc_crop_slice = nc_cropped.data[:, :, nc_crop_mid, 1, 1]
	ax2 = CM.Axis(fig[1, 2], aspect=CM.DataAspect())
	CM.heatmap!(ax2, nc_crop_slice', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax2)

	# NC resampled
	nc_res_mid = MIR.spatial_size(nc_resampled)[3] ÷ 2
	nc_res_slice = nc_resampled.data[:, :, nc_res_mid, 1, 1]
	ax3 = CM.Axis(fig[1, 3], aspect=CM.DataAspect())
	CM.heatmap!(ax3, nc_res_slice', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax3)

	# CCTA original
	ccta_orig_mid = ccta_data.size_voxels[3] ÷ 2
	ccta_orig_slice = ccta_data.volume[:, :, ccta_orig_mid, 1, 1]
	ax4 = CM.Axis(fig[2, 1], aspect=CM.DataAspect())
	CM.heatmap!(ax4, ccta_orig_slice', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax4)

	# CCTA cropped
	ccta_crop_mid = MIR.spatial_size(ccta_cropped)[3] ÷ 2
	ccta_crop_slice = ccta_cropped.data[:, :, ccta_crop_mid, 1, 1]
	ax5 = CM.Axis(fig[2, 2], aspect=CM.DataAspect())
	CM.heatmap!(ax5, ccta_crop_slice', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax5)

	# CCTA resampled
	ccta_res_mid = MIR.spatial_size(ccta_resampled)[3] ÷ 2
	ccta_res_slice = ccta_resampled.data[:, :, ccta_res_mid, 1, 1]
	ax6 = CM.Axis(fig[2, 3], aspect=CM.DataAspect())
	CM.heatmap!(ax6, ccta_res_slice', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax6)

	CM.Label(fig[-1, :], "Preprocessing Pipeline: Original → Cropped → Resampled", fontsize=20)
	fig
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889912
let
	# Checkerboard of preprocessed pair (should show rough alignment from COM)
	nc_mid = MIR.spatial_size(nc_windowed)[3] ÷ 2
	ccta_mid = MIR.spatial_size(ccta_windowed)[3] ÷ 2

	nc_slice = normalize_for_vis(nc_windowed.data[:, :, nc_mid, 1, 1])
	ccta_slice = normalize_for_vis(ccta_windowed.data[:, :, ccta_mid, 1, 1])

	# Images may have different sizes - use the smaller dimensions
	min_x = min(size(nc_slice, 1), size(ccta_slice, 1))
	min_y = min(size(nc_slice, 2), size(ccta_slice, 2))
	nc_trimmed = nc_slice[1:min_x, 1:min_y]
	ccta_trimmed = ccta_slice[1:min_x, 1:min_y]

	checker = checkerboard_overlay(nc_trimmed, ccta_trimmed; block_size=16)

	fig = CM.Figure(size=(1000, 400))

	ax1 = CM.Axis(fig[1, 1], title="NC Preprocessed", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_trimmed', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="CCTA Preprocessed", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_trimmed', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="Checkerboard (Preprocessed)", aspect=CM.DataAspect())
	CM.heatmap!(ax3, checker', colormap=:grays)
	CM.hidedecorations!(ax3)

	CM.Label(fig[0, :], "Preprocessed Images - Rough Alignment from COM", fontsize=18)
	fig
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889913
md"""
## Slice Browser (Preprocessed)

Browse through slices of the preprocessed images to inspect the alignment quality after preprocessing (before registration).
"""

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889914
@bind preprocess_slice_idx UI.Slider(1:min(MIR.spatial_size(nc_windowed)[3], MIR.spatial_size(ccta_windowed)[3]), default=min(MIR.spatial_size(nc_windowed)[3], MIR.spatial_size(ccta_windowed)[3]) ÷ 2, show_value=true)

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889915
let
	nc_slice = normalize_for_vis(nc_windowed.data[:, :, preprocess_slice_idx, 1, 1])
	ccta_slice = normalize_for_vis(ccta_windowed.data[:, :, preprocess_slice_idx, 1, 1])

	# Handle potentially different sizes
	min_x = min(size(nc_slice, 1), size(ccta_slice, 1))
	min_y = min(size(nc_slice, 2), size(ccta_slice, 2))
	nc_trimmed = nc_slice[1:min_x, 1:min_y]
	ccta_trimmed = ccta_slice[1:min_x, 1:min_y]

	checker = checkerboard_overlay(nc_trimmed, ccta_trimmed; block_size=16)

	fig = CM.Figure(size=(1000, 350))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_trimmed', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="CCTA (Preprocessed)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_trimmed', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="Checkerboard", aspect=CM.DataAspect())
	CM.heatmap!(ax3, checker', colormap=:grays)
	CM.hidedecorations!(ax3)

	nc_z_total = MIR.spatial_size(nc_windowed)[3]
	CM.Label(fig[0, :], "Preprocessed Slice $preprocess_slice_idx of $nc_z_total", fontsize=16)

	fig
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889916
md"""
# Registration

Now we run affine registration on the preprocessed images using Mutual Information (MI) loss. MI is essential for our use case because contrast agent changes blood from ~40 HU (non-contrast) to ~300+ HU (contrast). MSE would penalize correct alignment, but MI learns that these correspond to the same structures.
"""

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001100
md"""
## Step 5: Registration with Mutual Information
"""

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001101
begin
	# Ensure both preprocessed images have the same size for registration
	# After resampling, sizes may differ by 1-2 voxels due to rounding
	nc_prep_size = MIR.spatial_size(nc_windowed)
	ccta_prep_size = MIR.spatial_size(ccta_windowed)
	println("Sizes before alignment:")
	println("  NC:   $nc_prep_size")
	println("  CCTA: $ccta_prep_size")

	# Use the NC (static) size as target
	# If CCTA is different size, use register_clinical's internal size matching
	# For now, just use the raw arrays
	moving_data = ccta_windowed.data
	static_data = nc_windowed.data

	# If sizes differ, create matching-size arrays via resample
	if nc_prep_size != ccta_prep_size
		# Resample CCTA to match NC size
		println("  Resampling CCTA to match NC size...")
		theta_id = zeros(Float32, 3, 4, 1)
		theta_id[1,1,1] = 1f0; theta_id[2,2,1] = 1f0; theta_id[3,3,1] = 1f0
		grid_match = MIR.affine_grid(theta_id, nc_prep_size; align_corners=true)
		moving_data = MIR.grid_sample(ccta_windowed.data, grid_match; padding_mode=:border, align_corners=true)
		println("  CCTA resampled to: $(size(moving_data)[1:3])")
	end

	println("\nRegistration input sizes:")
	println("  Moving (CCTA): $(size(moving_data)[1:3])")
	println("  Static (NC):   $(size(static_data)[1:3])")
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001102
begin
	# Create AffineRegistration and run
	affine_reg = MIR.AffineRegistration{Float32}(
		is_3d=true,
		batch_size=1,
		scales=(4, 2, 1),
		iterations=(50, 25, 10),
		learning_rate=0.01f0,
		with_translation=true,
		with_rotation=true,
		with_zoom=true,
		with_shear=false,
		array_type=Array
	)

	println("Running affine registration with MI loss...")
	println("  Scales: $(affine_reg.scales)")
	println("  Iterations: $(affine_reg.iterations)")
	println("  Learning rate: $(affine_reg.learning_rate)")
	println()

	# Register
	moved_prep = MIR.register(
		affine_reg, moving_data, static_data;
		loss_fn=MIR.mi_loss,
		verbose=true,
		final_interpolation=:bilinear
	)

	println("\nRegistration complete!")
	println("  Loss history length: $(length(affine_reg.loss_history))")
	if !isempty(affine_reg.loss_history)
		println("  Initial loss: $(round(affine_reg.loss_history[1], digits=4))")
		println("  Final loss:   $(round(affine_reg.loss_history[end], digits=4))")
	end
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001103
begin
	# Get the learned affine matrix
	learned_theta = MIR.get_affine(affine_reg)
	println("Learned affine matrix (3x4):")
	theta_cpu = Array(learned_theta)
	for i in 1:3
		row = [round(theta_cpu[i, j, 1], digits=4) for j in 1:4]
		println("  ", row)
	end
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001104
let
	# Visualize: preprocessed moving BEFORE vs AFTER registration
	mid_z = MIR.spatial_size(nc_windowed)[3] ÷ 2

	nc_slice = normalize_for_vis(static_data[:, :, mid_z, 1, 1])
	ccta_before = normalize_for_vis(moving_data[:, :, mid_z, 1, 1])
	ccta_after = normalize_for_vis(moved_prep[:, :, mid_z, 1, 1])

	# Checkerboards
	checker_before = checkerboard_overlay(nc_slice, ccta_before; block_size=16)
	checker_after = checkerboard_overlay(nc_slice, ccta_after; block_size=16)

	fig = CM.Figure(size=(1200, 800))

	# Top row: images
	ax1 = CM.Axis(fig[1, 1], title="NC (Static)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="CCTA Before Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_before', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="CCTA After Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax3, ccta_after', colormap=:grays)
	CM.hidedecorations!(ax3)

	# Bottom row: checkerboards and difference
	ax4 = CM.Axis(fig[2, 1], title="Checkerboard (Before)", aspect=CM.DataAspect())
	CM.heatmap!(ax4, checker_before', colormap=:grays)
	CM.hidedecorations!(ax4)

	ax5 = CM.Axis(fig[2, 2], title="Checkerboard (After)", aspect=CM.DataAspect())
	CM.heatmap!(ax5, checker_after', colormap=:grays)
	CM.hidedecorations!(ax5)

	ax6 = CM.Axis(fig[2, 3], title="|NC - Registered CCTA|", aspect=CM.DataAspect())
	diff_img = abs.(nc_slice .- ccta_after)
	CM.heatmap!(ax6, diff_img', colormap=:hot, colorrange=(0, 0.5))
	CM.hidedecorations!(ax6)

	CM.Label(fig[0, :], "Registration Results (Preprocessed, Slice $mid_z)", fontsize=20)
	fig
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001105
md"""
## Step 6: Apply Transform to Original Resolution

The registration was performed at 2mm isotropic on preprocessed images. Now we need to apply the learned affine transform to the **original high-resolution CCTA** (0.5mm) using nearest-neighbor interpolation to preserve exact HU values.
"""

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001106
begin
	# Apply learned affine to the ORIGINAL CCTA volume at full resolution
	# We use the affine matrix learned on preprocessed images
	# and apply it to the original data with nearest-neighbor

	# The affine was learned in normalized [-1,1] space, so it applies
	# to any resolution - just need to match the output size

	# Target: register CCTA into non-contrast coordinate space
	nc_target_size = MIR.spatial_size(nc_physical)
	println("Applying transform to original resolution:")
	println("  CCTA original size: $(MIR.spatial_size(ccta_physical))")
	println("  NC target size:     $nc_target_size")
	println("  Using nearest-neighbor for HU preservation")

	moved_original = MIR.affine_transform(
		ccta_physical.data,
		learned_theta;
		shape=nc_target_size,
		padding_mode=:border,
		align_corners=true,
		interpolation=:nearest
	)

	println("  Output size: $(size(moved_original)[1:3])")
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001107
md"""
## Step 7: HU Preservation Validation

Since we used `interpolation=:nearest`, the output image should contain ONLY values that existed in the original CCTA scan. This is critical for quantitative analysis like calcium scoring (130 HU threshold).
"""

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001108
begin
	# Validate HU preservation
	original_values = Set(vec(ccta_physical.data))
	moved_values = Set(vec(moved_original))

	n_original = length(original_values)
	n_moved = length(moved_values)

	hu_preserved = moved_values ⊆ original_values

	println("=" ^ 60)
	println("HU PRESERVATION VALIDATION")
	println("=" ^ 60)
	println("Original CCTA unique values: $n_original")
	println("Registered output unique values: $n_moved")
	println()

	if hu_preserved
		println("HU PRESERVATION VERIFIED")
		println("  All output values exist in original input")
		println("  Safe for quantitative analysis!")
	else
		new_values = setdiff(moved_values, original_values)
		println("WARNING: $(length(new_values)) new values created")
		println("  Check interpolation settings")
	end

	println()
	println("Original HU range: $(extrema(ccta_physical.data))")
	println("Output HU range:   $(extrema(moved_original))")
	println("=" ^ 60)
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-778899001109
md"""
## Step 8: Final Comparison

Compare the non-contrast, original CCTA, and registered CCTA side by side.
"""

# ╔═╡ bb22cc33-dd44-ee55-ff66-77889900110a
let
	# 3-panel comparison at NC resolution
	mid_z = nc_data.size_voxels[3] ÷ 2

	nc_slice = nc_data.volume[:, :, mid_z, 1, 1]
	moved_slice = moved_original[:, :, mid_z, 1, 1]

	nc_norm = normalize_for_vis(nc_slice)
	moved_norm = normalize_for_vis(moved_slice)

	# Checkerboard: NC vs registered CCTA (at original resolution)
	checker = checkerboard_overlay(nc_norm, moved_norm; block_size=32)

	fig = CM.Figure(size=(1200, 400))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast (Static)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_norm', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="Registered CCTA (Nearest-Neighbor)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, moved_norm', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="Checkerboard Overlay", aspect=CM.DataAspect())
	CM.heatmap!(ax3, checker', colormap=:grays)
	CM.hidedecorations!(ax3)

	CM.Label(fig[0, :], "Final Result (Original Resolution, Slice $mid_z)", fontsize=20)
	fig
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-77889900110b
md"""
## Slice-by-Slice Comparison (Final Result)

Browse through slices of the final registered result.
"""

# ╔═╡ bb22cc33-dd44-ee55-ff66-77889900110c
@bind final_slice_idx UI.Slider(1:nc_data.size_voxels[3], default=nc_data.size_voxels[3] ÷ 2, show_value=true)

# ╔═╡ bb22cc33-dd44-ee55-ff66-77889900110d
let
	nc_slice = normalize_for_vis(nc_data.volume[:, :, final_slice_idx, 1, 1])
	moved_slice = normalize_for_vis(moved_original[:, :, final_slice_idx, 1, 1])

	checker = checkerboard_overlay(nc_slice, moved_slice; block_size=32)

	fig = CM.Figure(size=(1000, 350))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="Registered CCTA", aspect=CM.DataAspect())
	CM.heatmap!(ax2, moved_slice', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="Checkerboard", aspect=CM.DataAspect())
	CM.heatmap!(ax3, checker', colormap=:grays)
	CM.hidedecorations!(ax3)

	CM.Label(fig[0, :], "Slice $final_slice_idx of $(nc_data.size_voxels[3])", fontsize=16)
	fig
end

# ╔═╡ bb22cc33-dd44-ee55-ff66-77889900110e
md"""
## Summary

This notebook demonstrated the complete clinical cardiac CT registration workflow:

| Step | What it Does | Why it Matters |
|------|-------------|----------------|
| **1. COM Alignment** | Translates CCTA to align body centers | Gets images roughly overlapping |
| **2. FOV Overlap** | Finds region where both scans have data | CCTA FOV is smaller than non-contrast |
| **3. Resampling** | Brings both to 2mm isotropic grid | Needed for optimization (same voxel grid) |
| **4. Windowing** | Clips to [-200, 1000] HU | Focuses on tissue, removes extreme artifacts |
| **5. Registration** | Affine registration with MI loss | Handles contrast intensity mismatch |
| **6. Apply Transform** | Apply affine to original CCTA at full res | Get high-resolution registered output |
| **7. HU Validation** | Verify nearest-neighbor preserves HU | Critical for quantitative analysis |
| **8. Comparison** | Side-by-side and checkerboard views | Visual quality assessment |

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
# ╟─8a3f2d1e-5b6c-4d8a-9e7f-1a2b3c4d5e6f
# ╠═9b4e5f6a-7c8d-4e9a-0f1a-2b3c4d5e6f7a
# ╠═0c5d6e7b-8f9a-4b0c-1d2e-3f4a5b6c7d8e
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889900
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889901
# ╠═aa11bb22-cc33-dd44-ee55-ff6677889902
# ╠═aa11bb22-cc33-dd44-ee55-ff6677889903
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889904
# ╠═aa11bb22-cc33-dd44-ee55-ff6677889905
# ╠═aa11bb22-cc33-dd44-ee55-ff6677889906
# ╠═aa11bb22-cc33-dd44-ee55-ff6677889907
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889908
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889909
# ╠═aa11bb22-cc33-dd44-ee55-ff667788990a
# ╟─aa11bb22-cc33-dd44-ee55-ff667788990b
# ╟─aa11bb22-cc33-dd44-ee55-ff667788990c
# ╠═aa11bb22-cc33-dd44-ee55-ff667788990d
# ╟─aa11bb22-cc33-dd44-ee55-ff667788990e
# ╟─aa11bb22-cc33-dd44-ee55-ff667788990f
# ╠═aa11bb22-cc33-dd44-ee55-ff6677889910
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889911
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889912
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889913
# ╠═aa11bb22-cc33-dd44-ee55-ff6677889914
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889915
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889916
# ╟─bb22cc33-dd44-ee55-ff66-778899001100
# ╠═bb22cc33-dd44-ee55-ff66-778899001101
# ╠═bb22cc33-dd44-ee55-ff66-778899001102
# ╠═bb22cc33-dd44-ee55-ff66-778899001103
# ╟─bb22cc33-dd44-ee55-ff66-778899001104
# ╟─bb22cc33-dd44-ee55-ff66-778899001105
# ╠═bb22cc33-dd44-ee55-ff66-778899001106
# ╟─bb22cc33-dd44-ee55-ff66-778899001107
# ╠═bb22cc33-dd44-ee55-ff66-778899001108
# ╟─bb22cc33-dd44-ee55-ff66-778899001109
# ╟─bb22cc33-dd44-ee55-ff66-77889900110a
# ╟─bb22cc33-dd44-ee55-ff66-77889900110b
# ╠═bb22cc33-dd44-ee55-ff66-77889900110c
# ╟─bb22cc33-dd44-ee55-ff66-77889900110d
# ╟─bb22cc33-dd44-ee55-ff66-77889900110e
