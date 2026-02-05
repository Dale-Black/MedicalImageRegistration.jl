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

# ╔═╡ 3407d97d-485c-4cdf-bd46-9f9e051728bd
md"""
## Helper Functions
"""

# ╔═╡ ceea726d-db0f-4050-a00f-3e679dfe5c68
begin
	# DICOM tag hex codes
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
		# Get pixel data (stored as rows x cols)
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

# ╔═╡ f1a2b3c4-d5e6-4f7a-8b9c-0d1e2f3a4b5c
begin
	"""
	Normalize image to [0, 1] for visualization using a fixed HU window.
	Default window: [-200, 400] HU (good for soft tissue CT).
	"""
	function normalize_for_vis(img; min_hu=-200f0, max_hu=400f0)
		return clamp.((img .- min_hu) ./ (max_hu - min_hu), 0f0, 1f0)
	end

	"""
	Create a checkerboard overlay of two 2D images for visual comparison.
	Alternating blocks show img1 and img2 - good alignment shows smooth transitions.
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
end

# ╔═╡ 14f2d0b4-89b0-4b8c-b9cd-3c637d6a6fd4
md"""
## Load DICOM Data
"""

# ╔═╡ ecf6bdc9-109a-4a8c-b5b9-34424f2880a7
non_contrast_dir = joinpath(@__DIR__, "DICOMs/ca score VMI 70")

# ╔═╡ 351c65de-b533-4ebb-a055-744bbe94dd4b
ccta_dir = joinpath(@__DIR__, "DICOMs/ccta mono 70")

# ╔═╡ 7cb187a4-c0c7-465f-a58f-ba5cb51d84fb
nc_dcms = DICOM.dcmdir_parse(non_contrast_dir)

# ╔═╡ 33f0b630-e3a3-4b03-a1b7-fec51489b37d
ccta_dcms = DICOM.dcmdir_parse(ccta_dir)

# ╔═╡ bb5d236f-41cc-46d4-b6ce-2ffa10cf33ff
nc_data = load_dicom_volume(nc_dcms)

# ╔═╡ 3c8bbb15-13cd-4531-be85-d163958debad
ccta_data = load_dicom_volume(ccta_dcms)

# ╔═╡ 10000001-0001-0001-0001-000000000001
md"""
# Section 1: Data Inspection

No preprocessing -- just examine both volumes to understand the registration challenge.
"""

# ╔═╡ 10000001-0001-0001-0001-000000000002
begin
	# Compute physical extents for overlap calculation
	nc_x_range = (nc_data.origin[1], nc_data.origin[1] + (nc_data.size_voxels[1] - 1) * nc_data.spacing[1])
	nc_y_range = (nc_data.origin[2], nc_data.origin[2] + (nc_data.size_voxels[2] - 1) * nc_data.spacing[2])
	nc_z_range = (nc_data.origin[3], nc_data.origin[3] + (nc_data.size_voxels[3] - 1) * nc_data.spacing[3])

	ccta_x_range = (ccta_data.origin[1], ccta_data.origin[1] + (ccta_data.size_voxels[1] - 1) * ccta_data.spacing[1])
	ccta_y_range = (ccta_data.origin[2], ccta_data.origin[2] + (ccta_data.size_voxels[2] - 1) * ccta_data.spacing[2])
	ccta_z_range = (ccta_data.origin[3], ccta_data.origin[3] + (ccta_data.size_voxels[3] - 1) * ccta_data.spacing[3])

	# Overlap in each dimension
	overlap_x = max(0f0, min(nc_x_range[2], ccta_x_range[2]) - max(nc_x_range[1], ccta_x_range[1]))
	overlap_y = max(0f0, min(nc_y_range[2], ccta_y_range[2]) - max(nc_y_range[1], ccta_y_range[1]))
	overlap_z = max(0f0, min(nc_z_range[2], ccta_z_range[2]) - max(nc_z_range[1], ccta_z_range[1]))

	nc_extent_x = nc_x_range[2] - nc_x_range[1]
	nc_extent_y = nc_y_range[2] - nc_y_range[1]
	nc_extent_z = nc_z_range[2] - nc_z_range[1]

	ccta_extent_x = ccta_x_range[2] - ccta_x_range[1]
	ccta_extent_y = ccta_y_range[2] - ccta_y_range[1]
	ccta_extent_z = ccta_z_range[2] - ccta_z_range[1]

	# Use minimum extent per dimension for overlap %
	pct_x = overlap_x / min(nc_extent_x, ccta_extent_x) * 100
	pct_y = overlap_y / min(nc_extent_y, ccta_extent_y) * 100
	pct_z = overlap_z / min(nc_extent_z, ccta_extent_z) * 100
	pct_overall = (overlap_x * overlap_y * overlap_z) / min(nc_extent_x * nc_extent_y * nc_extent_z, ccta_extent_x * ccta_extent_y * ccta_extent_z) * 100

	dx = ccta_data.origin[1] - nc_data.origin[1]
	dy = ccta_data.origin[2] - nc_data.origin[2]
	dz = ccta_data.origin[3] - nc_data.origin[3]
end

# ╔═╡ 10000001-0001-0001-0001-000000000006
md"""
| Property | Non-Contrast (Ca Score) | CCTA (Contrast) |
|----------|:-----------------------:|:---------------:|
| **Origin (mm)** | $(round.(nc_data.origin, digits=2)) | $(round.(ccta_data.origin, digits=2)) |
| **Spacing (mm)** | $(round.(nc_data.spacing, digits=3)) | $(round.(ccta_data.spacing, digits=3)) |
| **Size (voxels)** | $(nc_data.size_voxels) | $(ccta_data.size_voxels) |
| **Size (mm)** | $(round.(nc_data.size_mm, digits=1)) | $(round.(ccta_data.size_mm, digits=1)) |
| **HU range** | $(extrema(nc_data.volume)) | $(extrema(ccta_data.volume)) |

**FOV overlap:** $(round(pct_overall, digits=1))% (X: $(round(pct_x, digits=1))%, Y: $(round(pct_y, digits=1))%, Z: $(round(pct_z, digits=1))%)

**Translation between origins:** ($(round(dx, digits=2)), $(round(dy, digits=2)), $(round(dz, digits=2))) mm
"""

# ╔═╡ 10000001-0001-0001-0001-000000000003
md"""
### Matching Slice Visualization

We pick a physical z-position that exists in BOTH volumes (the midpoint of the z-overlap region) and show the corresponding slices side-by-side.
"""

# ╔═╡ 10000001-0001-0001-0001-000000000004
begin
	# Find a matching physical z-position: use midpoint of the z-overlap region
	z_overlap_min = max(nc_z_range[1], ccta_z_range[1])
	z_overlap_max = min(nc_z_range[2], ccta_z_range[2])
	z_match_phys = (z_overlap_min + z_overlap_max) / 2

	# Convert physical z to voxel index for each volume
	nc_z_idx = clamp(round(Int, (z_match_phys - nc_data.origin[3]) / nc_data.spacing[3]) + 1, 1, nc_data.size_voxels[3])
	ccta_z_idx = clamp(round(Int, (z_match_phys - ccta_data.origin[3]) / ccta_data.spacing[3]) + 1, 1, ccta_data.size_voxels[3])

	# Verify the physical z positions
	nc_z_actual = nc_data.origin[3] + (nc_z_idx - 1) * nc_data.spacing[3]
	ccta_z_actual = ccta_data.origin[3] + (ccta_z_idx - 1) * ccta_data.spacing[3]

	@info "Matching slice selected" z_phys=round(z_match_phys, digits=1) nc_slice="$(nc_z_idx)/$(nc_data.size_voxels[3])" ccta_slice="$(ccta_z_idx)/$(ccta_data.size_voxels[3])"
end

# ╔═╡ 10000001-0001-0001-0001-000000000005
let
	nc_slice = nc_data.volume[:, :, nc_z_idx, 1, 1]
	ccta_slice = ccta_data.volume[:, :, ccta_z_idx, 1, 1]

	fig = CM.Figure(size=(1000, 500))

	ax1 = CM.Axis(fig[1, 1], title="Non-Contrast (slice $nc_z_idx, z=$(round(nc_z_actual, digits=1))mm)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays, colorrange=(-200, 400))

	ax2 = CM.Axis(fig[1, 2], title="CCTA (slice $ccta_z_idx, z=$(round(ccta_z_actual, digits=1))mm)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_slice', colormap=:grays, colorrange=(-200, 400))

	CM.Label(fig[0, :], "Raw Data at Matching Physical Z-Position", fontsize=18)
	fig
end

# ╔═╡ 20000001-0001-0001-0001-000000000001
md"""
# Section 2: Prepare Registration Inputs (Full Resolution)

Both images are 512x512 in XY -- **no downsampling needed**. The only difference is Z: NC has 71 slices at 2mm, CCTA has 403 slices at 0.36mm. We map the CCTA onto the NC's coordinate grid (512x512x71) so they share the same voxel space. XY is untouched.

The multi-scale pyramid inside `AffineRegistration` handles coarse-to-fine internally (e.g., scale=4 -> 128x128x18).
"""

# ╔═╡ 20000001-0001-0001-0001-000000000002
begin
	# Static = NC at its native resolution (512x512x71)
	static_data = nc_data.volume

	# Map CCTA (512x512x403) onto NC grid (512x512x71) via identity affine.
	# XY stays 512x512. Only Z is remapped (403 slices -> 71 slices = 2mm spacing).
	theta_id = zeros(Float32, 3, 4, 1)
	theta_id[1,1,1] = 1f0; theta_id[2,2,1] = 1f0; theta_id[3,3,1] = 1f0

	moving_data = MIR.affine_transform(
		ccta_data.volume, theta_id;
		shape=nc_data.size_voxels,
		padding_mode=:border,
		interpolation=:bilinear
	)

	@info "Registration inputs prepared" static=size(static_data)[1:3] moving=size(moving_data)[1:3] spacing=round.(nc_data.spacing, digits=3)
end

# ╔═╡ 20000001-0001-0001-0001-000000000003
md"""
### Registration Input Visualization

CCTA mapped onto NC grid at full 512x512 XY resolution.
"""

# ╔═╡ 20000001-0001-0001-0001-000000000004
let
	nc_slice = nc_data.volume[:, :, nc_z_idx, 1, 1]
	ccta_slice = moving_data[:, :, nc_z_idx, 1, 1]

	fig = CM.Figure(size=(1000, 500))

	ax1 = CM.Axis(fig[1, 1], title="NC (512x512, slice $nc_z_idx)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays, colorrange=(-200, 400))

	ax2 = CM.Axis(fig[1, 2], title="CCTA on NC grid (512x512, slice $nc_z_idx)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_slice', colormap=:grays, colorrange=(-200, 400))

	CM.Label(fig[0, :], "Registration Inputs -- Full Resolution", fontsize=18)
	fig
end

# ╔═╡ 20000001-0001-0001-0001-000000000005
md""

# ╔═╡ 30000001-0001-0001-0001-000000000001
md"""
# Section 3: Registration at Full Resolution (512x512x71)

Affine registration using Mutual Information (MI) loss directly at **full 512x512 XY resolution**. MI is essential because contrast agent changes blood HU dramatically (NC ~40 vs CCTA ~300+).

The internal multi-scale pyramid handles coarse-to-fine:
- **Scale 4** (128x128x18): captures large translations/rotations
- **Scale 2** (256x256x36): refines alignment
- **Scale 1** (512x512x71): final full-resolution optimization
"""

# ╔═╡ 30000001-0001-0001-0001-000000000002
begin
	# Create AffineRegistration with high-quality settings for full-res
	affine_reg = MIR.AffineRegistration{Float32}(
		is_3d=true,
		batch_size=1,
		scales=(4, 2, 1),
		iterations=(20, 5, 1),
		learning_rate=0.005f0,
		with_translation=true,
		with_rotation=true,
		with_zoom=true,
		with_shear=true,
		array_type=Array
	)

	@info "Starting registration" size=size(moving_data)[1:3] scales=affine_reg.scales iterations=affine_reg.iterations lr=affine_reg.learning_rate

	moved_result = MIR.register(
		affine_reg, moving_data, static_data;
		loss_fn=MIR.mi_loss,
		verbose=true,
		final_interpolation=:bilinear
	)

	@info "Registration complete" total_iterations=length(affine_reg.loss_history)
end

# ╔═╡ 30000001-0001-0001-0001-000000000003
md"""
**Registration Summary**

| Metric | Value |
|--------|-------|
| Total iterations | $(length(affine_reg.loss_history)) |
| Initial loss | $(round(affine_reg.loss_history[1], digits=6)) |
| Final loss | $(round(affine_reg.loss_history[end], digits=6)) |
| Loss change | $(round(affine_reg.loss_history[1] - affine_reg.loss_history[end], digits=6)) |
| Status | $(affine_reg.loss_history[end] < affine_reg.loss_history[1] ? "Converged (loss decreased)" : "WARNING: Loss did not decrease!") |
"""

# ╔═╡ 30000001-0001-0001-0001-000000000005
begin
	learned_theta = MIR.get_affine(affine_reg)
	theta_cpu = Array(learned_theta)
end

# ╔═╡ 30000001-0001-0001-0001-000000000009
md"""
**Learned affine matrix (3x4):**

| | col 1 | col 2 | col 3 | translation |
|---|------:|------:|------:|------:|
| **row 1** | $(round(theta_cpu[1,1,1], digits=6)) | $(round(theta_cpu[1,2,1], digits=6)) | $(round(theta_cpu[1,3,1], digits=6)) | $(round(theta_cpu[1,4,1], digits=6)) |
| **row 2** | $(round(theta_cpu[2,1,1], digits=6)) | $(round(theta_cpu[2,2,1], digits=6)) | $(round(theta_cpu[2,3,1], digits=6)) | $(round(theta_cpu[2,4,1], digits=6)) |
| **row 3** | $(round(theta_cpu[3,1,1], digits=6)) | $(round(theta_cpu[3,2,1], digits=6)) | $(round(theta_cpu[3,3,1], digits=6)) | $(round(theta_cpu[3,4,1], digits=6)) |
"""

# ╔═╡ 30000001-0001-0001-0001-000000000006
md"""
### Before/After Registration (Bilinear, Full Resolution)
"""

# ╔═╡ 30000001-0001-0001-0001-000000000007
let
	nc_slice = normalize_for_vis(static_data[:, :, nc_z_idx, 1, 1])
	ccta_before = normalize_for_vis(moving_data[:, :, nc_z_idx, 1, 1])
	ccta_after = normalize_for_vis(moved_result[:, :, nc_z_idx, 1, 1])

	fig = CM.Figure(size=(1200, 400))

	ax1 = CM.Axis(fig[1, 1], title="NC (Static)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_slice', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="CCTA Before Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_before', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="CCTA After Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax3, ccta_after', colormap=:grays)
	CM.hidedecorations!(ax3)

	CM.Label(fig[0, :], "Full-Resolution Registration Result (512x512, slice $nc_z_idx)", fontsize=18)
	fig
end

# ╔═╡ 35000001-0001-0001-0001-000000000001
md"""
# Section 3b: HU-Preserving Output (Nearest-Neighbor)

The registration above used **bilinear** for smooth gradients during optimization. For the **quantitative output**, we re-apply the learned transform with **nearest-neighbor** interpolation directly to the original CCTA (512x512x403).

**Why this matters:** Bilinear creates *new* HU values by averaging (e.g., 40 + 300 -> 170 HU, a phantom value). For calcium scoring (130 HU threshold), tissue density, or any quantitative analysis, every output voxel must be an *exact* value from the original input.

Output values ⊆ input values -- guaranteed by nearest-neighbor.
"""

# ╔═╡ 35000001-0001-0001-0001-000000000002
begin
	# Apply learned transform to ORIGINAL CCTA with nearest-neighbor for HU preservation
	# "Before": identity -- CCTA in NC voxel space, no alignment
	theta_identity = zeros(Float32, 3, 4, 1)
	theta_identity[1,1,1] = 1f0; theta_identity[2,2,1] = 1f0; theta_identity[3,3,1] = 1f0

	ccta_before_fullres = MIR.affine_transform(
		ccta_data.volume, theta_identity;
		shape=nc_data.size_voxels,
		padding_mode=:border,
		interpolation=:nearest
	)

	# "After": learned affine with nearest-neighbor -- HU preserved
	moved_fullres = MIR.affine_transform(
		ccta_data.volume, learned_theta;
		shape=nc_data.size_voxels,
		padding_mode=:border,
		interpolation=:nearest
	)

	@info "HU-preserving output applied" input=ccta_data.size_voxels output=size(moved_fullres)[1:3] interpolation=:nearest
end

# ╔═╡ 35000001-0001-0001-0001-000000000003
let
	fig = CM.Figure(size=(1200, 400))

	ax1 = CM.Axis(fig[1, 1], title="NC Original (512x512)", aspect=CM.DataAspect())
	CM.heatmap!(ax1, nc_data.volume[:, :, nc_z_idx, 1, 1]', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="CCTA Before (Nearest)", aspect=CM.DataAspect())
	CM.heatmap!(ax2, ccta_before_fullres[:, :, nc_z_idx, 1, 1]', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[1, 3], title="CCTA Registered (Nearest, HU-safe)", aspect=CM.DataAspect())
	CM.heatmap!(ax3, moved_fullres[:, :, nc_z_idx, 1, 1]', colormap=:grays, colorrange=(-200, 400))
	CM.hidedecorations!(ax3)

	CM.Label(fig[0, :], "HU-Preserving Result (512x512, slice $nc_z_idx)", fontsize=18)
	fig
end

# ╔═╡ 40000001-0001-0001-0001-000000000001
md"""
# Section 4: Evaluation

Quantitative and visual evaluation at **full original XY resolution** (512x512).
"""

# ╔═╡ 40000001-0001-0001-0001-000000000002
md"""
### Checkerboard Overlay Before/After
"""

# ╔═╡ 40000001-0001-0001-0001-000000000003
let
	nc_slice = normalize_for_vis(nc_data.volume[:, :, nc_z_idx, 1, 1])
	ccta_before = normalize_for_vis(ccta_before_fullres[:, :, nc_z_idx, 1, 1])
	ccta_after = normalize_for_vis(moved_fullres[:, :, nc_z_idx, 1, 1])

	checker_before = checkerboard_overlay(nc_slice, ccta_before; block_size=64)
	checker_after = checkerboard_overlay(nc_slice, ccta_after; block_size=64)

	fig = CM.Figure(size=(1000, 500))

	ax1 = CM.Axis(fig[1, 1], title="Checkerboard BEFORE Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax1, checker_before', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="Checkerboard AFTER Registration", aspect=CM.DataAspect())
	CM.heatmap!(ax2, checker_after', colormap=:grays)
	CM.hidedecorations!(ax2)

	CM.Label(fig[0, :], "Checkerboard Overlay - Full Resolution (512x512, slice $nc_z_idx)", fontsize=18)
	fig
end

# ╔═╡ 40000001-0001-0001-0001-000000000004
md"""
### Difference Image Before/After
"""

# ╔═╡ 40000001-0001-0001-0001-000000000005
let
	nc_slice = normalize_for_vis(nc_data.volume[:, :, nc_z_idx, 1, 1])
	ccta_before = normalize_for_vis(ccta_before_fullres[:, :, nc_z_idx, 1, 1])
	ccta_after = normalize_for_vis(moved_fullres[:, :, nc_z_idx, 1, 1])

	diff_before = abs.(nc_slice .- ccta_before)
	diff_after = abs.(nc_slice .- ccta_after)

	fig = CM.Figure(size=(1000, 500))

	ax1 = CM.Axis(fig[1, 1], title="|NC - CCTA| BEFORE", aspect=CM.DataAspect())
	CM.heatmap!(ax1, diff_before', colormap=:jet, colorrange=(0, 0.5))
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 2], title="|NC - CCTA| AFTER", aspect=CM.DataAspect())
	CM.heatmap!(ax2, diff_after', colormap=:jet, colorrange=(0, 0.5))
	CM.hidedecorations!(ax2)

	CM.Label(fig[0, :], "Absolute Difference - Full Resolution (512x512, slice $nc_z_idx)", fontsize=18)
	fig
end

# ╔═╡ 40000001-0001-0001-0001-000000000006
md"""
### Mutual Information Before/After
"""

# ╔═╡ 40000001-0001-0001-0001-000000000007
begin
	# MI with bilinear interpolation (what optimizer saw)
	mi_before_bilinear = MIR.mi_loss(moving_data, static_data)
	mi_after_bilinear = MIR.mi_loss(moved_result, static_data)

	# MI with nearest-neighbor interpolation (HU-preserving output)
	mi_before_nearest = MIR.mi_loss(ccta_before_fullres, nc_data.volume)
	mi_after_nearest = MIR.mi_loss(moved_fullres, nc_data.volume)
end

# ╔═╡ 40000001-0001-0001-0001-00000000000b
md"""
**Mutual Information Evaluation** *(lower = better alignment)*

| Interpolation | Before | After | Change |
|---------------|-------:|------:|-------:|
| **Bilinear** (optimization) | $(round(mi_before_bilinear[1], digits=6)) | $(round(mi_after_bilinear[1], digits=6)) | $(round(mi_before_bilinear[1] - mi_after_bilinear[1], digits=6)) |
| **Nearest-neighbor** (HU-preserving) | $(round(mi_before_nearest[1], digits=6)) | $(round(mi_after_nearest[1], digits=6)) | $(round(mi_before_nearest[1] - mi_after_nearest[1], digits=6)) |

$(mi_after_nearest[1] < mi_before_nearest[1] ? "**MI improved** by $(round((mi_before_nearest[1] - mi_after_nearest[1]) / abs(mi_before_nearest[1]) * 100, digits=1))%" : "**WARNING:** MI did not improve!")
"""

# ╔═╡ 45000001-0001-0001-0001-000000000001
md"""
### HU Preservation Validation

Verify that nearest-neighbor interpolation preserved exact original HU values. Every voxel in the output must be a value that exists in the original CCTA input.
"""

# ╔═╡ 45000001-0001-0001-0001-000000000002
begin
	original_values = Set(unique(ccta_data.volume))
	output_values = Set(unique(moved_fullres))
	values_preserved = issubset(output_values, original_values)
end

# ╔═╡ 45000001-0001-0001-0001-000000000003
md"""
**HU Preservation Validation**

| Metric | Value |
|--------|-------|
| Original CCTA unique values | $(length(original_values)) |
| Registered output unique values | $(length(output_values)) |
| Output values ⊆ input values | $(values_preserved ? "**YES**" : "**NO!!!**") |
| Original HU range | $(extrema(ccta_data.volume)) |
| Output HU range | $(extrema(moved_fullres)) |

$(values_preserved ? "**PASS:** All output HU values are exact copies from the original. Safe for quantitative analysis." : "**FAIL:** $(length(setdiff(output_values, original_values))) new values created by interpolation!")
"""

# ╔═╡ 40000001-0001-0001-0001-000000000008
md"""
### Slice Browser (Full Resolution)

Browse through all slices of the registered result at full 512x512 XY resolution.
"""

# ╔═╡ 40000001-0001-0001-0001-000000000009
@bind browse_slice_idx UI.Slider(1:nc_data.size_voxels[3], default=nc_data.size_voxels[3] ÷ 2, show_value=true)

# ╔═╡ 40000001-0001-0001-0001-00000000000a
let
	nc_slice = normalize_for_vis(nc_data.volume[:, :, browse_slice_idx, 1, 1])
	ccta_after = normalize_for_vis(moved_fullres[:, :, browse_slice_idx, 1, 1])

	checker = checkerboard_overlay(nc_slice, ccta_after; block_size=64)

	fig = CM.Figure(size=(800, 800))

	ax1 = CM.Axis(fig[1, 1:2], title="Non-Contrast (512x512)")
	CM.heatmap!(ax1, nc_slice', colormap=:grays)
	CM.hidedecorations!(ax1)

	ax2 = CM.Axis(fig[1, 3:4], title="Registered CCTA (Full-Res)")
	CM.heatmap!(ax2, ccta_after', colormap=:grays)
	CM.hidedecorations!(ax2)

	ax3 = CM.Axis(fig[2, 2:3], title="Checkerboard")
	CM.heatmap!(ax3, checker', colormap=:grays)
	CM.hidedecorations!(ax3)

	n_z = nc_data.size_voxels[3]
	CM.Label(fig[0, :], "Full Resolution - Slice $browse_slice_idx of $n_z", fontsize=16)

	fig
end

# ╔═╡ Cell order:
# ╠═a987dd55-059d-4a21-9bcc-9440b1899ed1
# ╠═c41f6864-09ef-4e45-a62a-4d63dc6cd9b2
# ╠═4df0ee4c-cdc1-4a92-8878-3508fe2378ff
# ╠═c569ab7c-67c4-46bd-8d8b-4f56a77990e5
# ╠═605f29e5-aeac-40a4-a0f9-cfa9882ce096
# ╠═8de86531-fada-421f-8ee1-dbfd4952b0da
# ╟─3407d97d-485c-4cdf-bd46-9f9e051728bd
# ╠═ceea726d-db0f-4050-a00f-3e679dfe5c68
# ╠═e670750f-b885-401d-ba4c-30f18a0442af
# ╠═046e3d1d-cd6f-4b06-bb80-65ff7ea60fcf
# ╠═4c66ad49-3fdf-408b-807b-c0cbe786937f
# ╠═f1a2b3c4-d5e6-4f7a-8b9c-0d1e2f3a4b5c
# ╟─14f2d0b4-89b0-4b8c-b9cd-3c637d6a6fd4
# ╠═ecf6bdc9-109a-4a8c-b5b9-34424f2880a7
# ╠═351c65de-b533-4ebb-a055-744bbe94dd4b
# ╠═7cb187a4-c0c7-465f-a58f-ba5cb51d84fb
# ╠═33f0b630-e3a3-4b03-a1b7-fec51489b37d
# ╠═bb5d236f-41cc-46d4-b6ce-2ffa10cf33ff
# ╠═3c8bbb15-13cd-4531-be85-d163958debad
# ╟─10000001-0001-0001-0001-000000000001
# ╠═10000001-0001-0001-0001-000000000002
# ╟─10000001-0001-0001-0001-000000000006
# ╟─10000001-0001-0001-0001-000000000003
# ╠═10000001-0001-0001-0001-000000000004
# ╟─10000001-0001-0001-0001-000000000005
# ╟─20000001-0001-0001-0001-000000000001
# ╠═20000001-0001-0001-0001-000000000002
# ╟─20000001-0001-0001-0001-000000000003
# ╟─20000001-0001-0001-0001-000000000004
# ╟─20000001-0001-0001-0001-000000000005
# ╟─30000001-0001-0001-0001-000000000001
# ╠═30000001-0001-0001-0001-000000000002
# ╟─30000001-0001-0001-0001-000000000003
# ╠═30000001-0001-0001-0001-000000000005
# ╟─30000001-0001-0001-0001-000000000009
# ╟─30000001-0001-0001-0001-000000000006
# ╟─30000001-0001-0001-0001-000000000007
# ╟─35000001-0001-0001-0001-000000000001
# ╠═35000001-0001-0001-0001-000000000002
# ╟─35000001-0001-0001-0001-000000000003
# ╟─40000001-0001-0001-0001-000000000001
# ╟─40000001-0001-0001-0001-000000000002
# ╟─40000001-0001-0001-0001-000000000003
# ╟─40000001-0001-0001-0001-000000000004
# ╟─40000001-0001-0001-0001-000000000005
# ╟─40000001-0001-0001-0001-000000000006
# ╠═40000001-0001-0001-0001-000000000007
# ╟─40000001-0001-0001-0001-00000000000b
# ╟─45000001-0001-0001-0001-000000000001
# ╠═45000001-0001-0001-0001-000000000002
# ╟─45000001-0001-0001-0001-000000000003
# ╟─40000001-0001-0001-0001-000000000008
# ╟─40000001-0001-0001-0001-000000000009
# ╟─40000001-0001-0001-0001-00000000000a
