# Tests for preprocessing pipeline (IMPL-PREPROCESS-001)
# Tests run on both CPU (Array) and GPU (MtlArray) where available

using Test
using MedicalImageRegistration
using Metal
using Random

# Check if Metal GPU is available
const HAS_METAL = Metal.functional()

# Helper to run tests on both CPU and GPU
function test_on_backends(f, name)
    @testset "$name - CPU" begin
        f(Array)
    end
    if HAS_METAL
        @testset "$name - Metal GPU" begin
            f(MtlArray)
        end
    end
end

# ============================================================================
# Tests for center_of_mass
# ============================================================================

@testset "center_of_mass" begin
    test_on_backends("3D center_of_mass") do ArrayType
        Random.seed!(42)

        # Create a simple image with known COM
        # Image is 64x64x64 with a single bright spot at (32, 32, 32)
        data = ArrayType(fill(Float32(-500), 64, 64, 64, 1, 1))  # Background at -500 HU

        # Put a bright region (1000 HU) at a known location
        # For MtlArray, we need to do this on CPU first then copy
        data_cpu = Array(data)
        data_cpu[30:34, 30:34, 30:34, 1, 1] .= 1000f0  # Bright cube centered near (32, 32, 32)
        copyto!(data, data_cpu)

        # Create PhysicalImage with 1mm isotropic spacing
        img = PhysicalImage(data; spacing=(1.0f0, 1.0f0, 1.0f0), origin=(0.0f0, 0.0f0, 0.0f0))

        # Compute COM
        com = center_of_mass(img; threshold=-200f0)

        # COM should be approximately at center of bright region (31.5, 31.5, 31.5) in physical coords
        # With 1mm spacing and origin at 0, voxel 32 corresponds to physical 31.0
        @test length(com) == 3
        @test com[1] ≈ 31.0f0 atol=2.0f0  # x
        @test com[2] ≈ 31.0f0 atol=2.0f0  # y
        @test com[3] ≈ 31.0f0 atol=2.0f0  # z
    end

    test_on_backends("3D center_of_mass with anisotropic spacing") do ArrayType
        Random.seed!(42)

        # Test with anisotropic spacing (like 3mm CT slices)
        # Image: 64x64x20 with 0.5mm x 0.5mm x 3mm spacing
        data = ArrayType(fill(Float32(-500), 64, 64, 20, 1, 1))

        data_cpu = Array(data)
        data_cpu[30:34, 30:34, 8:12, 1, 1] .= 500f0  # Bright region
        copyto!(data, data_cpu)

        img = PhysicalImage(data;
            spacing=(0.5f0, 0.5f0, 3.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        com = center_of_mass(img; threshold=-200f0)

        # Expected physical position:
        # x: voxel ~32 → (32-1)*0.5 = 15.5 mm
        # y: voxel ~32 → (32-1)*0.5 = 15.5 mm
        # z: voxel ~10 → (10-1)*3.0 = 27.0 mm
        @test com[1] ≈ 15.5f0 atol=2.0f0
        @test com[2] ≈ 15.5f0 atol=2.0f0
        @test com[3] ≈ 27.0f0 atol=5.0f0
    end

    test_on_backends("2D center_of_mass") do ArrayType
        Random.seed!(42)

        data = ArrayType(fill(Float32(-500), 64, 64, 1, 1))

        data_cpu = Array(data)
        data_cpu[40:44, 20:24, 1, 1] .= 800f0
        copyto!(data, data_cpu)

        img = PhysicalImage(data; spacing=(1.0f0, 1.0f0), origin=(0.0f0, 0.0f0))

        com = center_of_mass(img; threshold=-200f0)

        @test length(com) == 2
        @test com[1] ≈ 41.0f0 atol=2.0f0  # x
        @test com[2] ≈ 21.0f0 atol=2.0f0  # y
    end

    test_on_backends("center_of_mass threshold behavior") do ArrayType
        # All values below threshold should result in geometric center fallback
        data = ArrayType(fill(Float32(-600), 32, 32, 32, 1, 1))

        img = PhysicalImage(data; spacing=(1.0f0, 1.0f0, 1.0f0), origin=(0.0f0, 0.0f0, 0.0f0))

        com = center_of_mass(img; threshold=-200f0)

        # Should fall back to geometric center
        @test com[1] ≈ 15.5f0 atol=0.5f0
        @test com[2] ≈ 15.5f0 atol=0.5f0
        @test com[3] ≈ 15.5f0 atol=0.5f0
    end
end

# ============================================================================
# Tests for align_centers
# ============================================================================

@testset "align_centers" begin
    test_on_backends("3D align_centers basic") do ArrayType
        Random.seed!(42)

        # Create two images with COMs at different locations
        # Static: COM near center
        static_data = ArrayType(fill(Float32(-500), 64, 64, 64, 1, 1))
        static_cpu = Array(static_data)
        static_cpu[30:34, 30:34, 30:34, 1, 1] .= 500f0
        copyto!(static_data, static_cpu)

        static = PhysicalImage(static_data;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        # Moving: COM offset by (10, 10, 10) mm
        moving_data = ArrayType(fill(Float32(-500), 64, 64, 64, 1, 1))
        moving_cpu = Array(moving_data)
        moving_cpu[40:44, 40:44, 40:44, 1, 1] .= 500f0
        copyto!(moving_data, moving_cpu)

        moving = PhysicalImage(moving_data;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        # Align
        aligned, translation = align_centers(moving, static; threshold=-200f0)

        # Check that COMs now match within tolerance
        com_static = center_of_mass(static; threshold=-200f0)
        com_aligned = center_of_mass(aligned; threshold=-200f0)

        @test abs(com_aligned[1] - com_static[1]) < 0.5f0
        @test abs(com_aligned[2] - com_static[2]) < 0.5f0
        @test abs(com_aligned[3] - com_static[3]) < 0.5f0

        # Check translation is approximately (-10, -10, -10) mm
        @test translation[1] ≈ -10.0f0 atol=2.0f0
        @test translation[2] ≈ -10.0f0 atol=2.0f0
        @test translation[3] ≈ -10.0f0 atol=2.0f0
    end

    test_on_backends("align_centers returns PhysicalImage with correct metadata") do ArrayType
        static_data = ArrayType(fill(Float32(100), 32, 32, 32, 1, 1))
        moving_data = ArrayType(fill(Float32(100), 32, 32, 32, 1, 1))

        static = PhysicalImage(static_data;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )
        moving = PhysicalImage(moving_data;
            spacing=(2.0f0, 2.0f0, 2.0f0),
            origin=(10.0f0, 10.0f0, 10.0f0)
        )

        aligned, _ = align_centers(moving, static; threshold=0f0)

        # Spacing should be unchanged
        @test aligned.spacing == moving.spacing

        # Data should be the same object (not resampled)
        @test parent(aligned) === parent(moving)
    end

    test_on_backends("2D align_centers") do ArrayType
        static_data = ArrayType(fill(Float32(-500), 64, 64, 1, 1))
        static_cpu = Array(static_data)
        static_cpu[20:24, 20:24, 1, 1] .= 500f0
        copyto!(static_data, static_cpu)

        static = PhysicalImage(static_data; spacing=(1.0f0, 1.0f0), origin=(0.0f0, 0.0f0))

        moving_data = ArrayType(fill(Float32(-500), 64, 64, 1, 1))
        moving_cpu = Array(moving_data)
        moving_cpu[40:44, 40:44, 1, 1] .= 500f0
        copyto!(moving_data, moving_cpu)

        moving = PhysicalImage(moving_data; spacing=(1.0f0, 1.0f0), origin=(0.0f0, 0.0f0))

        aligned, translation = align_centers(moving, static; threshold=-200f0)

        com_static = center_of_mass(static; threshold=-200f0)
        com_aligned = center_of_mass(aligned; threshold=-200f0)

        @test abs(com_aligned[1] - com_static[1]) < 0.5f0
        @test abs(com_aligned[2] - com_static[2]) < 0.5f0
    end
end

# ============================================================================
# Tests for compute_overlap_region
# ============================================================================

@testset "compute_overlap_region" begin
    test_on_backends("3D fully overlapping images") do ArrayType
        # Two images that fully overlap
        data1 = ArrayType(zeros(Float32, 64, 64, 64, 1, 1))
        data2 = ArrayType(zeros(Float32, 64, 64, 64, 1, 1))

        img1 = PhysicalImage(data1;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )
        img2 = PhysicalImage(data2;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        overlap = compute_overlap_region(img1, img2)

        @test !isnothing(overlap)
        @test overlap.min == (0.0f0, 0.0f0, 0.0f0)
        @test overlap.max == (63.0f0, 63.0f0, 63.0f0)
    end

    test_on_backends("3D partially overlapping images") do ArrayType
        # Image 1: 0-63 mm in all dimensions
        data1 = ArrayType(zeros(Float32, 64, 64, 64, 1, 1))
        img1 = PhysicalImage(data1;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        # Image 2: 32-95 mm in all dimensions (overlaps from 32-63)
        data2 = ArrayType(zeros(Float32, 64, 64, 64, 1, 1))
        img2 = PhysicalImage(data2;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(32.0f0, 32.0f0, 32.0f0)
        )

        overlap = compute_overlap_region(img1, img2)

        @test !isnothing(overlap)
        @test overlap.min == (32.0f0, 32.0f0, 32.0f0)
        @test overlap.max == (63.0f0, 63.0f0, 63.0f0)
    end

    test_on_backends("3D non-overlapping images") do ArrayType
        data1 = ArrayType(zeros(Float32, 64, 64, 64, 1, 1))
        img1 = PhysicalImage(data1;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        # Image 2 starts after image 1 ends
        data2 = ArrayType(zeros(Float32, 64, 64, 64, 1, 1))
        img2 = PhysicalImage(data2;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(100.0f0, 100.0f0, 100.0f0)
        )

        overlap = compute_overlap_region(img1, img2)

        @test isnothing(overlap)
    end

    test_on_backends("3D tight FOV inside wide FOV (cardiac CT scenario)") do ArrayType
        # Wide FOV: 350mm extent (non-contrast CT)
        data_wide = ArrayType(zeros(Float32, 350, 350, 100, 1, 1))
        img_wide = PhysicalImage(data_wide;
            spacing=(1.0f0, 1.0f0, 3.0f0),  # 3mm slices
            origin=(-175.0f0, -175.0f0, 0.0f0)
        )

        # Tight FOV: 180mm extent (CCTA)
        data_tight = ArrayType(zeros(Float32, 360, 360, 200, 1, 1))
        img_tight = PhysicalImage(data_tight;
            spacing=(0.5f0, 0.5f0, 0.5f0),  # 0.5mm isotropic
            origin=(-90.0f0, -90.0f0, 50.0f0)  # Centered on heart
        )

        overlap = compute_overlap_region(img_wide, img_tight)

        @test !isnothing(overlap)
        # Overlap should be the tight FOV region (fully contained)
        @test overlap.min[1] ≈ -90.0f0 atol=1.0f0
        @test overlap.min[2] ≈ -90.0f0 atol=1.0f0
        @test overlap.max[1] ≈ 89.5f0 atol=1.0f0
        @test overlap.max[2] ≈ 89.5f0 atol=1.0f0
    end

    test_on_backends("2D overlap region") do ArrayType
        data1 = ArrayType(zeros(Float32, 100, 100, 1, 1))
        img1 = PhysicalImage(data1; spacing=(1.0f0, 1.0f0), origin=(0.0f0, 0.0f0))

        data2 = ArrayType(zeros(Float32, 50, 50, 1, 1))
        img2 = PhysicalImage(data2; spacing=(1.0f0, 1.0f0), origin=(25.0f0, 25.0f0))

        overlap = compute_overlap_region(img1, img2)

        @test !isnothing(overlap)
        @test overlap.min == (25.0f0, 25.0f0)
        @test overlap.max == (74.0f0, 74.0f0)
    end
end

# ============================================================================
# Tests for crop_to_overlap
# ============================================================================

@testset "crop_to_overlap" begin
    test_on_backends("3D crop produces correct size") do ArrayType
        data = ArrayType(rand(Float32, 100, 100, 50, 1, 1))
        img = PhysicalImage(data;
            spacing=(1.0f0, 1.0f0, 2.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        # Define a region to crop to (40mm cube starting at (20, 30, 20))
        region = (min=(20.0f0, 30.0f0, 20.0f0), max=(60.0f0, 70.0f0, 60.0f0))

        cropped = crop_to_overlap(img, region)

        # Check dimensions
        # x: (60-20)/1 + 1 = 41 voxels
        # y: (70-30)/1 + 1 = 41 voxels
        # z: (60-20)/2 + 1 = 21 voxels
        @test size(parent(cropped), 1) == 41
        @test size(parent(cropped), 2) == 41
        @test size(parent(cropped), 3) == 21
    end

    test_on_backends("3D crop preserves physical extent") do ArrayType
        data = ArrayType(rand(Float32, 100, 100, 50, 1, 1))
        img = PhysicalImage(data;
            spacing=(1.0f0, 1.0f0, 2.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        region = (min=(20.0f0, 30.0f0, 20.0f0), max=(60.0f0, 70.0f0, 60.0f0))

        cropped = crop_to_overlap(img, region)

        # Check new origin
        @test cropped.origin[1] ≈ 20.0f0 atol=0.5f0
        @test cropped.origin[2] ≈ 30.0f0 atol=0.5f0
        @test cropped.origin[3] ≈ 20.0f0 atol=0.5f0

        # Spacing should be unchanged
        @test cropped.spacing == img.spacing
    end

    test_on_backends("3D crop stays on GPU") do ArrayType
        data = ArrayType(rand(Float32, 64, 64, 32, 1, 1))
        img = PhysicalImage(data;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        region = (min=(10.0f0, 10.0f0, 10.0f0), max=(50.0f0, 50.0f0, 20.0f0))

        cropped = crop_to_overlap(img, region)

        @test parent(cropped) isa ArrayType
    end

    test_on_backends("2D crop" ) do ArrayType
        data = ArrayType(rand(Float32, 100, 100, 1, 1))
        img = PhysicalImage(data; spacing=(0.5f0, 0.5f0), origin=(0.0f0, 0.0f0))

        region = (min=(10.0f0, 20.0f0), max=(30.0f0, 40.0f0))

        cropped = crop_to_overlap(img, region)

        # x: (30-10)/0.5 + 1 = 41 voxels
        # y: (40-20)/0.5 + 1 = 41 voxels
        @test size(parent(cropped), 1) == 41
        @test size(parent(cropped), 2) == 41
        @test cropped.origin[1] ≈ 10.0f0 atol=0.25f0
        @test cropped.origin[2] ≈ 20.0f0 atol=0.25f0
    end
end

# ============================================================================
# Tests for window_intensity
# ============================================================================

@testset "window_intensity" begin
    test_on_backends("basic windowing") do ArrayType
        data = ArrayType(Float32[-1000, -500, 0, 500, 1000, 2000])
        data = reshape(data, 6, 1, 1, 1)

        windowed = window_intensity(data; min_hu=-200f0, max_hu=1000f0)

        windowed_cpu = Array(windowed)
        @test windowed_cpu[1, 1, 1, 1] == -200f0  # Clamped to min
        @test windowed_cpu[2, 1, 1, 1] == -200f0  # Clamped to min
        @test windowed_cpu[3, 1, 1, 1] == 0f0     # Unchanged
        @test windowed_cpu[4, 1, 1, 1] == 500f0   # Unchanged
        @test windowed_cpu[5, 1, 1, 1] == 1000f0  # At max
        @test windowed_cpu[6, 1, 1, 1] == 1000f0  # Clamped to max
    end

    test_on_backends("windowing PhysicalImage preserves metadata") do ArrayType
        data = ArrayType(rand(Float32, 64, 64, 32, 1, 1) .* 2000f0 .- 500f0)
        img = PhysicalImage(data;
            spacing=(0.5f0, 0.5f0, 2.0f0),
            origin=(10.0f0, 20.0f0, 30.0f0)
        )

        windowed = window_intensity(img; min_hu=-200f0, max_hu=500f0)

        @test windowed.spacing == img.spacing
        @test windowed.origin == img.origin
        @test size(parent(windowed)) == size(parent(img))
    end

    test_on_backends("windowing stays on GPU") do ArrayType
        data = ArrayType(rand(Float32, 32, 32, 16, 1, 1))
        windowed = window_intensity(data; min_hu=-100f0, max_hu=100f0)

        @test windowed isa ArrayType
    end
end

# ============================================================================
# Tests for preprocess_for_registration
# ============================================================================

@testset "preprocess_for_registration" begin
    test_on_backends("full pipeline 3D") do ArrayType
        Random.seed!(42)

        # Create synthetic images simulating cardiac CT scenario
        # Static: wide FOV with COM near (0, 0, 30)
        static_data = ArrayType(fill(Float32(-500), 128, 128, 50, 1, 1))
        static_cpu = Array(static_data)
        static_cpu[50:70, 50:70, 20:30, 1, 1] .= 200f0  # Soft tissue region
        copyto!(static_data, static_cpu)

        static = PhysicalImage(static_data;
            spacing=(1.0f0, 1.0f0, 2.0f0),
            origin=(-64.0f0, -64.0f0, 0.0f0)
        )

        # Moving: tight FOV with COM offset by ~(10, 10, 10)
        moving_data = ArrayType(fill(Float32(-500), 80, 80, 40, 1, 1))
        moving_cpu = Array(moving_data)
        moving_cpu[45:60, 45:60, 15:25, 1, 1] .= 200f0  # Soft tissue region
        copyto!(moving_data, moving_cpu)

        moving = PhysicalImage(moving_data;
            spacing=(1.0f0, 1.0f0, 1.5f0),
            origin=(-30.0f0, -30.0f0, 10.0f0)
        )

        # Run preprocessing
        moving_prep, static_prep, info = preprocess_for_registration(
            moving, static;
            registration_resolution=2.0f0,
            align_com=true,
            do_crop_to_overlap=true,
            window_hu=true,
            min_hu=-200f0,
            max_hu=500f0,
            com_threshold=-200f0
        )

        # Check that preprocessing returns valid images
        @test parent(moving_prep) isa ArrayType
        @test parent(static_prep) isa ArrayType

        # Check that images have similar sizes after preprocessing
        # Due to rounding in resampling, sizes may differ by 1-2 voxels
        moving_size = size(parent(moving_prep))
        static_size = size(parent(static_prep))
        @test abs(moving_size[1] - static_size[1]) <= 2
        @test abs(moving_size[2] - static_size[2]) <= 2
        @test abs(moving_size[3] - static_size[3]) <= 2

        # Check that spacing is the registration resolution
        @test moving_prep.spacing[1] ≈ 2.0f0
        @test moving_prep.spacing[2] ≈ 2.0f0
        @test moving_prep.spacing[3] ≈ 2.0f0

        # Check that info contains valid data
        @test length(info.translation) == 3
        @test length(info.com_moving) == 3
        @test length(info.com_static) == 3
        @test !isnothing(info.overlap_region)
    end

    test_on_backends("pipeline without COM alignment") do ArrayType
        static_data = ArrayType(fill(Float32(100), 64, 64, 32, 1, 1))
        moving_data = ArrayType(fill(Float32(100), 64, 64, 32, 1, 1))

        static = PhysicalImage(static_data;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )
        moving = PhysicalImage(moving_data;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        _, _, info = preprocess_for_registration(
            moving, static;
            registration_resolution=2.0f0,
            align_com=false,
            do_crop_to_overlap=false,
            window_hu=false
        )

        # Translation should be zero when COM alignment is disabled
        @test info.translation == (0.0f0, 0.0f0, 0.0f0)
    end

    test_on_backends("pipeline 2D") do ArrayType
        static_data = ArrayType(fill(Float32(-500), 128, 128, 1, 1))
        static_cpu = Array(static_data)
        static_cpu[50:70, 50:70, 1, 1] .= 200f0
        copyto!(static_data, static_cpu)

        static = PhysicalImage(static_data; spacing=(0.5f0, 0.5f0), origin=(-32.0f0, -32.0f0))

        moving_data = ArrayType(fill(Float32(-500), 100, 100, 1, 1))
        moving_cpu = Array(moving_data)
        moving_cpu[60:75, 60:75, 1, 1] .= 200f0
        copyto!(moving_data, moving_cpu)

        moving = PhysicalImage(moving_data; spacing=(0.5f0, 0.5f0), origin=(-25.0f0, -25.0f0))

        moving_prep, static_prep, info = preprocess_for_registration(
            moving, static;
            registration_resolution=1.0f0,
            align_com=true,
            do_crop_to_overlap=true,
            window_hu=true
        )

        @test parent(moving_prep) isa ArrayType
        @test parent(static_prep) isa ArrayType
        @test size(parent(moving_prep)) == size(parent(static_prep))
    end

    test_on_backends("pipeline preserves MtlArray type") do ArrayType
        data1 = ArrayType(fill(Float32(100), 64, 64, 32, 1, 1))
        data2 = ArrayType(fill(Float32(100), 64, 64, 32, 1, 1))

        img1 = PhysicalImage(data1;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )
        img2 = PhysicalImage(data2;
            spacing=(1.0f0, 1.0f0, 1.0f0),
            origin=(0.0f0, 0.0f0, 0.0f0)
        )

        moving_prep, static_prep, _ = preprocess_for_registration(
            img1, img2;
            registration_resolution=2.0f0
        )

        @test parent(moving_prep) isa ArrayType
        @test parent(static_prep) isa ArrayType
    end
end

# ============================================================================
# Integration test: Synthetic cardiac CT scenario
# ============================================================================

@testset "Integration: Cardiac CT Scenario" begin
    test_on_backends("tight FOV inside wide FOV registration prep") do ArrayType
        Random.seed!(42)

        # Non-contrast (wide FOV, 3mm slices, 350mm x 350mm x 300mm)
        # Simulating large chest FOV
        nc_data = ArrayType(fill(Float32(-500), 175, 175, 100, 1, 1))
        nc_cpu = Array(nc_data)
        # Add "soft tissue" in center (heart region)
        nc_cpu[70:105, 70:105, 30:70, 1, 1] .= 50f0   # ~50 HU
        # Add some "bone" (spine)
        nc_cpu[82:93, 85:90, 20:80, 1, 1] .= 400f0    # ~400 HU
        copyto!(nc_data, nc_cpu)

        nc = PhysicalImage(nc_data;
            spacing=(2.0f0, 2.0f0, 3.0f0),  # 2mm in-plane, 3mm slices
            origin=(-175.0f0, -175.0f0, 0.0f0)
        )

        # CCTA (tight FOV, 0.5mm slices, 180mm x 180mm x 100mm)
        # Simulating cardiac-focused scan
        ccta_data = ArrayType(fill(Float32(-500), 180, 180, 100, 1, 1))
        ccta_cpu = Array(ccta_data)
        # Add contrast-enhanced heart (~300 HU in blood pools)
        ccta_cpu[70:110, 70:110, 30:70, 1, 1] .= 300f0
        # Add some "bone" (spine)
        ccta_cpu[85:95, 85:92, 20:80, 1, 1] .= 500f0
        copyto!(ccta_data, ccta_cpu)

        ccta = PhysicalImage(ccta_data;
            spacing=(1.0f0, 1.0f0, 1.0f0),  # ~1mm isotropic
            origin=(-90.0f0, -90.0f0, 20.0f0)  # Centered on heart, higher z start
        )

        # Run preprocessing
        ccta_prep, nc_prep, info = preprocess_for_registration(
            ccta, nc;
            registration_resolution=2.0f0,
            align_com=true,
            do_crop_to_overlap=true,
            window_hu=true,
            min_hu=-200f0,
            max_hu=600f0,
            com_threshold=-200f0
        )

        # Verify preprocessing succeeded
        @test parent(ccta_prep) isa ArrayType
        @test parent(nc_prep) isa ArrayType

        # Both should have same size after preprocessing
        @test size(parent(ccta_prep)) == size(parent(nc_prep))

        # Should have 2mm isotropic spacing
        @test ccta_prep.spacing[1] ≈ 2.0f0
        @test nc_prep.spacing[1] ≈ 2.0f0

        # Translation should be non-zero (COMs were offset)
        @test !all(info.translation .== 0)

        # Overlap region should exist
        @test !isnothing(info.overlap_region)

        println("Cardiac CT preprocessing test:")
        println("  Translation: $(info.translation) mm")
        println("  COM moving: $(info.com_moving) mm")
        println("  COM static: $(info.com_static) mm")
        println("  Output size: $(size(parent(ccta_prep)))")
    end
end
