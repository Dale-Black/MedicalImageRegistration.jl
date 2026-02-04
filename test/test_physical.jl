using Test
using MedicalImageRegistration
using Random

# Try to load Metal for GPU testing
const HAS_METAL = try
    using Metal
    Metal.functional()
catch
    false
end

@testset "PhysicalImage" begin
    Random.seed!(42)

    @testset "PhysicalImage Construction" begin
        @testset "2D PhysicalImage" begin
            data = rand(Float32, 64, 64, 1, 1)
            img = PhysicalImage(data)

            @test size(img) == (64, 64, 1, 1)
            @test eltype(img) == Float32
            @test parent(img) === data
            @test MedicalImageRegistration.is_3d(img) == false
            @test MedicalImageRegistration.spatial_size(img) == (64, 64)
            @test MedicalImageRegistration.spatial_spacing(img) == (1.0f0, 1.0f0)
        end

        @testset "2D PhysicalImage with custom spacing" begin
            data = rand(Float32, 128, 256, 1, 2)
            img = PhysicalImage(data; spacing=(0.5f0, 0.25f0), origin=(10.0f0, 20.0f0))

            @test MedicalImageRegistration.spatial_spacing(img) == (0.5f0, 0.25f0)
            @test img.origin == (10.0f0, 20.0f0, 0.0f0)
        end

        @testset "3D PhysicalImage" begin
            data = rand(Float32, 64, 64, 32, 1, 1)
            img = PhysicalImage(data)

            @test size(img) == (64, 64, 32, 1, 1)
            @test MedicalImageRegistration.is_3d(img) == true
            @test MedicalImageRegistration.spatial_size(img) == (64, 64, 32)
            @test MedicalImageRegistration.spatial_spacing(img) == (1.0f0, 1.0f0, 1.0f0)
        end

        @testset "3D PhysicalImage with anisotropic spacing" begin
            # Simulate cardiac CT: 0.5mm x 0.5mm x 3mm
            data = rand(Float32, 512, 512, 100, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.5f0, 3.0f0))

            @test MedicalImageRegistration.spatial_spacing(img) == (0.5f0, 0.5f0, 3.0f0)
        end
    end

    @testset "Physical Extent and Bounds" begin
        @testset "2D extent" begin
            data = rand(Float32, 100, 200, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.25f0))

            extent = MedicalImageRegistration.physical_extent(img)
            @test extent[1] ≈ 49.5f0  # (100-1) * 0.5
            @test extent[2] ≈ 49.75f0 # (200-1) * 0.25
        end

        @testset "3D extent - anisotropic" begin
            # Static: 3mm slices, Moving: 0.5mm slices
            data_static = rand(Float32, 256, 256, 100, 1, 1)
            data_moving = rand(Float32, 256, 256, 600, 1, 1)

            img_static = PhysicalImage(data_static; spacing=(0.5f0, 0.5f0, 3.0f0))
            img_moving = PhysicalImage(data_moving; spacing=(0.4f0, 0.4f0, 0.5f0))

            extent_static = MedicalImageRegistration.physical_extent(img_static)
            extent_moving = MedicalImageRegistration.physical_extent(img_moving)

            # Both should cover similar z-range in physical space
            @test extent_static[3] ≈ 297.0f0  # (100-1) * 3.0
            @test extent_moving[3] ≈ 299.5f0  # (600-1) * 0.5
        end

        @testset "3D bounds" begin
            data = rand(Float32, 64, 64, 32, 1, 1)
            img = PhysicalImage(data; spacing=(2.0f0, 2.0f0, 2.0f0), origin=(10.0f0, 20.0f0, 30.0f0))

            bounds = MedicalImageRegistration.physical_bounds(img)
            @test bounds[1][1] ≈ 10.0f0   # x_min
            @test bounds[1][2] ≈ 136.0f0  # x_max = 10 + (64-1)*2
            @test bounds[2][1] ≈ 20.0f0   # y_min
            @test bounds[3][1] ≈ 30.0f0   # z_min
        end
    end

    @testset "Coordinate Transformations" begin
        @testset "2D voxel_to_physical" begin
            data = rand(Float32, 100, 100, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.5f0), origin=(10.0f0, 20.0f0))

            # First voxel (1,1) should be at origin
            x, y = MedicalImageRegistration.voxel_to_physical(img, 1, 1)
            @test x ≈ 10.0f0
            @test y ≈ 20.0f0

            # Voxel (11, 21) should be at (10 + 10*0.5, 20 + 20*0.5)
            x, y = MedicalImageRegistration.voxel_to_physical(img, 11, 21)
            @test x ≈ 15.0f0
            @test y ≈ 30.0f0
        end

        @testset "3D voxel_to_physical - anisotropic" begin
            data = rand(Float32, 256, 256, 100, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.5f0, 3.0f0), origin=(0.0f0, 0.0f0, 0.0f0))

            # Check that z-coordinate respects 3mm spacing
            x, y, z = MedicalImageRegistration.voxel_to_physical(img, 1, 1, 1)
            @test z ≈ 0.0f0

            x, y, z = MedicalImageRegistration.voxel_to_physical(img, 1, 1, 11)
            @test z ≈ 30.0f0  # (11-1) * 3.0
        end

        @testset "physical_to_voxel roundtrip" begin
            data = rand(Float32, 64, 64, 32, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.4f0, 3.0f0), origin=(10.0f0, 20.0f0, 30.0f0))

            # Roundtrip test
            i, j, k = 15, 25, 10
            phys = MedicalImageRegistration.voxel_to_physical(img, i, j, k)
            i2, j2, k2 = MedicalImageRegistration.physical_to_voxel(img, phys...)

            @test i2 ≈ Float32(i)
            @test j2 ≈ Float32(j)
            @test k2 ≈ Float32(k)
        end

        @testset "2D normalized coordinates" begin
            data = rand(Float32, 64, 64, 1, 1)
            img = PhysicalImage(data)

            # Center should be at (0, 0) normalized
            x, y = MedicalImageRegistration.voxel_to_normalized(img, 32.5, 32.5)
            @test abs(x) < 0.1f0
            @test abs(y) < 0.1f0

            # Corners should be at ±1
            x, y = MedicalImageRegistration.voxel_to_normalized(img, 1, 1)
            @test x ≈ -1.0f0
            @test y ≈ -1.0f0

            x, y = MedicalImageRegistration.voxel_to_normalized(img, 64, 64)
            @test x ≈ 1.0f0
            @test y ≈ 1.0f0
        end
    end

    @testset "affine_grid_physical" begin
        @testset "2D identity transform" begin
            # Identity affine should produce standard grid
            theta = zeros(Float32, 2, 3, 1)
            theta[1, 1, 1] = 1.0f0
            theta[2, 2, 1] = 1.0f0

            grid = affine_grid_physical(theta, (32, 32), (1.0f0, 1.0f0))

            @test size(grid) == (2, 32, 32, 1)

            # Grid should span [-1, 1]
            @test grid[1, 1, 1, 1] ≈ -1.0f0 atol=0.01f0
            @test grid[1, 32, 32, 1] ≈ 1.0f0 atol=0.01f0
        end

        @testset "3D identity transform - isotropic" begin
            theta = zeros(Float32, 3, 4, 1)
            theta[1, 1, 1] = 1.0f0
            theta[2, 2, 1] = 1.0f0
            theta[3, 3, 1] = 1.0f0

            grid = affine_grid_physical(theta, (16, 16, 16), (1.0f0, 1.0f0, 1.0f0))

            @test size(grid) == (3, 16, 16, 16, 1)
        end

        @testset "3D identity transform - anisotropic (cardiac CT)" begin
            # 3mm z-spacing vs 0.5mm xy-spacing
            theta = zeros(Float32, 3, 4, 1)
            theta[1, 1, 1] = 1.0f0
            theta[2, 2, 1] = 1.0f0
            theta[3, 3, 1] = 1.0f0

            # Size in voxels
            size_vox = (64, 64, 100)  # XY is small, Z has more physical extent
            spacing = (0.5f0, 0.5f0, 3.0f0)

            grid = affine_grid_physical(theta, size_vox, spacing)

            @test size(grid) == (3, 64, 64, 100, 1)

            # Physical extents:
            # X: (64-1) * 0.5 = 31.5 mm
            # Y: (64-1) * 0.5 = 31.5 mm
            # Z: (100-1) * 3.0 = 297 mm (largest)
            # So max_extent = 297, normalization is by Z

            # Grid corners should reflect physical geometry
            # The Z-axis should span full [-1, 1], while X and Y span smaller range
        end

        @testset "Translation in physical space" begin
            # 10mm translation in x
            theta = zeros(Float32, 3, 4, 1)
            theta[1, 1, 1] = 1.0f0
            theta[2, 2, 1] = 1.0f0
            theta[3, 3, 1] = 1.0f0

            # No translation
            grid1 = affine_grid_physical(theta, (32, 32, 32), (1.0f0, 1.0f0, 1.0f0))

            # Add translation (in normalized units, since we normalize by max_extent)
            # For 32x32x32 @ 1mm, max_extent = 31mm
            # 10mm = 10/31 * 2 ≈ 0.645 in normalized space
            theta[1, 4, 1] = 0.645f0
            grid2 = affine_grid_physical(theta, (32, 32, 32), (1.0f0, 1.0f0, 1.0f0))

            # X coordinates should be shifted
            @test grid2[1, 16, 16, 16, 1] > grid1[1, 16, 16, 16, 1]
            @test (grid2[1, 16, 16, 16, 1] - grid1[1, 16, 16, 16, 1]) ≈ 0.645f0 atol=0.01f0
        end

        @testset "affine_grid_physical with PhysicalImage" begin
            data = rand(Float32, 64, 64, 32, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.5f0, 2.0f0))

            theta = zeros(Float32, 3, 4, 1)
            theta[1, 1, 1] = 1.0f0
            theta[2, 2, 1] = 1.0f0
            theta[3, 3, 1] = 1.0f0

            grid = affine_grid_physical(theta, img)

            @test size(grid) == (3, 64, 64, 32, 1)
        end
    end

    @testset "resample" begin
        @testset "2D downsample" begin
            # Create 128x128 image at 0.5mm spacing
            data = rand(Float32, 128, 128, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.5f0))

            # Resample to 2mm spacing (should be ~32x32)
            img_ds = resample(img, (2.0f0, 2.0f0))

            # Check approximate size (physical extent preserved)
            @test size(img_ds.data, 1) ≈ 32 atol=2
            @test size(img_ds.data, 2) ≈ 32 atol=2
            @test MedicalImageRegistration.spatial_spacing(img_ds) == (2.0f0, 2.0f0)
        end

        @testset "3D downsample - anisotropic to isotropic" begin
            # Create image with anisotropic voxels (like cardiac CT)
            data = rand(Float32, 128, 128, 200, 1, 1)
            img = PhysicalImage(data; spacing=(0.5f0, 0.5f0, 0.5f0))

            # Resample to 2mm isotropic
            img_ds = resample(img, (2.0f0, 2.0f0, 2.0f0))

            # Should be smaller in all dimensions
            @test size(img_ds.data, 1) < size(img.data, 1)
            @test size(img_ds.data, 3) < size(img.data, 3)
        end

        @testset "3D upsample" begin
            # Create low-res image
            data = rand(Float32, 32, 32, 16, 1, 1)
            img = PhysicalImage(data; spacing=(2.0f0, 2.0f0, 4.0f0))

            # Resample to higher resolution
            img_us = resample(img, (1.0f0, 1.0f0, 1.0f0))

            # Should be larger
            @test size(img_us.data, 1) > size(img.data, 1)
            @test size(img_us.data, 3) > size(img.data, 3)
        end

        @testset "resample with nearest neighbor preserves values" begin
            # Create image with distinct values
            data = Float32[1 2; 3 4]
            data = reshape(data, 2, 2, 1, 1)
            img = PhysicalImage(data; spacing=(1.0f0, 1.0f0))

            # Resample with nearest neighbor - values should be subset of original
            img_rs = resample(img, (0.5f0, 0.5f0); interpolation=:nearest)

            unique_out = unique(img_rs.data)
            unique_in = unique(img.data)
            @test all(v -> v ∈ unique_in, unique_out)
        end
    end

    @testset "Convenience functions" begin
        @testset "with_spacing" begin
            data = rand(Float32, 64, 64, 32, 1, 1)
            img = PhysicalImage(data; spacing=(1.0f0, 1.0f0, 1.0f0))

            img2 = MedicalImageRegistration.with_spacing(img, (0.5f0, 0.5f0, 3.0f0))

            # Same data, different spacing
            @test img2.data === img.data
            @test MedicalImageRegistration.spatial_spacing(img2) == (0.5f0, 0.5f0, 3.0f0)
        end

        @testset "with_origin" begin
            data = rand(Float32, 64, 64, 32, 1, 1)
            img = PhysicalImage(data)

            img2 = MedicalImageRegistration.with_origin(img, (10.0f0, 20.0f0, 30.0f0))

            @test img2.data === img.data
            @test img2.origin == (10.0f0, 20.0f0, 30.0f0)
        end
    end

    if HAS_METAL
        @testset "GPU Tests (Metal)" begin
            @testset "PhysicalImage on GPU" begin
                data_cpu = rand(Float32, 64, 64, 32, 1, 1)
                data_gpu = MtlArray(data_cpu)

                img = PhysicalImage(data_gpu; spacing=(0.5f0, 0.5f0, 3.0f0))

                @test parent(img) isa MtlArray
                @test MedicalImageRegistration.spatial_spacing(img) == (0.5f0, 0.5f0, 3.0f0)
            end

            @testset "affine_grid_physical on GPU" begin
                theta = MtlArray(zeros(Float32, 3, 4, 1))
                theta_cpu = zeros(Float32, 3, 4, 1)
                theta_cpu[1, 1, 1] = 1.0f0
                theta_cpu[2, 2, 1] = 1.0f0
                theta_cpu[3, 3, 1] = 1.0f0
                copyto!(theta, theta_cpu)

                grid = affine_grid_physical(theta, (32, 32, 32), (0.5f0, 0.5f0, 3.0f0))

                @test grid isa MtlArray
                @test size(grid) == (3, 32, 32, 32, 1)
            end

            @testset "resample on GPU" begin
                data_cpu = rand(Float32, 64, 64, 32, 1, 1)
                data_gpu = MtlArray(data_cpu)
                img = PhysicalImage(data_gpu; spacing=(0.5f0, 0.5f0, 2.0f0))

                img_ds = resample(img, (2.0f0, 2.0f0, 2.0f0))

                @test parent(img_ds) isa MtlArray
            end

            @testset "Anisotropic voxel handling on GPU" begin
                # Simulate cardiac CT registration scenario
                # Static: 3mm slices (thick)
                # Moving: 0.5mm slices (thin)

                static_data = MtlArray(rand(Float32, 128, 128, 50, 1, 1))
                moving_data = MtlArray(rand(Float32, 128, 128, 300, 1, 1))

                img_static = PhysicalImage(static_data; spacing=(0.5f0, 0.5f0, 3.0f0))
                img_moving = PhysicalImage(moving_data; spacing=(0.4f0, 0.4f0, 0.5f0))

                # Both should cover similar physical z-range
                ext_s = MedicalImageRegistration.physical_extent(img_static)
                ext_m = MedicalImageRegistration.physical_extent(img_moving)

                # Z extents: (50-1)*3 = 147mm vs (300-1)*0.5 = 149.5mm
                @test abs(ext_s[3] - ext_m[3]) < 10  # Similar physical coverage

                # Generate grids for both - should work with anisotropic spacing
                theta = MtlArray(zeros(Float32, 3, 4, 1))
                theta_cpu = zeros(Float32, 3, 4, 1)
                theta_cpu[1, 1, 1] = 1.0f0
                theta_cpu[2, 2, 1] = 1.0f0
                theta_cpu[3, 3, 1] = 1.0f0
                copyto!(theta, theta_cpu)

                grid_static = affine_grid_physical(theta, img_static)
                grid_moving = affine_grid_physical(theta, img_moving)

                @test grid_static isa MtlArray
                @test grid_moving isa MtlArray
            end
        end
    else
        @info "Skipping GPU tests: Metal not available"
    end
end
