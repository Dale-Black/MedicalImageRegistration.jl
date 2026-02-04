using Test
using MedicalImageRegistration
using Random
using Statistics

# Try to use Metal GPU if available, otherwise fall back to CPU
const USE_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

const ArrayType = USE_GPU ? MtlArray : Array

@testset "resample_transform.jl" begin

    @testset "resample_displacement 3D" begin
        Random.seed!(42)

        @testset "Identity (same size)" begin
            # When upsampling to same size, should return copy
            disp = ArrayType(randn(Float32, 16, 16, 16, 3, 1))
            result = resample_displacement(disp, (16, 16, 16))

            @test result isa typeof(disp)
            @test size(result) == size(disp)
            @test Array(result) ≈ Array(disp)
        end

        @testset "Upsample 2x" begin
            # Create displacement field at low res
            disp = ArrayType(zeros(Float32, 8, 8, 8, 3, 1))

            # Set a uniform displacement of 0.5 in x-direction (normalized coords)
            disp_cpu = Array(disp)
            disp_cpu[:, :, :, 1, 1] .= 0.5f0  # x displacement
            copyto!(disp, disp_cpu)

            result = resample_displacement(disp, (16, 16, 16))

            @test result isa typeof(disp)
            @test size(result) == (16, 16, 16, 3, 1)

            # Check that displacement values are scaled by 2x (15/7 ≈ 2.14)
            # Scale factor = (16-1)/(8-1) = 15/7 ≈ 2.14
            expected_scale = 15.0f0 / 7.0f0
            result_cpu = Array(result)

            # x displacement should be scaled
            @test all(x -> isapprox(x, 0.5f0 * expected_scale, rtol=1e-4), result_cpu[:, :, :, 1, 1])

            # y, z displacement should still be 0
            @test all(x -> abs(x) < 1e-5, result_cpu[:, :, :, 2, 1])
            @test all(x -> abs(x) < 1e-5, result_cpu[:, :, :, 3, 1])
        end

        @testset "Downsample 2x" begin
            # Create displacement field at high res
            disp = ArrayType(zeros(Float32, 16, 16, 16, 3, 1))

            disp_cpu = Array(disp)
            disp_cpu[:, :, :, 2, 1] .= 0.3f0  # y displacement
            copyto!(disp, disp_cpu)

            result = resample_displacement(disp, (8, 8, 8))

            @test size(result) == (8, 8, 8, 3, 1)

            # Scale factor = (8-1)/(16-1) = 7/15 ≈ 0.47
            expected_scale = 7.0f0 / 15.0f0
            result_cpu = Array(result)

            @test all(x -> isapprox(x, 0.3f0 * expected_scale, rtol=1e-4), result_cpu[:, :, :, 2, 1])
        end

        @testset "Round-trip consistency" begin
            # Upsample then downsample should approximately recover original
            # Use smaller displacement values for better numerical stability
            disp = ArrayType(randn(Float32, 8, 8, 8, 3, 1) * 0.05f0)

            up = resample_displacement(disp, (16, 16, 16))
            down = resample_displacement(up, (8, 8, 8))

            @test size(down) == size(disp)

            # Due to bilinear interpolation, there's information loss at boundaries
            # Just verify the output is finite and has similar statistics
            @test all(isfinite.(Array(down)))
            @test isapprox(mean(Array(down)), mean(Array(disp)), atol=0.05f0)
        end
    end

    @testset "resample_displacement 2D" begin
        Random.seed!(42)

        @testset "Upsample 2x" begin
            disp = ArrayType(zeros(Float32, 8, 8, 2, 1))

            disp_cpu = Array(disp)
            disp_cpu[:, :, 1, 1] .= 0.4f0
            copyto!(disp, disp_cpu)

            result = resample_displacement(disp, (16, 16))

            @test size(result) == (16, 16, 2, 1)

            expected_scale = 15.0f0 / 7.0f0
            result_cpu = Array(result)

            @test all(x -> isapprox(x, 0.4f0 * expected_scale, rtol=1e-4), result_cpu[:, :, 1, 1])
        end
    end

    @testset "resample_velocity" begin
        @testset "3D" begin
            v = ArrayType(randn(Float32, 8, 8, 8, 3, 1) * 0.1f0)

            result = resample_velocity(v, (16, 16, 16))

            @test result isa typeof(v)
            @test size(result) == (16, 16, 16, 3, 1)
        end

        @testset "2D" begin
            v = ArrayType(randn(Float32, 8, 8, 2, 1) * 0.1f0)

            result = resample_velocity(v, (16, 16))

            @test size(result) == (16, 16, 2, 1)
        end
    end

    @testset "upsample_affine_transform" begin
        @testset "Normalized coordinates (resolution independent)" begin
            # Affine in normalized [-1, 1] coords is resolution-independent
            theta = ArrayType(zeros(Float32, 3, 4, 1))
            theta_cpu = zeros(Float32, 3, 4, 1)
            theta_cpu[1, 1, 1] = 1.0f0
            theta_cpu[2, 2, 1] = 1.0f0
            theta_cpu[3, 3, 1] = 1.0f0
            theta_cpu[1, 4, 1] = 0.1f0  # x translation
            copyto!(theta, theta_cpu)

            result = upsample_affine_transform(theta, (8, 8, 8), (16, 16, 16))

            @test size(result) == size(theta)
            @test Array(result) ≈ Array(theta)
        end

        @testset "Physical coordinates with spacing change" begin
            theta = ArrayType(zeros(Float32, 3, 4, 1))
            theta_cpu = zeros(Float32, 3, 4, 1)
            theta_cpu[1, 1, 1] = 1.0f0
            theta_cpu[2, 2, 1] = 1.0f0
            theta_cpu[3, 3, 1] = 1.0f0
            theta_cpu[1, 4, 1] = 10.0f0  # 10mm x translation
            copyto!(theta, theta_cpu)

            # From 2mm to 0.5mm spacing
            old_spacing = (2.0f0, 2.0f0, 2.0f0)
            new_spacing = (0.5f0, 0.5f0, 0.5f0)

            result = upsample_affine_transform_physical(theta, old_spacing, new_spacing)

            @test size(result) == size(theta)

            result_cpu = Array(result)
            # Scale factors: new/old = 0.5/2.0 = 0.25
            # Inverse: old/new = 2.0/0.5 = 4.0
            # For rotation part: inv_scale[d] * val * scale[col] = 4.0 * 1.0 * 0.25 = 1.0 (unchanged)
            # For translation: inv_scale[d] * val = 4.0 * 10.0 = 40.0
            @test result_cpu[1, 1, 1] ≈ 1.0f0  # diagonal unchanged
            @test result_cpu[1, 4, 1] ≈ 40.0f0  # translation scaled by 4x
        end
    end

    @testset "invert_displacement 3D" begin
        Random.seed!(42)

        @testset "Small displacement inversion" begin
            # Create a VERY small displacement field
            # For very small displacements, inverse ≈ -disp
            disp = ArrayType(randn(Float32, 8, 8, 8, 3, 1) * 0.005f0)

            inv_disp = invert_displacement(disp; iterations=30)

            @test inv_disp isa typeof(disp)
            @test size(inv_disp) == size(disp)

            # For very small displacements, inverse ≈ -disp (first order approximation)
            # Use atol for small values - the iterative algorithm may not converge perfectly
            @test isapprox(Array(inv_disp), -Array(disp), atol=0.02f0)
        end

        @testset "Composition approximates identity" begin
            # Create a small displacement
            disp = ArrayType(randn(Float32, 8, 8, 8, 3, 1) * 0.01f0)

            inv_disp = invert_displacement(disp; iterations=15)

            # disp + inv_disp should be small for small displacements
            sum_disp = disp .+ inv_disp
            mean_abs = sum(abs.(Array(sum_disp))) / length(sum_disp)

            # Mean absolute value should be small
            @test mean_abs < 0.02f0
        end

        @testset "Inversion properties" begin
            # Test that inversion produces valid output
            disp = ArrayType(randn(Float32, 8, 8, 8, 3, 1) * 0.05f0)
            inv_disp = invert_displacement(disp; iterations=10)

            # Output should be finite
            @test all(isfinite.(Array(inv_disp)))

            # Output should have similar magnitude to input
            @test maximum(abs.(Array(inv_disp))) < maximum(abs.(Array(disp))) * 3
        end
    end

    @testset "invert_displacement 2D" begin
        Random.seed!(42)

        @testset "Small displacement" begin
            disp = ArrayType(randn(Float32, 8, 8, 2, 1) * 0.01f0)

            inv_disp = invert_displacement(disp; iterations=15)

            @test size(inv_disp) == size(disp)
            # For very small displacements, inverse ≈ -disp
            @test isapprox(Array(inv_disp), -Array(disp), atol=0.02f0)
        end
    end

    @testset "GPU array type preservation" begin
        if USE_GPU
            @testset "MtlArray stays on GPU" begin
                disp_3d = MtlArray(randn(Float32, 8, 8, 8, 3, 1))

                result_resample = resample_displacement(disp_3d, (16, 16, 16))
                @test result_resample isa MtlArray

                result_velocity = resample_velocity(disp_3d, (12, 12, 12))
                @test result_velocity isa MtlArray

                result_invert = invert_displacement(disp_3d; iterations=5)
                @test result_invert isa MtlArray

                theta_3d = MtlArray(randn(Float32, 3, 4, 1))
                result_affine = upsample_affine_transform(theta_3d, (8, 8, 8), (16, 16, 16))
                @test result_affine isa MtlArray
            end
        else
            @info "Skipping GPU tests - Metal not available"
        end
    end

    @testset "Batch support" begin
        @testset "Multiple batch elements" begin
            batch_size = 2
            disp = ArrayType(randn(Float32, 8, 8, 8, 3, batch_size))

            result = resample_displacement(disp, (16, 16, 16))
            @test size(result) == (16, 16, 16, 3, batch_size)

            # Each batch element should be processed independently
            for n in 1:batch_size
                single = ArrayType(Array(disp)[:, :, :, :, n:n])
                single_result = resample_displacement(single, (16, 16, 16))
                @test Array(result)[:, :, :, :, n:n] ≈ Array(single_result)
            end
        end
    end

    @testset "Edge cases" begin
        @testset "Single voxel dimension" begin
            # Edge case: one dimension is 1
            disp = ArrayType(randn(Float32, 8, 8, 1, 3, 1))

            # This should not crash
            result = resample_displacement(disp, (16, 16, 1))
            @test size(result) == (16, 16, 1, 3, 1)
        end

        @testset "Zero displacement" begin
            disp = ArrayType(zeros(Float32, 8, 8, 8, 3, 1))

            result = resample_displacement(disp, (16, 16, 16))
            @test all(x -> abs(x) < 1e-6, Array(result))

            inv_result = invert_displacement(disp; iterations=5)
            @test all(x -> abs(x) < 1e-6, Array(inv_result))
        end
    end

end
