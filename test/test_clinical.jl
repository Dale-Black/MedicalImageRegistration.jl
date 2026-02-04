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

@testset "clinical.jl" begin

    @testset "ClinicalRegistrationResult construction" begin
        # Create synthetic test data
        T = Float32
        X, Y, Z, C, N = 8, 8, 8, 1, 1

        moving_data = ArrayType(randn(T, X, Y, Z, C, N))
        transform = ArrayType(randn(T, X, Y, Z, 3, N) * 0.1f0)
        metrics = Dict{Symbol, T}(:mi_before => T(0.5), :mi_after => T(0.8), :mi_improvement => T(0.3))
        metadata = Dict{Symbol, Any}(:test => "value")

        moved_image = PhysicalImage(moving_data; spacing=(1f0, 1f0, 1f0))

        result = MedicalImageRegistration.ClinicalRegistrationResult{T, 5, typeof(moving_data)}(
            moved_image,
            transform,
            nothing,
            metrics,
            metadata
        )

        @test result.moved_image === moved_image
        @test result.transform === transform
        @test result.inverse_transform === nothing
        @test result.metrics[:mi_before] ≈ T(0.5)
        @test result.metrics[:mi_after] ≈ T(0.8)
        @test result.metadata[:test] == "value"
    end

    @testset "register_clinical 3D - basic workflow" begin
        Random.seed!(42)

        T = Float32
        X, Y, Z, C, N = 16, 16, 16, 1, 1

        # Create synthetic images with different patterns
        # Static: smooth blob in center
        static_data = ArrayType(zeros(T, X, Y, Z, C, N))
        static_cpu = zeros(T, X, Y, Z, C, N)
        for k in 1:Z, j in 1:Y, i in 1:X
            dist = sqrt((i - X÷2)^2 + (j - Y÷2)^2 + (k - Z÷2)^2)
            static_cpu[i, j, k, 1, 1] = exp(-dist^2 / 50)
        end
        copyto!(static_data, static_cpu)

        # Moving: same blob but slightly shifted
        moving_data = ArrayType(zeros(T, X, Y, Z, C, N))
        moving_cpu = zeros(T, X, Y, Z, C, N)
        for k in 1:Z, j in 1:Y, i in 1:X
            # Shift by 2 voxels in x direction
            src_i = clamp(i - 2, 1, X)
            dist = sqrt((src_i - X÷2)^2 + (j - Y÷2)^2 + (k - Z÷2)^2)
            moving_cpu[i, j, k, 1, 1] = exp(-dist^2 / 50)
        end
        copyto!(moving_data, moving_cpu)

        # Create PhysicalImages with spacing
        moving = PhysicalImage(moving_data; spacing=(1f0, 1f0, 1f0))
        static = PhysicalImage(static_data; spacing=(1f0, 1f0, 1f0))

        # Run registration with minimal iterations for testing
        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mse_loss,  # Use MSE for same-modality test
            preserve_hu=false,  # Use bilinear for this test
            registration_type=:affine,
            affine_scales=(2, 1),
            affine_iterations=(10, 5),
            learning_rate=0.1f0,
            verbose=false
        )

        @test result isa MedicalImageRegistration.ClinicalRegistrationResult
        @test size(result.moved_image.data) == (X, Y, Z, C, N)
        @test size(result.transform) == (X, Y, Z, 3, N)
        @test haskey(result.metrics, :mi_before)
        @test haskey(result.metrics, :mi_after)
        @test haskey(result.metrics, :mi_improvement)
        @test haskey(result.metadata, :registration_resolution)
    end

    @testset "register_clinical - HU preservation" begin
        Random.seed!(42)

        T = Float32
        X, Y, Z, C, N = 12, 12, 12, 1, 1

        # Create images with distinct integer HU values (CT-like)
        # Static: three distinct regions (air=-1000, soft tissue=40, bone=1000)
        static_data = ArrayType(zeros(T, X, Y, Z, C, N))
        static_cpu = fill(T(40), X, Y, Z, C, N)  # Soft tissue background
        # Air pocket
        static_cpu[2:4, 2:4, 2:4, 1, 1] .= T(-1000)
        # Bone region
        static_cpu[8:10, 8:10, 8:10, 1, 1] .= T(1000)
        copyto!(static_data, static_cpu)

        # Moving: same structure, small shift
        moving_data = ArrayType(zeros(T, X, Y, Z, C, N))
        moving_cpu = fill(T(40), X, Y, Z, C, N)
        moving_cpu[3:5, 2:4, 2:4, 1, 1] .= T(-1000)  # Shifted by 1
        moving_cpu[9:11, 8:10, 8:10, 1, 1] .= T(1000)
        copyto!(moving_data, moving_cpu)

        moving = PhysicalImage(moving_data; spacing=(1f0, 1f0, 1f0))
        static = PhysicalImage(static_data; spacing=(1f0, 1f0, 1f0))

        # Get unique values before registration
        original_values = Set(vec(Array(moving.data)))

        # Register with HU preservation
        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mse_loss,
            preserve_hu=true,  # Use nearest-neighbor
            registration_type=:affine,
            affine_scales=(2,),
            affine_iterations=(5,),
            learning_rate=0.1f0,
            verbose=false
        )

        # Check that output values are a subset of input values
        output_values = Set(vec(Array(result.moved_image.data)))

        @test output_values ⊆ original_values
    end

    @testset "register_clinical - anisotropic spacing" begin
        Random.seed!(42)

        T = Float32
        X, Y, Z, C, N = 16, 16, 8, 1, 1  # Fewer slices in Z

        # Create images with anisotropic spacing (simulating clinical CT)
        # Moving: 0.5mm x 0.5mm x 3mm (contrast CT-like)
        # Static: 1mm x 1mm x 2mm

        moving_data = ArrayType(randn(T, X, Y, Z, C, N) * 0.5f0)
        static_data = ArrayType(randn(T, X, Y, Z, C, N) * 0.5f0)

        moving = PhysicalImage(moving_data; spacing=(0.5f0, 0.5f0, 3.0f0))
        static = PhysicalImage(static_data; spacing=(1.0f0, 1.0f0, 2.0f0))

        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mse_loss,
            preserve_hu=false,
            registration_type=:affine,
            affine_scales=(2,),
            affine_iterations=(3,),
            learning_rate=0.05f0,
            verbose=false
        )

        @test result isa MedicalImageRegistration.ClinicalRegistrationResult
        @test result.metadata[:moving_spacing] == (0.5f0, 0.5f0, 3.0f0)
        @test result.metadata[:static_spacing] == (1.0f0, 1.0f0, 2.0f0)
    end

    @testset "register_clinical - MI loss" begin
        Random.seed!(42)

        T = Float32
        X, Y, Z, C, N = 12, 12, 12, 1, 1

        # Create images with different intensity mappings (simulating contrast difference)
        # Static: Low intensity blob (40 HU non-contrast)
        static_data = ArrayType(zeros(T, X, Y, Z, C, N))
        static_cpu = zeros(T, X, Y, Z, C, N)
        for k in 1:Z, j in 1:Y, i in 1:X
            dist = sqrt((i - X÷2)^2 + (j - Y÷2)^2 + (k - Z÷2)^2)
            static_cpu[i, j, k, 1, 1] = dist < 4 ? T(40) : T(0)
        end
        copyto!(static_data, static_cpu)

        # Moving: High intensity blob (300 HU contrast) - same location
        moving_data = ArrayType(zeros(T, X, Y, Z, C, N))
        moving_cpu = zeros(T, X, Y, Z, C, N)
        for k in 1:Z, j in 1:Y, i in 1:X
            dist = sqrt((i - X÷2)^2 + (j - Y÷2)^2 + (k - Z÷2)^2)
            moving_cpu[i, j, k, 1, 1] = dist < 4 ? T(300) : T(0)
        end
        copyto!(moving_data, moving_cpu)

        moving = PhysicalImage(moving_data; spacing=(1f0, 1f0, 1f0))
        static = PhysicalImage(static_data; spacing=(1f0, 1f0, 1f0))

        # MI loss should work even though intensities are different
        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mi_loss,  # Use MI for different intensities
            preserve_hu=true,
            registration_type=:affine,
            affine_scales=(2,),
            affine_iterations=(5,),
            learning_rate=0.05f0,
            verbose=false
        )

        @test result isa MedicalImageRegistration.ClinicalRegistrationResult
        # MI should be computable even with intensity mismatch
        @test isfinite(result.metrics[:mi_before])
        @test isfinite(result.metrics[:mi_after])
    end

    @testset "transform_clinical" begin
        Random.seed!(42)

        T = Float32
        X, Y, Z, C, N = 12, 12, 12, 1, 1

        # Create simple test case
        moving_data = ArrayType(randn(T, X, Y, Z, C, N))
        static_data = ArrayType(randn(T, X, Y, Z, C, N))

        moving = PhysicalImage(moving_data; spacing=(1f0, 1f0, 1f0))
        static = PhysicalImage(static_data; spacing=(1f0, 1f0, 1f0))

        # Register
        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mse_loss,
            preserve_hu=false,
            registration_type=:affine,
            affine_scales=(2,),
            affine_iterations=(3,),
            learning_rate=0.05f0,
            verbose=false
        )

        # Create another image with same dimensions
        other_image = PhysicalImage(ArrayType(randn(T, X, Y, Z, C, N)); spacing=(1f0, 1f0, 1f0))

        # Apply transform
        transformed = transform_clinical(result, other_image; interpolation=:bilinear)

        @test size(transformed.data) == (X, Y, Z, C, N)
        @test transformed.spacing == static.spacing
    end

    @testset "transform_clinical - wrong size error" begin
        Random.seed!(42)

        T = Float32
        X, Y, Z, C, N = 12, 12, 12, 1, 1

        moving_data = ArrayType(randn(T, X, Y, Z, C, N))
        static_data = ArrayType(randn(T, X, Y, Z, C, N))

        moving = PhysicalImage(moving_data; spacing=(1f0, 1f0, 1f0))
        static = PhysicalImage(static_data; spacing=(1f0, 1f0, 1f0))

        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mse_loss,
            preserve_hu=false,
            registration_type=:affine,
            affine_scales=(2,),
            affine_iterations=(2,),
            learning_rate=0.05f0,
            verbose=false
        )

        # Create image with different dimensions
        wrong_size = PhysicalImage(ArrayType(randn(T, X+1, Y, Z, C, N)); spacing=(1f0, 1f0, 1f0))

        @test_throws ErrorException transform_clinical(result, wrong_size)
    end

    @testset "register_clinical 2D" begin
        Random.seed!(42)

        T = Float32
        X, Y, C, N = 16, 16, 1, 1

        # Create 2D test images
        moving_data = ArrayType(randn(T, X, Y, C, N))
        static_data = ArrayType(randn(T, X, Y, C, N))

        moving = PhysicalImage(moving_data; spacing=(1f0, 1f0))
        static = PhysicalImage(static_data; spacing=(1f0, 1f0))

        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mse_loss,
            preserve_hu=false,
            registration_type=:affine,
            affine_scales=(2,),
            affine_iterations=(3,),
            learning_rate=0.05f0,
            verbose=false
        )

        @test result isa MedicalImageRegistration.ClinicalRegistrationResult{T, 4}
        @test size(result.moved_image.data) == (X, Y, C, N)
        @test size(result.transform) == (X, Y, 2, N)
    end

    @testset "GPU array type preservation" begin
        if USE_GPU
            @testset "MtlArray stays on GPU" begin
                Random.seed!(42)

                T = Float32
                X, Y, Z, C, N = 12, 12, 12, 1, 1

                moving_data = MtlArray(randn(T, X, Y, Z, C, N))
                static_data = MtlArray(randn(T, X, Y, Z, C, N))

                moving = PhysicalImage(moving_data; spacing=(1f0, 1f0, 1f0))
                static = PhysicalImage(static_data; spacing=(1f0, 1f0, 1f0))

                result = register_clinical(
                    moving, static;
                    registration_resolution=2f0,
                    loss_fn=mse_loss,
                    preserve_hu=false,
                    registration_type=:affine,
                    affine_scales=(2,),
                    affine_iterations=(2,),
                    learning_rate=0.05f0,
                    verbose=false
                )

                @test result.moved_image.data isa MtlArray
                @test result.transform isa MtlArray
            end
        else
            @info "Skipping GPU tests - Metal not available"
        end
    end

    @testset "Metadata completeness" begin
        Random.seed!(42)

        T = Float32
        X, Y, Z, C, N = 10, 10, 10, 1, 1

        moving_data = ArrayType(randn(T, X, Y, Z, C, N))
        static_data = ArrayType(randn(T, X, Y, Z, C, N))

        moving = PhysicalImage(moving_data; spacing=(0.5f0, 0.5f0, 2.0f0))
        static = PhysicalImage(static_data; spacing=(1.0f0, 1.0f0, 1.0f0))

        result = register_clinical(
            moving, static;
            registration_resolution=2f0,
            loss_fn=mse_loss,
            preserve_hu=true,
            registration_type=:affine,
            affine_scales=(2,),
            affine_iterations=(2,),
            verbose=false
        )

        @test result.metadata[:moving_spacing] == (0.5f0, 0.5f0, 2.0f0)
        @test result.metadata[:static_spacing] == (1.0f0, 1.0f0, 1.0f0)
        @test result.metadata[:moving_size] == (X, Y, Z)
        @test result.metadata[:static_size] == (X, Y, Z)
        @test result.metadata[:registration_resolution] == 2f0
        @test result.metadata[:registration_type] == :affine
        @test result.metadata[:preserve_hu] == true
    end

end
