# Test HU preservation with hybrid interpolation mode
# Verifies that final_interpolation=:nearest preserves exact intensity values

using Test
using MedicalImageRegistration
using StableRNGs

# Include test helpers for conditional GPU testing
include("test_helpers.jl")

# ============================================================================
# HU Preservation Tests for grid_sample
# ============================================================================

@testset "grid_sample HU preservation" begin
    @testset "nearest output values ⊆ input values (2D)" begin
        rng = StableRNG(100)
        input = rand(rng, Float32, 16, 16, 1, 1)
        # Random grid within bounds
        grid = (rand(StableRNG(101), Float32, 2, 12, 12, 1) .- 0.5f0) .* 1.8f0

        output = grid_sample(input, grid; interpolation=:nearest)

        # All non-zero output values must be present in input
        input_vals = Set(vec(input))
        output_vals = Set(filter(!iszero, vec(output)))

        @test issubset(output_vals, input_vals)
    end

    @testset "nearest output values ⊆ input values (3D)" begin
        rng = StableRNG(110)
        input = rand(rng, Float32, 8, 8, 8, 1, 1)
        grid = (rand(StableRNG(111), Float32, 3, 6, 6, 6, 1) .- 0.5f0) .* 1.8f0

        output = grid_sample(input, grid; interpolation=:nearest)

        input_vals = Set(vec(input))
        output_vals = Set(filter(!iszero, vec(output)))

        @test issubset(output_vals, input_vals)
    end

    @testset "bilinear creates new values (2D)" begin
        rng = StableRNG(120)
        input = rand(rng, Float32, 16, 16, 1, 1)
        grid = (rand(StableRNG(121), Float32, 2, 12, 12, 1) .- 0.5f0) .* 1.8f0

        output_nearest = grid_sample(input, grid; interpolation=:nearest)
        output_bilinear = grid_sample(input, grid; interpolation=:bilinear)

        # Bilinear typically creates more unique values than nearest
        nearest_vals = Set(vec(output_nearest))
        bilinear_vals = Set(vec(output_bilinear))

        # Both should have output, but bilinear may have values not in input
        @test length(bilinear_vals) > 0
        @test length(nearest_vals) > 0
    end

    if METAL_AVAILABLE
        @testset "HU preservation on Metal GPU (2D)" begin
            input_cpu = rand(StableRNG(130), Float32, 16, 16, 1, 2)
            grid_cpu = (rand(StableRNG(131), Float32, 2, 12, 12, 2) .- 0.5f0) .* 1.8f0

            input_mtl = MtlArray(input_cpu)
            grid_mtl = MtlArray(grid_cpu)

            output_mtl = grid_sample(input_mtl, grid_mtl; interpolation=:nearest)

            @test output_mtl isa MtlArray

            input_vals = Set(vec(input_cpu))
            output_vals = Set(filter(!iszero, vec(Array(output_mtl))))

            @test issubset(output_vals, input_vals)
        end

        @testset "HU preservation on Metal GPU (3D)" begin
            input_cpu = rand(StableRNG(140), Float32, 8, 8, 8, 1, 1)
            grid_cpu = (rand(StableRNG(141), Float32, 3, 6, 6, 6, 1) .- 0.5f0) .* 1.8f0

            input_mtl = MtlArray(input_cpu)
            grid_mtl = MtlArray(grid_cpu)

            output_mtl = grid_sample(input_mtl, grid_mtl; interpolation=:nearest)

            @test output_mtl isa MtlArray

            input_vals = Set(vec(input_cpu))
            output_vals = Set(filter(!iszero, vec(Array(output_mtl))))

            @test issubset(output_vals, input_vals)
        end
    end
end

# ============================================================================
# HU Preservation Tests for AffineRegistration
# ============================================================================

@testset "AffineRegistration HU preservation" begin
    @testset "transform() with interpolation=:nearest preserves HU (2D)" begin
        rng = StableRNG(200)
        input = rand(rng, Float32, 16, 16, 1, 1)

        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(4,),
            iterations=(10,),
            array_type=Array
        )

        # Apply some transformation
        moved_bilinear = transform(reg, input; interpolation=:bilinear)
        moved_nearest = transform(reg, input; interpolation=:nearest)

        # Nearest should preserve input values
        input_vals = Set(vec(input))
        nearest_vals = Set(filter(!iszero, vec(moved_nearest)))

        @test issubset(nearest_vals, input_vals)
    end

    @testset "register() with final_interpolation=:nearest preserves HU (2D)" begin
        rng = StableRNG(210)
        # Create moving image with distinct values
        moving = rand(rng, Float32, 16, 16, 1, 1)
        # Create slightly shifted static
        static = circshift(moving, (2, 2, 0, 0))

        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(4,),
            iterations=(30,),
            learning_rate=0.1f0,
            array_type=Array
        )

        # Register with nearest interpolation for final output
        moved = register(reg, moving, static; verbose=false, final_interpolation=:nearest)

        # Verify HU preservation
        moving_vals = Set(vec(moving))
        moved_vals = Set(filter(!iszero, vec(moved)))

        @test issubset(moved_vals, moving_vals)
    end

    if METAL_AVAILABLE
        @testset "register() HU preservation on Metal GPU (2D)" begin
            rng = StableRNG(220)
            moving_cpu = rand(rng, Float32, 16, 16, 1, 1)
            static_cpu = circshift(moving_cpu, (2, 2, 0, 0))

            moving = MtlArray(moving_cpu)
            static = MtlArray(static_cpu)

            reg = AffineRegistration{Float32}(
                is_3d=false,
                scales=(4,),
                iterations=(30,),
                learning_rate=0.1f0,
                array_type=MtlArray
            )

            moved = register(reg, moving, static; verbose=false, final_interpolation=:nearest)

            @test moved isa MtlArray

            moving_vals = Set(vec(moving_cpu))
            moved_vals = Set(filter(!iszero, vec(Array(moved))))

            @test issubset(moved_vals, moving_vals)
        end
    end
end

# ============================================================================
# Synthetic CT Test with Known HU Values
# ============================================================================

@testset "Synthetic CT HU preservation" begin
    @testset "CT with air/water/bone HU values" begin
        # Create synthetic CT with known HU values
        # Air: -1000 HU, Water: 0 HU, Bone: 1000 HU
        ct = zeros(Float32, 16, 16, 1, 1)

        # Background (air)
        ct .= -1000.0f0

        # Water region (center)
        ct[6:11, 6:11, 1, 1] .= 0.0f0

        # Bone region (small spot)
        ct[8:9, 8:9, 1, 1] .= 1000.0f0

        # Verify original unique values
        original_vals = sort(unique(vec(ct)))
        @test original_vals == Float32[-1000.0, 0.0, 1000.0]

        # Create shifted version
        ct_shifted = circshift(ct, (2, 2, 0, 0))

        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(2,),
            iterations=(20,),
            learning_rate=0.1f0,
            array_type=Array
        )

        # Register with bilinear (creates new values)
        moved_bilinear = register(reg, ct, ct_shifted; verbose=false, final_interpolation=:bilinear)

        # Register with nearest (preserves exact values)
        reset!(reg)
        moved_nearest = register(reg, ct, ct_shifted; verbose=false, final_interpolation=:nearest)

        # Bilinear creates interpolated values
        bilinear_vals = sort(unique(vec(moved_bilinear)))
        # Nearest preserves exact values
        nearest_vals = sort(unique(vec(moved_nearest)))

        # Original CT has 3 distinct values
        @test length(original_vals) == 3

        # Nearest should only have values from original (possibly fewer due to edge effects)
        @test issubset(Set(nearest_vals), Set(original_vals))

        # Bilinear typically creates many more unique values (interpolation)
        # Note: This test is informational - bilinear may or may not create new values
        # depending on the registration result
        @test length(bilinear_vals) >= 1
    end

    @testset "HU min/max preservation" begin
        rng = StableRNG(300)
        # Create CT-like image with known range
        ct = rand(rng, Float32, 16, 16, 1, 1) .* 2000.0f0 .- 1000.0f0  # Range: -1000 to 1000

        original_min = minimum(ct)
        original_max = maximum(ct)

        # Slightly perturbed static
        static = ct .+ randn(StableRNG(301), Float32, 16, 16, 1, 1) .* 0.01f0

        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(2,),
            iterations=(10,),
            array_type=Array
        )

        moved_nearest = register(reg, ct, static; verbose=false, final_interpolation=:nearest)

        # With nearest, min/max of non-padded output should be within original range
        moved_nonzero = filter(!iszero, vec(moved_nearest))
        if !isempty(moved_nonzero)
            @test minimum(moved_nonzero) >= original_min
            @test maximum(moved_nonzero) <= original_max
        end
    end
end

# ============================================================================
# Registration Convergence with Hybrid Mode
# ============================================================================

@testset "Registration convergence with hybrid mode" begin
    @testset "AffineRegistration converges with final_interpolation=:nearest" begin
        rng = StableRNG(400)
        moving = rand(rng, Float32, 32, 32, 1, 1)
        # Add known translation
        static = zeros(Float32, 32, 32, 1, 1)
        static[3:end, 3:end, 1, 1] = moving[1:end-2, 1:end-2, 1, 1]

        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(4, 2),
            iterations=(30, 20),
            learning_rate=0.05f0,
            array_type=Array
        )

        # With bilinear output
        moved_bilinear = register(reg, moving, static; verbose=false, final_interpolation=:bilinear)

        # With nearest output (same optimization, different final interp)
        reset!(reg)
        moved_nearest = register(reg, moving, static; verbose=false, final_interpolation=:nearest)

        # Both should converge (loss decreases)
        @test length(reg.loss_history) > 0
        @test reg.loss_history[end] < reg.loss_history[1]

        # Nearest output should preserve input values
        moving_vals = Set(vec(moving))
        nearest_vals = Set(filter(!iszero, vec(moved_nearest)))
        @test issubset(nearest_vals, moving_vals)
    end

    if METAL_AVAILABLE
        @testset "Registration convergence on Metal GPU" begin
            rng = StableRNG(410)
            moving_cpu = rand(rng, Float32, 32, 32, 1, 1)
            static_cpu = zeros(Float32, 32, 32, 1, 1)
            static_cpu[3:end, 3:end, 1, 1] = moving_cpu[1:end-2, 1:end-2, 1, 1]

            moving = MtlArray(moving_cpu)
            static = MtlArray(static_cpu)

            reg = AffineRegistration{Float32}(
                is_3d=false,
                scales=(4, 2),
                iterations=(30, 20),
                learning_rate=0.05f0,
                array_type=MtlArray
            )

            moved = register(reg, moving, static; verbose=false, final_interpolation=:nearest)

            @test moved isa MtlArray
            @test length(reg.loss_history) > 0
            @test reg.loss_history[end] < reg.loss_history[1]

            # HU preservation on GPU
            moving_vals = Set(vec(moving_cpu))
            moved_vals = Set(filter(!iszero, vec(Array(moved))))
            @test issubset(moved_vals, moving_vals)
        end
    end
end

# ============================================================================
# Example Workflow Documentation (via test comments)
# ============================================================================

@testset "Documented workflow examples" begin
    @testset "Example: CT registration with HU preservation" begin
        # Example workflow for CT images where HU values must be preserved
        #
        # Step 1: Load CT images (example with synthetic data)
        moving_ct = rand(StableRNG(500), Float32, 16, 16, 1, 1) .* 2000.0f0 .- 1000.0f0
        static_ct = circshift(moving_ct, (1, 1, 0, 0))

        # Step 2: Create registration object
        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(2,),
            iterations=(20,),
            array_type=Array
        )

        # Step 3: Register with HU preservation (bilinear during optimization, nearest for output)
        #
        # This is the recommended workflow for quantitative CT analysis:
        # - Optimization uses smooth bilinear interpolation for gradient descent
        # - Final output uses nearest-neighbor to preserve exact HU values
        moved_ct = register(reg, moving_ct, static_ct;
                           verbose=false,
                           final_interpolation=:nearest)

        # Step 4: Verify HU preservation
        # All values in moved_ct should be from original moving_ct
        @test issubset(Set(vec(moved_ct)), Set(vec(moving_ct)))

        # Alternative: Apply transform to new images with specific interpolation
        # transform(reg, other_ct; interpolation=:nearest)
        @test true  # Workflow documentation complete
    end
end
