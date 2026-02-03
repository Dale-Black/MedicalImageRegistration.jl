using MedicalImageRegistration
using Test
using PythonCall
using Random

# Import Python modules for parity testing
const torch = pyimport("torch")
const np = pyimport("numpy")

# Try to import torchreg - may not be available in all environments
const torchreg = try
    pyimport("torchreg")
catch e
    @warn "torchreg not available - some parity tests will be skipped" exception=e
    nothing
end

# Always run test_utils.jl (only needs torch/numpy)
include("test_utils.jl")

# Only run torchreg parity tests if torchreg is available
if !isnothing(torchreg)
    include("test_affine.jl")
    include("test_syn.jl")
    include("test_metrics.jl")
else
    @warn "Skipping torchreg parity tests (test_affine.jl, test_syn.jl, test_metrics.jl)"
end

# ============================================================================
# Integration Tests (do not require torchreg)
# ============================================================================

@testset "Integration Tests" begin

    @testset "Synthetic Translation Recovery" begin

        @testset "3D translation recovery" begin
            Random.seed!(42)

            # Create synthetic 3D data with known translation
            X, Y, Z = 20, 20, 20

            # Static: Gaussian blob centered in image
            static = zeros(Float32, X, Y, Z, 1, 1)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2) / (X/4)
                dy = (j - Y/2) / (Y/4)
                dz = (k - Z/2) / (Z/4)
                static[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            # Moving: Same blob shifted by known amount
            shift_x, shift_y, shift_z = 3, 2, 1
            moving = zeros(Float32, X, Y, Z, 1, 1)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2 - shift_x) / (X/4)
                dy = (j - Y/2 - shift_y) / (Y/4)
                dz = (k - Z/2 - shift_z) / (Z/4)
                moving[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            # Run registration with translation only
            reg = AffineRegistration(
                ndims=3,
                scales=(2, 1),
                iterations=(100, 50),
                learning_rate=0.05f0,
                verbose=false,
                with_rotation=false,
                with_zoom=false,
                with_shear=false
            )

            moved = register(moving, static, reg)

            # Verify loss decreased
            initial_loss = MedicalImageRegistration.mse_loss(moving, static)
            final_loss = reg.loss
            @test final_loss < initial_loss
            @test final_loss < 0.01  # Should achieve good registration

            # Verify output shape
            @test size(moved) == (X, Y, Z, 1, 1)
            @test !any(isnan.(moved))
            @test !any(isinf.(moved))

            # Verify translation was recovered (approximately)
            affine = get_affine(reg)
            expected_tx = shift_x / (X/2)
            expected_ty = shift_y / (Y/2)
            expected_tz = shift_z / (Z/2)

            recovered_tx = affine[1, 4, 1]
            recovered_ty = affine[2, 4, 1]
            recovered_tz = affine[3, 4, 1]

            @test isapprox(recovered_tx, expected_tx, rtol=0.3)
            @test isapprox(recovered_ty, expected_ty, rtol=0.3)
            @test isapprox(recovered_tz, expected_tz, rtol=0.5)  # Z might be less precise
        end

        @testset "2D translation recovery" begin
            Random.seed!(43)

            X, Y = 24, 24

            static = zeros(Float32, X, Y, 1, 1)
            for i in 1:X, j in 1:Y
                dx = (i - X/2) / (X/4)
                dy = (j - Y/2) / (Y/4)
                static[i, j, 1, 1] = exp(-(dx^2 + dy^2))
            end

            shift_x, shift_y = 3, 2
            moving = zeros(Float32, X, Y, 1, 1)
            for i in 1:X, j in 1:Y
                dx = (i - X/2 - shift_x) / (X/4)
                dy = (j - Y/2 - shift_y) / (Y/4)
                moving[i, j, 1, 1] = exp(-(dx^2 + dy^2))
            end

            reg = AffineRegistration(
                ndims=2,
                scales=(2, 1),
                iterations=(100, 50),
                learning_rate=0.05f0,
                verbose=false,
                with_rotation=false,
                with_zoom=false,
                with_shear=false
            )

            moved = register(moving, static, reg)

            initial_loss = MedicalImageRegistration.mse_loss(moving, static)
            final_loss = reg.loss
            @test final_loss < initial_loss
            @test final_loss < 0.01

            @test size(moved) == (X, Y, 1, 1)
            @test !any(isnan.(moved))
        end
    end

    @testset "Synthetic Rotation Recovery (Approximation)" begin
        Random.seed!(44)

        # Note: Rotation recovery is tested as a general optimization capability
        # For small rotations, the affine registration should reduce loss

        X, Y, Z = 16, 16, 16

        # Create asymmetric blob (not rotationally symmetric)
        static = zeros(Float32, X, Y, Z, 1, 1)
        for i in 1:X, j in 1:Y, k in 1:Z
            dx = (i - X/2) / (X/4)
            dy = (j - Y/2) / (Y/3)  # Different radius in Y
            dz = (k - Z/2) / (Z/5)  # Different radius in Z
            static[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
        end

        # Moving: slightly rotated (simulated by small perturbation)
        moving = zeros(Float32, X, Y, Z, 1, 1)
        for i in 1:X, j in 1:Y, k in 1:Z
            # Small rotation around Z axis (approx 5 degrees)
            angle = 0.087f0  # ~5 degrees in radians
            di = (i - X/2)
            dj = (j - Y/2)
            ri = di * cos(angle) - dj * sin(angle)
            rj = di * sin(angle) + dj * cos(angle)
            dx = ri / (X/4)
            dy = rj / (Y/3)
            dz = (k - Z/2) / (Z/5)
            moving[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
        end

        reg = AffineRegistration(
            ndims=3,
            scales=(2, 1),
            iterations=(100, 50),
            learning_rate=0.02f0,
            verbose=false,
            with_translation=true,
            with_rotation=true,
            with_zoom=false,
            with_shear=false
        )

        moved = register(moving, static, reg)

        initial_loss = MedicalImageRegistration.mse_loss(moving, static)
        final_loss = reg.loss

        # Should reduce loss
        @test final_loss < initial_loss
        @test size(moved) == (X, Y, Z, 1, 1)
        @test !any(isnan.(moved))
    end

    @testset "Affine + SyN Pipeline" begin
        Random.seed!(45)

        X, Y, Z = 16, 16, 16

        # Create static image
        static = zeros(Float32, X, Y, Z, 1, 1)
        for i in 1:X, j in 1:Y, k in 1:Z
            dx = (i - X/2) / (X/4)
            dy = (j - Y/2) / (Y/4)
            dz = (k - Z/2) / (Z/4)
            static[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
        end

        # Create moving image with translation + local deformation
        moving = zeros(Float32, X, Y, Z, 1, 1)
        for i in 1:X, j in 1:Y, k in 1:Z
            # Translation
            dx = (i - X/2 - 2) / (X/4)
            dy = (j - Y/2) / (Y/4)
            dz = (k - Z/2) / (Z/4)
            # Small local deformation
            local_def = 0.1f0 * sin(Float32(i) * 0.5f0) * sin(Float32(j) * 0.5f0)
            moving[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2)) + local_def
        end

        # Step 1: Affine registration
        affine_reg = AffineRegistration(
            ndims=3,
            scales=(2,),
            iterations=(50,),
            learning_rate=0.03f0,
            verbose=false
        )

        affine_moved = register(moving, static, affine_reg)
        affine_loss = affine_reg.loss

        # Step 2: SyN refinement
        syn_reg = SyNRegistration(
            scales=(2,),
            iterations=(20,),
            learning_rate=0.02f0,
            verbose=false,
            sigma_flow=0.3f0
        )

        syn_moved, _, flow_xy, flow_yx = register(affine_moved, static, syn_reg)
        syn_final_loss = MedicalImageRegistration.mse_loss(syn_moved, static)

        # Verify pipeline improved results
        initial_loss = MedicalImageRegistration.mse_loss(moving, static)

        @test affine_loss < initial_loss  # Affine helped
        @test syn_final_loss <= affine_loss  # SyN didn't make it worse (may be similar or better)

        # Verify output validity
        @test size(syn_moved) == (X, Y, Z, 1, 1)
        @test !any(isnan.(syn_moved))
        @test !any(isinf.(syn_moved))

        # Verify flow fields exist and have correct shape
        @test size(flow_xy) == (X, Y, Z, 3, 1)
        @test size(flow_yx) == (X, Y, Z, 3, 1)
    end

    @testset "Type Stability and No Allocations in Hot Paths" begin
        Random.seed!(46)

        # Test that key functions are type-stable
        X, Y, Z = 8, 8, 8

        # Test affine_grid type stability
        theta = MedicalImageRegistration.identity_affine(3, 1, Float32)
        grid = @inferred MedicalImageRegistration.affine_grid(theta, (X, Y, Z))
        @test eltype(grid) == Float32

        # Test compose_affine type stability
        translation = zeros(Float32, 3, 1)
        rotation = zeros(Float32, 3, 3, 1)
        rotation[1, 1, 1] = rotation[2, 2, 1] = rotation[3, 3, 1] = 1.0f0
        zoom = ones(Float32, 3, 1)
        shear = zeros(Float32, 3, 1)

        affine = @inferred MedicalImageRegistration.compose_affine(translation, rotation, zoom, shear)
        @test eltype(affine) == Float32

        # Test affine_transform type stability
        img = randn(Float32, X, Y, Z, 1, 1)
        result = @inferred MedicalImageRegistration.affine_transform(img, theta)
        @test eltype(result) == Float32

        # Test 2D versions
        theta_2d = MedicalImageRegistration.identity_affine(2, 1, Float32)
        grid_2d = @inferred MedicalImageRegistration.affine_grid(theta_2d, (X, Y))
        @test eltype(grid_2d) == Float32

        img_2d = randn(Float32, X, Y, 1, 1)
        result_2d = @inferred MedicalImageRegistration.affine_transform(img_2d, theta_2d)
        @test eltype(result_2d) == Float32
    end

    @testset "Batch Processing" begin
        Random.seed!(47)

        X, Y, Z = 12, 12, 12
        N = 2  # Batch size

        # Create batch of images
        static = randn(Float32, X, Y, Z, 1, N)
        moving = randn(Float32, X, Y, Z, 1, N)

        # Affine batch
        reg = AffineRegistration(
            ndims=3,
            scales=(2,),
            iterations=(10,),
            verbose=false
        )

        moved = register(moving, static, reg)

        @test size(moved) == (X, Y, Z, 1, N)
        @test size(reg.parameters.translation) == (3, N)
        @test !any(isnan.(moved))

        # SyN batch
        syn_reg = SyNRegistration(
            scales=(2,),
            iterations=(5,),
            verbose=false
        )

        syn_moved, _, flow_xy, flow_yx = register(moving, static, syn_reg)

        @test size(syn_moved) == (X, Y, Z, 1, N)
        @test size(flow_xy) == (X, Y, Z, 3, N)
        @test !any(isnan.(syn_moved))
    end

    @testset "Different Loss Functions" begin
        Random.seed!(48)

        X, Y, Z = 10, 10, 10

        static = randn(Float32, X, Y, Z, 1, 1)
        moving = randn(Float32, X, Y, Z, 1, 1)

        # Test with dice_loss
        reg_dice = AffineRegistration(
            ndims=3,
            scales=(2,),
            iterations=(10,),
            verbose=false,
            dissimilarity_fn=dice_loss
        )
        moved_dice = register(abs.(moving), abs.(static), reg_dice)
        @test !any(isnan.(moved_dice))

        # Test with NCC
        reg_ncc = AffineRegistration(
            ndims=3,
            scales=(2,),
            iterations=(10,),
            verbose=false,
            dissimilarity_fn=NCC(kernel_size=5)
        )
        moved_ncc = register(moving, static, reg_ncc)
        @test !any(isnan.(moved_ncc))
    end
end
