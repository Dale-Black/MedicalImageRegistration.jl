# Affine registration tests
# Tests compose_affine, affine_transform, AffineRegistration, and full registration

using Test
using MedicalImageRegistration
using Random
using LinearAlgebra
using Statistics

# Try to import Metal, fall back to CPU if not available
const USE_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

if USE_GPU
    println("Testing with Metal GPU (MtlArray)")
    const ArrayType = MtlArray
else
    println("Testing with CPU arrays")
    const ArrayType = Array
end

# ============================================================================
# GPU AffineRegistration Tests (No Python dependencies)
# ============================================================================

@testset "AffineRegistration GPU Tests" begin

    @testset "Constructor and Reset" begin
        # Test 2D constructor
        reg2d = AffineRegistration{Float32}(
            is_3d=false,
            scales=(4, 2),
            iterations=(10, 5),
            batch_size=1,
            array_type=ArrayType
        )
        @test reg2d.is_3d == false
        @test size(reg2d.translation) == (2, 1)
        @test size(reg2d.rotation) == (2, 2, 1)
        @test size(reg2d.zoom) == (2, 1)
        @test size(reg2d.shear) == (2, 1)

        # Check initial values
        @test all(Array(reg2d.translation) .== 0)
        @test Array(reg2d.rotation[:, :, 1]) ≈ [1.0 0.0; 0.0 1.0]
        @test all(Array(reg2d.zoom) .== 1)
        @test all(Array(reg2d.shear) .== 0)

        # Test 3D constructor
        reg3d = AffineRegistration{Float32}(
            is_3d=true,
            scales=(2,),
            iterations=(5,),
            batch_size=1,
            array_type=ArrayType
        )
        @test reg3d.is_3d == true
        @test size(reg3d.translation) == (3, 1)
        @test size(reg3d.rotation) == (3, 3, 1)
        @test size(reg3d.zoom) == (3, 1)
        @test size(reg3d.shear) == (3, 1)

        # Test reset
        reg3d.translation .= 1.0f0
        reset!(reg3d)
        @test all(Array(reg3d.translation) .== 0)
    end

    @testset "get_affine" begin
        reg = AffineRegistration{Float32}(
            is_3d=false,
            batch_size=1,
            array_type=ArrayType
        )

        theta = get_affine(reg)
        @test size(theta) == (2, 3, 1)

        # Identity transformation
        theta_cpu = Array(theta)
        @test theta_cpu[1, 1, 1] ≈ 1.0  # scale x
        @test theta_cpu[2, 2, 1] ≈ 1.0  # scale y
        @test theta_cpu[1, 2, 1] ≈ 0.0  # no shear/rotation
        @test theta_cpu[2, 1, 1] ≈ 0.0  # no shear/rotation
        @test theta_cpu[1, 3, 1] ≈ 0.0  # no translation x
        @test theta_cpu[2, 3, 1] ≈ 0.0  # no translation y
    end

    @testset "affine_transform 2D" begin
        # Create simple test image
        img_cpu = zeros(Float32, 16, 16, 1, 1)
        img_cpu[4:12, 4:12, 1, 1] .= 1.0f0  # White square in center
        img = ArrayType(img_cpu)

        # Identity transform
        theta_cpu = Float32[1 0 0; 0 1 0]
        theta = ArrayType(reshape(theta_cpu, 2, 3, 1))

        out = affine_transform(img, theta)
        @test size(out) == size(img)
        @test Array(out) ≈ Array(img) atol=1e-5
    end

    @testset "affine_transform 3D" begin
        # Create simple test volume
        vol_cpu = zeros(Float32, 8, 8, 8, 1, 1)
        vol_cpu[2:6, 2:6, 2:6, 1, 1] .= 1.0f0  # White cube in center
        vol = ArrayType(vol_cpu)

        # Identity transform
        theta_cpu = Float32[1 0 0 0; 0 1 0 0; 0 0 1 0]
        theta = ArrayType(reshape(theta_cpu, 3, 4, 1))

        out = affine_transform(vol, theta)
        @test size(out) == size(vol)
        @test Array(out) ≈ Array(vol) atol=1e-5
    end

    @testset "fit! 2D registration" begin
        # Create moving and static images with known transformation
        # Static: square in center
        static_cpu = zeros(Float32, 32, 32, 1, 1)
        static_cpu[10:22, 10:22, 1, 1] .= 1.0f0

        # Moving: same square but slightly shifted
        moving_cpu = zeros(Float32, 32, 32, 1, 1)
        moving_cpu[12:24, 12:24, 1, 1] .= 1.0f0  # Shifted by 2 pixels

        static = ArrayType(static_cpu)
        moving = ArrayType(moving_cpu)

        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(2,),
            iterations=(50,),  # Few iterations for test
            learning_rate=0.05f0,
            with_translation=true,
            with_rotation=false,
            with_zoom=false,
            with_shear=false,
            batch_size=1,
            array_type=ArrayType
        )

        fit!(reg, moving, static; verbose=false)

        # Check that loss decreased
        @test length(reg.loss_history) == 50
        @test reg.loss_history[end] < reg.loss_history[1]  # Loss should decrease

        # Check that translation was learned (should be negative to shift moving back)
        t = Array(reg.translation)
        # The translation should be non-zero and in the right direction
        @test abs(t[1, 1]) > 0.001 || abs(t[2, 1]) > 0.001
    end

    @testset "fit! 3D registration" begin
        # Create moving and static volumes
        static_cpu = zeros(Float32, 16, 16, 16, 1, 1)
        static_cpu[4:12, 4:12, 4:12, 1, 1] .= 1.0f0

        # Moving: same cube but slightly shifted
        moving_cpu = zeros(Float32, 16, 16, 16, 1, 1)
        moving_cpu[5:13, 5:13, 5:13, 1, 1] .= 1.0f0  # Shifted by 1 pixel

        static = ArrayType(static_cpu)
        moving = ArrayType(moving_cpu)

        reg = AffineRegistration{Float32}(
            is_3d=true,
            scales=(2,),
            iterations=(30,),  # Few iterations for test
            learning_rate=0.05f0,
            with_translation=true,
            with_rotation=false,
            with_zoom=false,
            with_shear=false,
            batch_size=1,
            array_type=ArrayType
        )

        fit!(reg, moving, static; verbose=false)

        # Check that loss decreased
        @test length(reg.loss_history) == 30
        @test reg.loss_history[end] < reg.loss_history[1]
    end

    @testset "register convenience function 2D" begin
        # Create simple images
        static_cpu = zeros(Float32, 32, 32, 1, 1)
        static_cpu[8:24, 8:24, 1, 1] .= 1.0f0

        moving_cpu = zeros(Float32, 32, 32, 1, 1)
        moving_cpu[10:26, 10:26, 1, 1] .= 1.0f0

        static = ArrayType(static_cpu)
        moving = ArrayType(moving_cpu)

        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(2,),
            iterations=(30,),
            learning_rate=0.05f0,
            with_translation=true,
            with_rotation=false,
            with_zoom=false,
            with_shear=false,
            batch_size=1,
            array_type=ArrayType
        )

        moved = register(reg, moving, static; verbose=false)

        @test size(moved) == size(static)
        @test moved isa typeof(static)

        # Moved should be more similar to static than moving was
        initial_mse = mean((moving_cpu .- static_cpu).^2)
        final_mse = mean((Array(moved) .- static_cpu).^2)
        @test final_mse < initial_mse
    end

    @testset "transform function" begin
        reg = AffineRegistration{Float32}(
            is_3d=false,
            batch_size=1,
            array_type=ArrayType
        )

        # Create test image
        img = ArrayType(rand(Float32, 32, 32, 1, 1))

        # Transform with identity (no fit yet)
        out = transform(reg, img)
        @test size(out) == size(img)
        @test Array(out) ≈ Array(img) atol=1e-5

        # Transform to different size
        out_small = transform(reg, img, (16, 16))
        @test size(out_small) == (16, 16, 1, 1)
    end

    @testset "Multi-resolution pyramid 2D" begin
        static_cpu = zeros(Float32, 64, 64, 1, 1)
        static_cpu[16:48, 16:48, 1, 1] .= 1.0f0

        moving_cpu = zeros(Float32, 64, 64, 1, 1)
        moving_cpu[20:52, 20:52, 1, 1] .= 1.0f0

        static = ArrayType(static_cpu)
        moving = ArrayType(moving_cpu)

        # Test multi-scale
        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(4, 2),
            iterations=(20, 10),  # 20 at scale 4, 10 at scale 2
            learning_rate=0.05f0,
            with_translation=true,
            batch_size=1,
            array_type=ArrayType
        )

        fit!(reg, moving, static; verbose=false)

        # Should have total of 30 iterations
        @test length(reg.loss_history) == 30
        # Loss should generally decrease
        @test reg.loss_history[end] < reg.loss_history[1]
    end

    @testset "Parameters enabled/disabled" begin
        static_cpu = zeros(Float32, 32, 32, 1, 1)
        static_cpu[8:24, 8:24, 1, 1] .= 1.0f0

        moving_cpu = zeros(Float32, 32, 32, 1, 1)
        moving_cpu[10:26, 10:26, 1, 1] .= 1.0f0

        static = ArrayType(static_cpu)
        moving = ArrayType(moving_cpu)

        # Only zoom enabled
        reg = AffineRegistration{Float32}(
            is_3d=false,
            scales=(2,),
            iterations=(20,),
            learning_rate=0.01f0,
            with_translation=false,
            with_rotation=false,
            with_zoom=true,
            with_shear=false,
            batch_size=1,
            array_type=ArrayType
        )

        fit!(reg, moving, static; verbose=false)

        # Translation should remain zero
        @test all(abs.(Array(reg.translation)) .< 1e-6)
    end

end

# ============================================================================
# PyTorch Parity Tests (require PythonCall)
# ============================================================================

@testset "AffineRegistration PyTorch Parity" begin
    # Import torchreg.affine
    sys = pyimport("sys")
    sys.path.insert(0, "/Users/daleblack/Documents/dev/torchreg_temp")
    torchreg_affine = pyimport("torchreg.affine")
    F = pyimport("torch.nn.functional")

    # =========================================================================
    # compose_affine Parity Tests
    # =========================================================================
    @testset "compose_affine parity" begin

        @testset "3D identity parameters" begin
            # Julia parameters: translation (3, N), rotation (3, 3, N), zoom (3, N), shear (3, N)
            N = 1
            translation_julia = zeros(Float32, 3, N)
            rotation_julia = zeros(Float32, 3, 3, N)
            rotation_julia[1, 1, 1] = 1.0f0
            rotation_julia[2, 2, 1] = 1.0f0
            rotation_julia[3, 3, 1] = 1.0f0
            zoom_julia = ones(Float32, 3, N)
            shear_julia = zeros(Float32, 3, N)

            julia_affine = MedicalImageRegistration.compose_affine(
                translation_julia, rotation_julia, zoom_julia, shear_julia
            )

            # PyTorch parameters: translation (N, 3), rotation (N, 3, 3), zoom (N, 3), shear (N, 3)
            translation_torch = torch.zeros(N, 3)
            rotation_torch = torch.eye(3).unsqueeze(0)
            zoom_torch = torch.ones(N, 3)
            shear_torch = torch.zeros(N, 3)

            torch_affine = torchreg_affine.compose_affine(
                translation_torch, rotation_torch, zoom_torch, shear_torch
            )

            # Convert torch result to Julia: PyTorch (N, 3, 4) -> Julia (3, 4, N)
            torch_affine_np = pyconvert(Array{Float32}, torch_affine.detach().numpy())
            torch_affine_julia = permutedims(torch_affine_np, (2, 3, 1))

            @test size(julia_affine) == (3, 4, 1)
            @test isapprox(julia_affine, torch_affine_julia, rtol=1e-5)
        end

        @testset "2D identity parameters" begin
            N = 1
            translation_julia = zeros(Float32, 2, N)
            rotation_julia = zeros(Float32, 2, 2, N)
            rotation_julia[1, 1, 1] = 1.0f0
            rotation_julia[2, 2, 1] = 1.0f0
            zoom_julia = ones(Float32, 2, N)
            shear_julia = zeros(Float32, 2, N)

            julia_affine = MedicalImageRegistration.compose_affine(
                translation_julia, rotation_julia, zoom_julia, shear_julia
            )

            # PyTorch: is_3d=false -> n_dim=2
            translation_torch = torch.zeros(N, 2)
            rotation_torch = torch.eye(2).unsqueeze(0)
            zoom_torch = torch.ones(N, 2)
            shear_torch = torch.zeros(N, 2)

            torch_affine = torchreg_affine.compose_affine(
                translation_torch, rotation_torch, zoom_torch, shear_torch
            )

            torch_affine_np = pyconvert(Array{Float32}, torch_affine.detach().numpy())
            torch_affine_julia = permutedims(torch_affine_np, (2, 3, 1))

            @test size(julia_affine) == (2, 3, 1)
            @test isapprox(julia_affine, torch_affine_julia, rtol=1e-5)
        end

        @testset "3D with translation" begin
            N = 1
            translation_julia = Float32[0.5; -0.3; 0.2;;]  # (3, 1)
            rotation_julia = zeros(Float32, 3, 3, N)
            rotation_julia[1, 1, 1] = 1.0f0
            rotation_julia[2, 2, 1] = 1.0f0
            rotation_julia[3, 3, 1] = 1.0f0
            zoom_julia = ones(Float32, 3, N)
            shear_julia = zeros(Float32, 3, N)

            julia_affine = MedicalImageRegistration.compose_affine(
                translation_julia, rotation_julia, zoom_julia, shear_julia
            )

            translation_torch = torch.tensor([[0.5, -0.3, 0.2]])
            rotation_torch = torch.eye(3).unsqueeze(0)
            zoom_torch = torch.ones(N, 3)
            shear_torch = torch.zeros(N, 3)

            torch_affine = torchreg_affine.compose_affine(
                translation_torch, rotation_torch, zoom_torch, shear_torch
            )

            torch_affine_np = pyconvert(Array{Float32}, torch_affine.detach().numpy())
            torch_affine_julia = permutedims(torch_affine_np, (2, 3, 1))

            @test isapprox(julia_affine, torch_affine_julia, rtol=1e-5)
            @test julia_affine[1, 4, 1] ≈ 0.5f0  # tx
            @test julia_affine[2, 4, 1] ≈ -0.3f0 # ty
            @test julia_affine[3, 4, 1] ≈ 0.2f0  # tz
        end

        @testset "3D with zoom" begin
            N = 1
            translation_julia = zeros(Float32, 3, N)
            rotation_julia = zeros(Float32, 3, 3, N)
            rotation_julia[1, 1, 1] = 1.0f0
            rotation_julia[2, 2, 1] = 1.0f0
            rotation_julia[3, 3, 1] = 1.0f0
            zoom_julia = Float32[1.5; 2.0; 0.5;;]
            shear_julia = zeros(Float32, 3, N)

            julia_affine = MedicalImageRegistration.compose_affine(
                translation_julia, rotation_julia, zoom_julia, shear_julia
            )

            translation_torch = torch.zeros(N, 3)
            rotation_torch = torch.eye(3).unsqueeze(0)
            zoom_torch = torch.tensor([[1.5, 2.0, 0.5]])
            shear_torch = torch.zeros(N, 3)

            torch_affine = torchreg_affine.compose_affine(
                translation_torch, rotation_torch, zoom_torch, shear_torch
            )

            torch_affine_np = pyconvert(Array{Float32}, torch_affine.detach().numpy())
            torch_affine_julia = permutedims(torch_affine_np, (2, 3, 1))

            @test isapprox(julia_affine, torch_affine_julia, rtol=1e-5)
            @test julia_affine[1, 1, 1] ≈ 1.5f0  # zoom_x
            @test julia_affine[2, 2, 1] ≈ 2.0f0  # zoom_y
            @test julia_affine[3, 3, 1] ≈ 0.5f0  # zoom_z
        end

        @testset "3D with shear" begin
            N = 1
            translation_julia = zeros(Float32, 3, N)
            rotation_julia = zeros(Float32, 3, 3, N)
            rotation_julia[1, 1, 1] = 1.0f0
            rotation_julia[2, 2, 1] = 1.0f0
            rotation_julia[3, 3, 1] = 1.0f0
            zoom_julia = ones(Float32, 3, N)
            shear_julia = Float32[0.1; 0.2; 0.15;;]  # shear_xy, shear_xz, shear_yz

            julia_affine = MedicalImageRegistration.compose_affine(
                translation_julia, rotation_julia, zoom_julia, shear_julia
            )

            translation_torch = torch.zeros(N, 3)
            rotation_torch = torch.eye(3).unsqueeze(0)
            zoom_torch = torch.ones(N, 3)
            shear_torch = torch.tensor([[0.1, 0.2, 0.15]])

            torch_affine = torchreg_affine.compose_affine(
                translation_torch, rotation_torch, zoom_torch, shear_torch
            )

            torch_affine_np = pyconvert(Array{Float32}, torch_affine.detach().numpy())
            torch_affine_julia = permutedims(torch_affine_np, (2, 3, 1))

            @test isapprox(julia_affine, torch_affine_julia, rtol=1e-5)
        end

        @testset "batch_size > 1 (N=2)" begin
            N = 2
            # Different parameters for each batch element
            translation_julia = Float32[0.1 0.5; -0.2 0.3; 0.3 -0.1]  # (3, 2)
            rotation_julia = zeros(Float32, 3, 3, N)
            rotation_julia[1, 1, 1] = 1.0f0; rotation_julia[2, 2, 1] = 1.0f0; rotation_julia[3, 3, 1] = 1.0f0
            rotation_julia[1, 1, 2] = 1.0f0; rotation_julia[2, 2, 2] = 1.0f0; rotation_julia[3, 3, 2] = 1.0f0
            zoom_julia = Float32[1.0 1.2; 1.0 0.8; 1.0 1.1]
            shear_julia = zeros(Float32, 3, N)

            julia_affine = MedicalImageRegistration.compose_affine(
                translation_julia, rotation_julia, zoom_julia, shear_julia
            )

            translation_torch = torch.tensor([[0.1, -0.2, 0.3], [0.5, 0.3, -0.1]])
            rotation_torch = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
            zoom_torch = torch.tensor([[1.0, 1.0, 1.0], [1.2, 0.8, 1.1]])
            shear_torch = torch.zeros(N, 3)

            torch_affine = torchreg_affine.compose_affine(
                translation_torch, rotation_torch, zoom_torch, shear_torch
            )

            torch_affine_np = pyconvert(Array{Float32}, torch_affine.detach().numpy())
            torch_affine_julia = permutedims(torch_affine_np, (2, 3, 1))

            @test size(julia_affine) == (3, 4, 2)
            @test isapprox(julia_affine, torch_affine_julia, rtol=1e-5)
        end

        @testset "random parameters" begin
            Random.seed!(42)
            np.random.seed(42)

            N = 1
            translation_julia = randn(Float32, 3, N) * 0.5f0
            zoom_julia = ones(Float32, 3, N) .+ randn(Float32, 3, N) * 0.1f0
            shear_julia = randn(Float32, 3, N) * 0.1f0
            rotation_julia = zeros(Float32, 3, 3, N)
            rotation_julia[:, :, 1] = Float32.(I(3))

            # Convert to numpy then torch for proper array handling
            translation_np = np.ascontiguousarray(permutedims(translation_julia, (2, 1)))
            zoom_np = np.ascontiguousarray(permutedims(zoom_julia, (2, 1)))
            shear_np = np.ascontiguousarray(permutedims(shear_julia, (2, 1)))
            translation_torch = torch.from_numpy(translation_np)
            zoom_torch = torch.from_numpy(zoom_np)
            shear_torch = torch.from_numpy(shear_np)
            rotation_torch = torch.eye(3).unsqueeze(0)

            julia_affine = MedicalImageRegistration.compose_affine(
                translation_julia, rotation_julia, zoom_julia, shear_julia
            )

            torch_affine = torchreg_affine.compose_affine(
                translation_torch, rotation_torch, zoom_torch, shear_torch
            )

            torch_affine_np = pyconvert(Array{Float32}, torch_affine.detach().numpy())
            torch_affine_julia = permutedims(torch_affine_np, (2, 3, 1))

            @test isapprox(julia_affine, torch_affine_julia, rtol=1e-5)
        end
    end

    # =========================================================================
    # affine_transform Parity Tests
    # =========================================================================
    @testset "affine_transform parity" begin

        @testset "3D identity transform" begin
            Random.seed!(123)

            # Create test image: Julia (X, Y, Z, C, N) = (8, 8, 8, 1, 1)
            X, Y, Z = 8, 8, 8
            img_julia = randn(Float32, X, Y, Z, 1, 1)

            # Identity affine: Julia (3, 4, 1)
            affine_julia = MedicalImageRegistration.identity_affine(3, 1, Float32)

            # Transform
            result_julia = MedicalImageRegistration.affine_transform(
                img_julia, affine_julia; padding_mode=:border
            )

            # PyTorch: (N, C, Z, Y, X)
            img_torch = torch.tensor(np.ascontiguousarray(permutedims(img_julia, (5, 4, 3, 2, 1))))

            # PyTorch affine: (N, 3, 4)
            affine_torch = torch.eye(4).__getitem__((pyimport("builtins").slice(nothing, 3), pyimport("builtins").slice(nothing))).unsqueeze(0)

            result_torch = torchreg_affine.affine_transform(
                img_torch, affine_torch, nothing, "trilinear", "border", true
            )

            # Convert torch result to Julia
            result_torch_np = pyconvert(Array{Float32}, result_torch.detach().numpy())
            result_torch_julia = permutedims(result_torch_np, (5, 4, 3, 2, 1))

            @test size(result_julia) == size(img_julia)
            @test isapprox(result_julia, result_torch_julia, rtol=1e-4)
        end

        @testset "2D identity transform" begin
            Random.seed!(456)

            X, Y = 10, 10
            img_julia = randn(Float32, X, Y, 1, 1)
            affine_julia = MedicalImageRegistration.identity_affine(2, 1, Float32)

            result_julia = MedicalImageRegistration.affine_transform(
                img_julia, affine_julia; padding_mode=:border
            )

            # PyTorch: (N, C, Y, X)
            img_torch = torch.tensor(np.ascontiguousarray(permutedims(img_julia, (4, 3, 2, 1))))
            affine_torch = torch.eye(3).__getitem__((pyimport("builtins").slice(nothing, 2), pyimport("builtins").slice(nothing))).unsqueeze(0)

            result_torch = torchreg_affine.affine_transform(
                img_torch, affine_torch, nothing, "bilinear", "border", true
            )

            result_torch_np = pyconvert(Array{Float32}, result_torch.detach().numpy())
            result_torch_julia = permutedims(result_torch_np, (4, 3, 2, 1))

            @test size(result_julia) == size(img_julia)
            @test isapprox(result_julia, result_torch_julia, rtol=1e-4)
        end

        @testset "3D translation transform" begin
            Random.seed!(789)

            X, Y, Z = 12, 12, 12
            img_julia = randn(Float32, X, Y, Z, 1, 1)

            # Create affine with translation
            affine_julia = MedicalImageRegistration.identity_affine(3, 1, Float32)
            affine_julia[1, 4, 1] = 0.25f0  # translate x

            result_julia = MedicalImageRegistration.affine_transform(
                img_julia, affine_julia; padding_mode=:border
            )

            # PyTorch
            img_torch = torch.tensor(np.ascontiguousarray(permutedims(img_julia, (5, 4, 3, 2, 1))))
            affine_torch_np = permutedims(affine_julia, (3, 1, 2))  # (1, 3, 4)
            affine_torch = torch.tensor(np.ascontiguousarray(affine_torch_np))

            result_torch = torchreg_affine.affine_transform(
                img_torch, affine_torch, nothing, "trilinear", "border", true
            )

            result_torch_np = pyconvert(Array{Float32}, result_torch.detach().numpy())
            result_torch_julia = permutedims(result_torch_np, (5, 4, 3, 2, 1))

            @test isapprox(result_julia, result_torch_julia, rtol=1e-4)
        end

        @testset "3D zoom transform" begin
            Random.seed!(101)

            X, Y, Z = 10, 10, 10
            img_julia = randn(Float32, X, Y, Z, 1, 1)

            # Create affine with zoom
            affine_julia = zeros(Float32, 3, 4, 1)
            affine_julia[1, 1, 1] = 1.2f0  # zoom x
            affine_julia[2, 2, 1] = 0.8f0  # zoom y
            affine_julia[3, 3, 1] = 1.0f0  # zoom z

            result_julia = MedicalImageRegistration.affine_transform(
                img_julia, affine_julia; padding_mode=:zeros
            )

            # PyTorch
            img_torch = torch.tensor(np.ascontiguousarray(permutedims(img_julia, (5, 4, 3, 2, 1))))
            affine_torch_np = permutedims(affine_julia, (3, 1, 2))
            affine_torch = torch.tensor(np.ascontiguousarray(affine_torch_np))

            result_torch = torchreg_affine.affine_transform(
                img_torch, affine_torch, nothing, "trilinear", "zeros", true
            )

            result_torch_np = pyconvert(Array{Float32}, result_torch.detach().numpy())
            result_torch_julia = permutedims(result_torch_np, (5, 4, 3, 2, 1))

            @test isapprox(result_julia, result_torch_julia, rtol=1e-4)
        end

        @testset "batch_size > 1 (N=2)" begin
            Random.seed!(202)

            X, Y, Z = 8, 8, 8
            N = 2
            img_julia = randn(Float32, X, Y, Z, 1, N)

            # Different transforms per batch element
            affine_julia = zeros(Float32, 3, 4, N)
            affine_julia[:, :, 1] = MedicalImageRegistration.identity_affine(3, 1, Float32)[:, :, 1]
            affine_julia[:, :, 2] = MedicalImageRegistration.identity_affine(3, 1, Float32)[:, :, 1]
            affine_julia[1, 4, 2] = 0.1f0  # translate second image

            result_julia = MedicalImageRegistration.affine_transform(
                img_julia, affine_julia; padding_mode=:border
            )

            # PyTorch
            img_torch = torch.tensor(np.ascontiguousarray(permutedims(img_julia, (5, 4, 3, 2, 1))))
            affine_torch_np = permutedims(affine_julia, (3, 1, 2))
            affine_torch = torch.tensor(np.ascontiguousarray(affine_torch_np))

            result_torch = torchreg_affine.affine_transform(
                img_torch, affine_torch, nothing, "trilinear", "border", true
            )

            result_torch_np = pyconvert(Array{Float32}, result_torch.detach().numpy())
            result_torch_julia = permutedims(result_torch_np, (5, 4, 3, 2, 1))

            @test size(result_julia) == (X, Y, Z, 1, N)
            @test isapprox(result_julia, result_torch_julia, rtol=1e-4)
        end

        @testset "output shape resizing" begin
            Random.seed!(303)

            X_in, Y_in, Z_in = 16, 16, 16
            X_out, Y_out, Z_out = 8, 8, 8

            img_julia = randn(Float32, X_in, Y_in, Z_in, 1, 1)
            affine_julia = MedicalImageRegistration.identity_affine(3, 1, Float32)

            result_julia = MedicalImageRegistration.affine_transform(
                img_julia, affine_julia; shape=(X_out, Y_out, Z_out), padding_mode=:border
            )

            # PyTorch
            img_torch = torch.tensor(np.ascontiguousarray(permutedims(img_julia, (5, 4, 3, 2, 1))))
            affine_torch = torch.eye(4).__getitem__((pyimport("builtins").slice(nothing, 3), pyimport("builtins").slice(nothing))).unsqueeze(0)

            result_torch = torchreg_affine.affine_transform(
                img_torch, affine_torch, pylist([Z_out, Y_out, X_out]), "trilinear", "border", true
            )

            result_torch_np = pyconvert(Array{Float32}, result_torch.detach().numpy())
            result_torch_julia = permutedims(result_torch_np, (5, 4, 3, 2, 1))

            @test size(result_julia) == (X_out, Y_out, Z_out, 1, 1)
            @test isapprox(result_julia, result_torch_julia, rtol=1e-4)
        end
    end

    # =========================================================================
    # Full Registration Tests
    # =========================================================================
    @testset "Registration convergence" begin

        @testset "3D synthetic translation recovery" begin
            Random.seed!(42)

            # Create synthetic data with known translation
            X, Y, Z = 16, 16, 16

            # Static image: Gaussian blob in center
            static = zeros(Float32, X, Y, Z, 1, 1)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2) / (X/4)
                dy = (j - Y/2) / (Y/4)
                dz = (k - Z/2) / (Z/4)
                static[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            # Moving image: same blob shifted by 2 voxels in x
            shift = 2
            moving = zeros(Float32, X, Y, Z, 1, 1)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2 - shift) / (X/4)
                dy = (j - Y/2) / (Y/4)
                dz = (k - Z/2) / (Z/4)
                moving[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            # Run registration
            reg = AffineRegistration(
                ndims=3,
                scales=(2,),
                iterations=(100,),
                learning_rate=0.05f0,
                verbose=false,
                with_rotation=false,
                with_zoom=false,
                with_shear=false
            )

            moved = register(moving, static, reg)

            # Check convergence
            initial_loss = MedicalImageRegistration.mse_loss(moving, static)
            final_loss = reg.loss

            @test final_loss < initial_loss
            @test final_loss < 0.01  # Should converge well

            # Check recovered translation
            affine = get_affine(reg)
            recovered_tx = affine[1, 4, 1]

            # Expected translation in normalized coordinates: shift/X * 2 (because range is -1 to 1)
            expected_tx = shift / (X/2)  # ≈ 0.25

            @test isapprox(recovered_tx, expected_tx, rtol=0.2)  # 20% tolerance
        end

        @testset "2D synthetic translation recovery" begin
            Random.seed!(43)

            X, Y = 16, 16

            # Create 2D Gaussian blobs
            static = zeros(Float32, X, Y, 1, 1)
            for i in 1:X, j in 1:Y
                dx = (i - X/2) / (X/4)
                dy = (j - Y/2) / (Y/4)
                static[i, j, 1, 1] = exp(-(dx^2 + dy^2))
            end

            shift = 2
            moving = zeros(Float32, X, Y, 1, 1)
            for i in 1:X, j in 1:Y
                dx = (i - X/2 - shift) / (X/4)
                dy = (j - Y/2) / (Y/4)
                moving[i, j, 1, 1] = exp(-(dx^2 + dy^2))
            end

            reg = AffineRegistration(
                ndims=2,
                scales=(2,),
                iterations=(100,),
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
        end

        @testset "3D zoom recovery" begin
            Random.seed!(44)

            X, Y, Z = 16, 16, 16

            # Static: Gaussian blob
            static = zeros(Float32, X, Y, Z, 1, 1)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2) / (X/4)
                dy = (j - Y/2) / (Y/4)
                dz = (k - Z/2) / (Z/4)
                static[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            # Moving: slightly smaller blob (zoom needed)
            scale = 0.8
            moving = zeros(Float32, X, Y, Z, 1, 1)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2) / (X/4) / scale
                dy = (j - Y/2) / (Y/4) / scale
                dz = (k - Z/2) / (Z/4) / scale
                moving[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            reg = AffineRegistration(
                ndims=3,
                scales=(2,),
                iterations=(150,),
                learning_rate=0.02f0,
                verbose=false,
                with_translation=false,
                with_rotation=false,
                with_zoom=true,
                with_shear=false
            )

            moved = register(moving, static, reg)

            initial_loss = MedicalImageRegistration.mse_loss(moving, static)
            final_loss = reg.loss

            @test final_loss < initial_loss
        end

        @testset "batch_size=2" begin
            Random.seed!(45)

            X, Y, Z = 12, 12, 12
            N = 2

            # Create batch of images
            static = randn(Float32, X, Y, Z, 1, N)
            moving = randn(Float32, X, Y, Z, 1, N)

            reg = AffineRegistration(
                ndims=3,
                scales=(2,),
                iterations=(20,),
                learning_rate=0.01f0,
                verbose=false
            )

            moved = register(moving, static, reg)

            @test size(moved) == (X, Y, Z, 1, N)
            @test !any(isnan.(moved))
            @test !any(isinf.(moved))

            # Verify registration ran for batch
            @test reg.parameters !== nothing
            @test size(reg.parameters.translation) == (3, N)
        end
    end

    # =========================================================================
    # API Tests
    # =========================================================================
    @testset "API" begin

        @testset "transform function" begin
            Random.seed!(50)

            X, Y, Z = 10, 10, 10
            static = randn(Float32, X, Y, Z, 1, 1)
            moving = randn(Float32, X, Y, Z, 1, 1)

            reg = AffineRegistration(
                ndims=3,
                scales=(2,),
                iterations=(10,),
                verbose=false
            )

            register(moving, static, reg)

            # Apply transform to another image
            other_img = randn(Float32, X, Y, Z, 1, 1)
            transformed = transform(other_img, reg)

            @test size(transformed) == size(other_img)
            @test !any(isnan.(transformed))
        end

        @testset "get_affine function" begin
            Random.seed!(51)

            X, Y, Z = 8, 8, 8
            static = randn(Float32, X, Y, Z, 1, 1)
            moving = randn(Float32, X, Y, Z, 1, 1)

            reg = AffineRegistration(
                ndims=3,
                scales=(2,),
                iterations=(5,),
                verbose=false
            )

            register(moving, static, reg)

            affine = get_affine(reg)

            @test size(affine) == (3, 4, 1)
            @test !any(isnan.(affine))
        end

        @testset "custom dissimilarity function" begin
            Random.seed!(52)

            X, Y, Z = 8, 8, 8

            # Binary masks for dice loss
            static = Float32.(rand(X, Y, Z, 1, 1) .> 0.5)
            moving = Float32.(rand(X, Y, Z, 1, 1) .> 0.5)

            reg = AffineRegistration(
                ndims=3,
                scales=(2,),
                iterations=(10,),
                verbose=false,
                dissimilarity_fn=dice_loss
            )

            moved = register(moving, static, reg)

            @test size(moved) == size(static)
            @test !any(isnan.(moved))
        end
    end
end
