# Test compose_affine on GPU (Metal) with Mooncake AD

using Test
using MedicalImageRegistration
using Metal
using StableRNGs
import Mooncake
import Mooncake: CoDual, NoFData, NoRData

# ============================================================================
# Basic Forward Pass Tests
# ============================================================================

@testset "compose_affine 2D forward" begin
    @testset "Identity transformation" begin
        translation = zeros(Float32, 2, 1)
        rotation = Float32.(reshape([1 0; 0 1], 2, 2, 1))
        zoom = ones(Float32, 2, 1)
        shear = zeros(Float32, 2, 1)

        theta = compose_affine(translation, rotation, zoom, shear)

        @test size(theta) == (2, 3, 1)
        @test isapprox(theta[:, :, 1], Float32[1 0 0; 0 1 0]; rtol=1e-5)
    end

    @testset "Translation only" begin
        translation = Float32[0.5f0; -0.3f0;;]
        rotation = Float32.(reshape([1 0; 0 1], 2, 2, 1))
        zoom = ones(Float32, 2, 1)
        shear = zeros(Float32, 2, 1)

        theta = compose_affine(translation, rotation, zoom, shear)

        @test isapprox(theta[:, :, 1], Float32[1 0 0.5; 0 1 -0.3]; rtol=1e-5)
    end

    @testset "Zoom only" begin
        translation = zeros(Float32, 2, 1)
        rotation = Float32.(reshape([1 0; 0 1], 2, 2, 1))
        zoom = Float32[2.0f0; 0.5f0;;]
        shear = zeros(Float32, 2, 1)

        theta = compose_affine(translation, rotation, zoom, shear)

        @test isapprox(theta[:, :, 1], Float32[2 0 0; 0 0.5 0]; rtol=1e-5)
    end

    @testset "Shear" begin
        translation = zeros(Float32, 2, 1)
        rotation = Float32.(reshape([1 0; 0 1], 2, 2, 1))
        zoom = ones(Float32, 2, 1)
        shear = Float32[0.5f0; 0.0f0;;]

        theta = compose_affine(translation, rotation, zoom, shear)

        @test isapprox(theta[:, :, 1], Float32[1 0.5 0; 0 1 0]; rtol=1e-5)
    end

    @testset "Metal GPU test" begin
        translation = MtlArray(rand(StableRNG(100), Float32, 2, 2))
        rotation = MtlArray(Float32.(cat(reshape([1 0; 0 1], 2, 2), reshape([1 0; 0 1], 2, 2), dims=3)))
        zoom = MtlArray(ones(Float32, 2, 2))
        shear = MtlArray(zeros(Float32, 2, 2))

        theta = compose_affine(translation, rotation, zoom, shear)

        @test theta isa MtlArray
        @test size(theta) == (2, 3, 2)
    end
end

@testset "compose_affine 3D forward" begin
    @testset "Identity transformation" begin
        translation = zeros(Float32, 3, 1)
        rotation = Float32.(reshape([1 0 0; 0 1 0; 0 0 1], 3, 3, 1))
        zoom = ones(Float32, 3, 1)
        shear = zeros(Float32, 3, 1)

        theta = compose_affine(translation, rotation, zoom, shear)

        @test size(theta) == (3, 4, 1)
        expected = Float32[1 0 0 0; 0 1 0 0; 0 0 1 0]
        @test isapprox(theta[:, :, 1], expected; rtol=1e-5)
    end

    @testset "Translation and zoom" begin
        translation = Float32[0.1f0; 0.2f0; 0.3f0;;]
        rotation = Float32.(reshape([1 0 0; 0 1 0; 0 0 1], 3, 3, 1))
        zoom = Float32[2.0f0; 1.5f0; 0.5f0;;]
        shear = zeros(Float32, 3, 1)

        theta = compose_affine(translation, rotation, zoom, shear)

        expected = Float32[2 0 0 0.1; 0 1.5 0 0.2; 0 0 0.5 0.3]
        @test isapprox(theta[:, :, 1], expected; rtol=1e-5)
    end

    @testset "3D shear" begin
        translation = zeros(Float32, 3, 1)
        rotation = Float32.(reshape([1 0 0; 0 1 0; 0 0 1], 3, 3, 1))
        zoom = ones(Float32, 3, 1)
        shear = Float32[0.1f0; 0.2f0; 0.3f0;;]  # sxy, sxz, syz

        theta = compose_affine(translation, rotation, zoom, shear)

        # S = [1 0.1 0.2; 0 1 0.3; 0 0 1]
        expected = Float32[1 0.1 0.2 0; 0 1 0.3 0; 0 0 1 0]
        @test isapprox(theta[:, :, 1], expected; rtol=1e-5)
    end

    @testset "Metal GPU test 3D" begin
        translation = MtlArray(rand(StableRNG(200), Float32, 3, 1))
        rotation = MtlArray(Float32.(reshape([1 0 0; 0 1 0; 0 0 1], 3, 3, 1)))
        zoom = MtlArray(ones(Float32, 3, 1))
        shear = MtlArray(zeros(Float32, 3, 1))

        theta = compose_affine(translation, rotation, zoom, shear)

        @test theta isa MtlArray
        @test size(theta) == (3, 4, 1)
    end
end

# ============================================================================
# Gradient Tests
# ============================================================================

@testset "compose_affine 2D gradients" begin
    @testset "Mooncake rrule!! on CPU" begin
        translation = zeros(Float32, 2, 1)
        rotation = Float32.(reshape([1 0; 0 1], 2, 2, 1))
        zoom = ones(Float32, 2, 1)
        shear = zeros(Float32, 2, 1)

        d_translation = zeros(Float32, 2, 1)
        d_rotation = zeros(Float32, 2, 2, 1)
        d_zoom = zeros(Float32, 2, 1)
        d_shear = zeros(Float32, 2, 1)

        translation_codual = CoDual(translation, d_translation)
        rotation_codual = CoDual(rotation, d_rotation)
        zoom_codual = CoDual(zoom, d_zoom)
        shear_codual = CoDual(shear, d_shear)
        fn_codual = CoDual(compose_affine, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, translation_codual, rotation_codual, zoom_codual, shear_codual)

        @test size(output_codual.x) == (2, 3, 1)

        output_codual.dx .= 1.0f0
        pullback(NoRData())

        # Translation gradient should be 1 for each element
        @test isapprox(d_translation, ones(Float32, 2, 1); rtol=1e-5)
    end

    @testset "Mooncake rrule!! on Metal" begin
        translation = MtlArray(zeros(Float32, 2, 1))
        rotation = MtlArray(Float32.(reshape([1 0; 0 1], 2, 2, 1)))
        zoom = MtlArray(ones(Float32, 2, 1))
        shear = MtlArray(zeros(Float32, 2, 1))

        d_translation = Metal.zeros(Float32, 2, 1)
        d_rotation = Metal.zeros(Float32, 2, 2, 1)
        d_zoom = Metal.zeros(Float32, 2, 1)
        d_shear = Metal.zeros(Float32, 2, 1)

        translation_codual = CoDual(translation, d_translation)
        rotation_codual = CoDual(rotation, d_rotation)
        zoom_codual = CoDual(zoom, d_zoom)
        shear_codual = CoDual(shear, d_shear)
        fn_codual = CoDual(compose_affine, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, translation_codual, rotation_codual, zoom_codual, shear_codual)

        @test output_codual.x isa MtlArray

        output_codual.dx .= 1.0f0
        pullback(NoRData())

        @test isapprox(Array(d_zoom), ones(Float32, 2, 1); rtol=1e-5)
    end

    @testset "Gradient correctness via finite differences" begin
        translation = rand(StableRNG(300), Float32, 2, 1) .- 0.5f0
        rotation = Float32.(reshape([1 0; 0 1], 2, 2, 1))
        zoom = ones(Float32, 2, 1) .+ rand(StableRNG(301), Float32, 2, 1) .* 0.2f0
        shear = rand(StableRNG(302), Float32, 2, 1) .- 0.5f0

        ε = 1f-4

        # Test zoom[1] gradient
        zoom_plus = copy(zoom); zoom_plus[1, 1] += ε
        zoom_minus = copy(zoom); zoom_minus[1, 1] -= ε
        out_plus = sum(compose_affine(translation, rotation, zoom_plus, shear))
        out_minus = sum(compose_affine(translation, rotation, zoom_minus, shear))
        fd_grad = (out_plus - out_minus) / (2ε)

        # Analytical
        d_translation = zeros(Float32, 2, 1)
        d_rotation = zeros(Float32, 2, 2, 1)
        d_zoom = zeros(Float32, 2, 1)
        d_shear = zeros(Float32, 2, 1)

        translation_codual = CoDual(translation, d_translation)
        rotation_codual = CoDual(rotation, d_rotation)
        zoom_codual = CoDual(zoom, d_zoom)
        shear_codual = CoDual(shear, d_shear)
        fn_codual = CoDual(compose_affine, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, translation_codual, rotation_codual, zoom_codual, shear_codual)
        output_codual.dx .= 1.0f0
        pullback(NoRData())

        @test isapprox(d_zoom[1, 1], fd_grad; rtol=1e-2)
    end
end

@testset "compose_affine 3D gradients" begin
    @testset "Mooncake rrule!! on Metal 3D" begin
        translation = MtlArray(zeros(Float32, 3, 1))
        rotation = MtlArray(Float32.(reshape([1 0 0; 0 1 0; 0 0 1], 3, 3, 1)))
        zoom = MtlArray(ones(Float32, 3, 1))
        shear = MtlArray(zeros(Float32, 3, 1))

        d_translation = Metal.zeros(Float32, 3, 1)
        d_rotation = Metal.zeros(Float32, 3, 3, 1)
        d_zoom = Metal.zeros(Float32, 3, 1)
        d_shear = Metal.zeros(Float32, 3, 1)

        translation_codual = CoDual(translation, d_translation)
        rotation_codual = CoDual(rotation, d_rotation)
        zoom_codual = CoDual(zoom, d_zoom)
        shear_codual = CoDual(shear, d_shear)
        fn_codual = CoDual(compose_affine, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, translation_codual, rotation_codual, zoom_codual, shear_codual)

        @test output_codual.x isa MtlArray
        @test size(output_codual.x) == (3, 4, 1)

        output_codual.dx .= 1.0f0
        pullback(NoRData())

        # Translation gradients should be 1
        @test isapprox(Array(d_translation), ones(Float32, 3, 1); rtol=1e-5)
    end
end

# ============================================================================
# torchreg Parity Tests
# ============================================================================

@testset "torchreg parity" begin
    using PythonCall

    sys = pyimport("sys")
    sys.path.append("/Users/daleblack/Documents/dev/torchreg_temp")

    torch = pyimport("torch")
    torchreg_affine = pyimport("torchreg.affine")
    np = pyimport("numpy")

    @testset "2D compose_affine" begin
        torch.manual_seed(42)

        N = 2
        D = 2

        translation_pt = torch.randn(N, D) * 0.1
        rotation_pt = torch.eye(D).unsqueeze(0).repeat(N, 1, 1)
        zoom_pt = torch.ones(N, D) + torch.randn(N, D) * 0.1
        shear_pt = torch.randn(N, D) * 0.1

        theta_pt = torchreg_affine.compose_affine(translation_pt, rotation_pt, zoom_pt, shear_pt)
        theta_pt_arr = pyconvert(Array{Float32}, theta_pt.detach().numpy())

        # Convert to Julia format
        translation_jl = permutedims(pyconvert(Array{Float32}, translation_pt.numpy()), (2, 1))
        rotation_jl = permutedims(pyconvert(Array{Float32}, rotation_pt.numpy()), (2, 3, 1))
        zoom_jl = permutedims(pyconvert(Array{Float32}, zoom_pt.numpy()), (2, 1))
        shear_jl = permutedims(pyconvert(Array{Float32}, shear_pt.numpy()), (2, 1))

        theta_jl = compose_affine(translation_jl, rotation_jl, zoom_jl, shear_jl)
        theta_jl_pt = permutedims(theta_jl, (3, 1, 2))

        @test isapprox(theta_pt_arr, theta_jl_pt; rtol=1e-5)
    end

    @testset "3D compose_affine" begin
        torch.manual_seed(123)

        N = 2
        D = 3

        translation_pt = torch.randn(N, D) * 0.1
        rotation_pt = torch.eye(D).unsqueeze(0).repeat(N, 1, 1)
        zoom_pt = torch.ones(N, D) + torch.randn(N, D) * 0.1
        shear_pt = torch.randn(N, D) * 0.1

        theta_pt = torchreg_affine.compose_affine(translation_pt, rotation_pt, zoom_pt, shear_pt)
        theta_pt_arr = pyconvert(Array{Float32}, theta_pt.detach().numpy())

        translation_jl = permutedims(pyconvert(Array{Float32}, translation_pt.numpy()), (2, 1))
        rotation_jl = permutedims(pyconvert(Array{Float32}, rotation_pt.numpy()), (2, 3, 1))
        zoom_jl = permutedims(pyconvert(Array{Float32}, zoom_pt.numpy()), (2, 1))
        shear_jl = permutedims(pyconvert(Array{Float32}, shear_pt.numpy()), (2, 1))

        theta_jl = compose_affine(translation_jl, rotation_jl, zoom_jl, shear_jl)
        theta_jl_pt = permutedims(theta_jl, (3, 1, 2))

        @test isapprox(theta_pt_arr, theta_jl_pt; rtol=1e-5)
    end
end
