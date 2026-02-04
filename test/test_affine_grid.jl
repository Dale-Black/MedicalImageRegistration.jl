# Test affine_grid on GPU (Metal) with Mooncake AD

using Test
using MedicalImageRegistration
using StableRNGs
import Mooncake
import Mooncake: CoDual, NoFData, NoRData

# Include test helpers for conditional GPU testing
include("test_helpers.jl")

# ============================================================================
# Basic Forward Pass Tests
# ============================================================================

@testset "affine_grid 2D forward" begin
    @testset "Identity transformation" begin
        theta = zeros(Float32, 2, 3, 1)
        theta[1, 1, 1] = 1.0f0  # scale x
        theta[2, 2, 1] = 1.0f0  # scale y

        grid = affine_grid(theta, (4, 4))

        @test size(grid) == (2, 4, 4, 1)

        # Check corners
        @test isapprox(grid[:, 1, 1, 1], [-1.0f0, -1.0f0]; rtol=1e-5)
        @test isapprox(grid[:, 4, 1, 1], [1.0f0, -1.0f0]; rtol=1e-5)
        @test isapprox(grid[:, 1, 4, 1], [-1.0f0, 1.0f0]; rtol=1e-5)
        @test isapprox(grid[:, 4, 4, 1], [1.0f0, 1.0f0]; rtol=1e-5)
    end

    @testset "Translation" begin
        theta = zeros(Float32, 2, 3, 1)
        theta[1, 1, 1] = 1.0f0
        theta[2, 2, 1] = 1.0f0
        theta[1, 3, 1] = 0.5f0  # translate x by 0.5
        theta[2, 3, 1] = -0.25f0  # translate y by -0.25

        grid = affine_grid(theta, (4, 4))

        # Corners should be shifted
        @test isapprox(grid[:, 1, 1, 1], [-0.5f0, -1.25f0]; rtol=1e-5)
        @test isapprox(grid[:, 4, 4, 1], [1.5f0, 0.75f0]; rtol=1e-5)
    end

    @testset "Scale" begin
        theta = zeros(Float32, 2, 3, 1)
        theta[1, 1, 1] = 2.0f0  # scale x by 2
        theta[2, 2, 1] = 0.5f0  # scale y by 0.5

        grid = affine_grid(theta, (4, 4))

        @test isapprox(grid[:, 1, 1, 1], [-2.0f0, -0.5f0]; rtol=1e-5)
        @test isapprox(grid[:, 4, 4, 1], [2.0f0, 0.5f0]; rtol=1e-5)
    end

    if METAL_AVAILABLE
        @testset "Metal GPU test" begin
            theta_cpu = rand(StableRNG(100), Float32, 2, 3, 2) .- 0.5f0
            theta_mtl = MtlArray(theta_cpu)

            grid_cpu = affine_grid(theta_cpu, (8, 8))
            grid_mtl = affine_grid(theta_mtl, (8, 8))

            @test grid_mtl isa MtlArray
            @test size(grid_mtl) == (2, 8, 8, 2)
            @test isapprox(Array(grid_mtl), grid_cpu; rtol=1e-5)
        end
    end

    @testset "4-tuple size (XYCN convention)" begin
        theta = zeros(Float32, 2, 3, 1)
        theta[1, 1, 1] = 1.0f0
        theta[2, 2, 1] = 1.0f0

        grid = affine_grid(theta, (8, 6, 3, 1))  # X=8, Y=6, C=3, N=1

        @test size(grid) == (2, 8, 6, 1)  # Output ignores C
    end
end

@testset "affine_grid 3D forward" begin
    @testset "Identity transformation" begin
        theta = zeros(Float32, 3, 4, 1)
        theta[1, 1, 1] = 1.0f0
        theta[2, 2, 1] = 1.0f0
        theta[3, 3, 1] = 1.0f0

        grid = affine_grid(theta, (3, 3, 3))

        @test size(grid) == (3, 3, 3, 3, 1)

        # Check corners
        @test isapprox(grid[:, 1, 1, 1, 1], [-1.0f0, -1.0f0, -1.0f0]; rtol=1e-5)
        @test isapprox(grid[:, 3, 3, 3, 1], [1.0f0, 1.0f0, 1.0f0]; rtol=1e-5)
    end

    if METAL_AVAILABLE
        @testset "Metal GPU test" begin
            theta_cpu = rand(StableRNG(200), Float32, 3, 4, 1) .- 0.5f0
            theta_mtl = MtlArray(theta_cpu)

            grid_cpu = affine_grid(theta_cpu, (4, 4, 4))
            grid_mtl = affine_grid(theta_mtl, (4, 4, 4))

            @test grid_mtl isa MtlArray
            @test size(grid_mtl) == (3, 4, 4, 4, 1)
            @test isapprox(Array(grid_mtl), grid_cpu; rtol=1e-5)
        end
    end
end

# ============================================================================
# Gradient Tests
# ============================================================================

@testset "affine_grid 2D gradients" begin
    @testset "Mooncake rrule!! on CPU" begin
        theta = zeros(Float32, 2, 3, 1)
        theta[1, 1, 1] = 1.0f0
        theta[2, 2, 1] = 1.0f0

        theta_fdata = zeros(Float32, 2, 3, 1)
        theta_codual = CoDual(theta, theta_fdata)
        fn_codual = CoDual(affine_grid, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, theta_codual, CoDual((4, 4), NoFData()))

        @test size(output_codual.x) == (2, 4, 4, 1)

        output_codual.dx .= 1.0f0
        pullback(NoRData())

        # Translation gradients should equal grid size
        @test isapprox(theta_fdata[1, 3, 1], 16.0f0; rtol=1e-5)  # 4*4 = 16
        @test isapprox(theta_fdata[2, 3, 1], 16.0f0; rtol=1e-5)
    end

    if METAL_AVAILABLE
        @testset "Mooncake rrule!! on Metal" begin
            theta_cpu = rand(StableRNG(300), Float32, 2, 3, 1)
            theta_mtl = MtlArray(theta_cpu)
            theta_fdata = Metal.zeros(Float32, 2, 3, 1)

            theta_codual = CoDual(theta_mtl, theta_fdata)
            fn_codual = CoDual(affine_grid, NoFData())

            output_codual, pullback = Mooncake.rrule!!(fn_codual, theta_codual, CoDual((8, 8), NoFData()))

            @test output_codual.x isa MtlArray

            output_codual.dx .= 1.0f0
            pullback(NoRData())

            # Check Metal gives non-zero gradients
            @test any(Array(theta_fdata) .!= 0)
        end
    end

    @testset "Translation gradient via finite differences" begin
        theta = zeros(Float32, 2, 3, 1)
        theta[1, 1, 1] = 1.0f0
        theta[2, 2, 1] = 1.0f0

        ε = 1f-4

        # Gradient for x-translation
        theta_plus = copy(theta); theta_plus[1, 3, 1] += ε
        theta_minus = copy(theta); theta_minus[1, 3, 1] -= ε
        out_plus = sum(affine_grid(theta_plus, (4, 4)))
        out_minus = sum(affine_grid(theta_minus, (4, 4)))
        fd_grad = (out_plus - out_minus) / (2ε)

        # Analytical
        theta_fdata = zeros(Float32, 2, 3, 1)
        theta_codual = CoDual(theta, theta_fdata)
        fn_codual = CoDual(affine_grid, NoFData())
        output_codual, pullback = Mooncake.rrule!!(fn_codual, theta_codual, CoDual((4, 4), NoFData()))
        output_codual.dx .= 1.0f0
        pullback(NoRData())

        @test isapprox(theta_fdata[1, 3, 1], fd_grad; rtol=1e-2)
    end

    @testset "Scale gradient via finite differences" begin
        # Use asymmetric scaling to avoid symmetric cancellation
        theta = zeros(Float32, 2, 3, 1)
        theta[1, 1, 1] = 1.5f0
        theta[2, 2, 1] = 0.8f0
        theta[1, 3, 1] = 0.3f0  # some translation to break symmetry

        ε = 1f-4

        # Gradient for x-scale (theta[1,1])
        # We use a weighted sum to avoid symmetric cancellation
        function weighted_sum(grid)
            # Weight by position to break symmetry
            total = 0.0f0
            for n in 1:size(grid, 4), j in 1:size(grid, 3), i in 1:size(grid, 2)
                w = Float32(i + j)
                total += w * grid[1, i, j, n]
            end
            return total
        end

        theta_plus = copy(theta); theta_plus[1, 1, 1] += ε
        theta_minus = copy(theta); theta_minus[1, 1, 1] -= ε
        grid_plus = affine_grid(theta_plus, (4, 4))
        grid_minus = affine_grid(theta_minus, (4, 4))
        fd_grad = (weighted_sum(grid_plus) - weighted_sum(grid_minus)) / (2ε)

        # Analytical - with matching weights
        theta_fdata = zeros(Float32, 2, 3, 1)
        theta_codual = CoDual(theta, theta_fdata)
        fn_codual = CoDual(affine_grid, NoFData())
        output_codual, pullback = Mooncake.rrule!!(fn_codual, theta_codual, CoDual((4, 4), NoFData()))

        # Set weighted upstream gradient
        for j in 1:4, i in 1:4
            w = Float32(i + j)
            output_codual.dx[1, i, j, 1] = w
        end
        pullback(NoRData())

        @test isapprox(theta_fdata[1, 1, 1], fd_grad; rtol=1e-2)
    end
end

@testset "affine_grid 3D gradients" begin
    if METAL_AVAILABLE
        @testset "Mooncake rrule!! on Metal 3D" begin
            theta_cpu = rand(StableRNG(400), Float32, 3, 4, 1)
            theta_mtl = MtlArray(theta_cpu)
            theta_fdata = Metal.zeros(Float32, 3, 4, 1)

            theta_codual = CoDual(theta_mtl, theta_fdata)
            fn_codual = CoDual(affine_grid, NoFData())

            output_codual, pullback = Mooncake.rrule!!(fn_codual, theta_codual, CoDual((4, 4, 4), NoFData()))

            @test output_codual.x isa MtlArray
            @test size(output_codual.x) == (3, 4, 4, 4, 1)

            output_codual.dx .= 1.0f0
            pullback(NoRData())

            # Translation gradients should equal grid size (4*4*4 = 64)
            theta_fdata_cpu = Array(theta_fdata)
            @test isapprox(theta_fdata_cpu[1, 4, 1], 64.0f0; rtol=1e-4)
            @test isapprox(theta_fdata_cpu[2, 4, 1], 64.0f0; rtol=1e-4)
            @test isapprox(theta_fdata_cpu[3, 4, 1], 64.0f0; rtol=1e-4)
        end
    end

    @testset "Mooncake rrule!! on CPU 3D" begin
        theta = rand(StableRNG(400), Float32, 3, 4, 1)
        theta_fdata = zeros(Float32, 3, 4, 1)

        theta_codual = CoDual(theta, theta_fdata)
        fn_codual = CoDual(affine_grid, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, theta_codual, CoDual((4, 4, 4), NoFData()))

        @test size(output_codual.x) == (3, 4, 4, 4, 1)

        output_codual.dx .= 1.0f0
        pullback(NoRData())

        # Translation gradients should equal grid size (4*4*4 = 64)
        @test isapprox(theta_fdata[1, 4, 1], 64.0f0; rtol=1e-4)
        @test isapprox(theta_fdata[2, 4, 1], 64.0f0; rtol=1e-4)
        @test isapprox(theta_fdata[3, 4, 1], 64.0f0; rtol=1e-4)
    end
end

# ============================================================================
# PyTorch Parity Tests
# ============================================================================

@testset "PyTorch parity" begin
    using PythonCall

    torch = pyimport("torch")
    F = pyimport("torch.nn.functional")
    np = pyimport("numpy")

    function to_torch(arr)
        return torch.from_numpy(np.array(arr))
    end

    @testset "2D identity" begin
        theta_jl = zeros(Float32, 2, 3, 1)
        theta_jl[1, 1, 1] = 1.0f0
        theta_jl[2, 2, 1] = 1.0f0

        # Julia -> PyTorch: (2, 3, N) -> (N, 2, 3)
        theta_pt = permutedims(theta_jl, (3, 1, 2))

        grid_jl = affine_grid(theta_jl, (4, 4))

        size_pt = pytuple((1, 1, 4, 4))
        grid_torch = F.affine_grid(to_torch(theta_pt), size_pt, align_corners=true)
        grid_pt_raw = pyconvert(Array{Float32}, grid_torch.numpy())
        # PyTorch (N, H, W, 2) -> Julia (2, W, H, N)
        grid_pt_jl = permutedims(grid_pt_raw, (4, 3, 2, 1))

        @test isapprox(grid_jl, grid_pt_jl; rtol=1e-5)
    end

    @testset "2D with transform" begin
        theta_jl = rand(StableRNG(500), Float32, 2, 3, 2) .- 0.5f0
        theta_pt = permutedims(theta_jl, (3, 1, 2))

        grid_jl = affine_grid(theta_jl, (8, 8))

        size_pt = pytuple((2, 1, 8, 8))
        grid_torch = F.affine_grid(to_torch(theta_pt), size_pt, align_corners=true)
        grid_pt_raw = pyconvert(Array{Float32}, grid_torch.numpy())
        grid_pt_jl = permutedims(grid_pt_raw, (4, 3, 2, 1))

        @test isapprox(grid_jl, grid_pt_jl; rtol=1e-5)
    end

    @testset "3D identity" begin
        theta_jl = zeros(Float32, 3, 4, 1)
        theta_jl[1, 1, 1] = 1.0f0
        theta_jl[2, 2, 1] = 1.0f0
        theta_jl[3, 3, 1] = 1.0f0

        theta_pt = permutedims(theta_jl, (3, 1, 2))

        grid_jl = affine_grid(theta_jl, (4, 4, 4))

        size_pt = pytuple((1, 1, 4, 4, 4))
        grid_torch = F.affine_grid(to_torch(theta_pt), size_pt, align_corners=true)
        grid_pt_raw = pyconvert(Array{Float32}, grid_torch.numpy())
        # PyTorch (N, D, H, W, 3) -> Julia (3, W, H, D, N)
        grid_pt_jl = permutedims(grid_pt_raw, (5, 4, 3, 2, 1))

        @test isapprox(grid_jl, grid_pt_jl; rtol=1e-5)
    end

    @testset "3D with transform" begin
        theta_jl = rand(StableRNG(510), Float32, 3, 4, 1) .- 0.5f0
        theta_pt = permutedims(theta_jl, (3, 1, 2))

        grid_jl = affine_grid(theta_jl, (4, 4, 4))

        size_pt = pytuple((1, 1, 4, 4, 4))
        grid_torch = F.affine_grid(to_torch(theta_pt), size_pt, align_corners=true)
        grid_pt_raw = pyconvert(Array{Float32}, grid_torch.numpy())
        grid_pt_jl = permutedims(grid_pt_raw, (5, 4, 3, 2, 1))

        @test isapprox(grid_jl, grid_pt_jl; rtol=1e-5)
    end
end
