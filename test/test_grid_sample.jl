# Test grid_sample on GPU (Metal) with Mooncake AD

using Test
using MedicalImageRegistration
using Metal
using StableRNGs
import Mooncake
import Mooncake: CoDual, NoFData, NoRData

# ============================================================================
# Helper: Create identity grid (samples at original positions)
# ============================================================================

function create_identity_grid_2d(X_out::Int, Y_out::Int, N::Int, ::Type{T}=Float32) where T
    grid = Array{T}(undef, 2, X_out, Y_out, N)
    for n in 1:N
        for j in 1:Y_out
            for i in 1:X_out
                # Normalized coordinates in [-1, 1]
                grid[1, i, j, n] = T(2) * T(i - 1) / T(X_out - 1) - T(1)
                grid[2, i, j, n] = T(2) * T(j - 1) / T(Y_out - 1) - T(1)
            end
        end
    end
    return grid
end

function create_identity_grid_3d(X_out::Int, Y_out::Int, Z_out::Int, N::Int, ::Type{T}=Float32) where T
    grid = Array{T}(undef, 3, X_out, Y_out, Z_out, N)
    for n in 1:N
        for k in 1:Z_out
            for j in 1:Y_out
                for i in 1:X_out
                    grid[1, i, j, k, n] = T(2) * T(i - 1) / T(X_out - 1) - T(1)
                    grid[2, i, j, k, n] = T(2) * T(j - 1) / T(Y_out - 1) - T(1)
                    grid[3, i, j, k, n] = T(2) * T(k - 1) / T(Z_out - 1) - T(1)
                end
            end
        end
    end
    return grid
end

# ============================================================================
# Basic Forward Pass Tests
# ============================================================================

@testset "grid_sample 2D forward" begin
    @testset "CPU basic test" begin
        # Simple 4x4 image, 1 channel, batch 1
        input = reshape(Float32.(1:16), 4, 4, 1, 1)
        # Identity grid - should return the same image
        grid = create_identity_grid_2d(4, 4, 1, Float32)

        output = grid_sample(input, grid)

        @test size(output) == (4, 4, 1, 1)
        # With identity grid and align_corners=true, output should match input
        @test isapprox(output, input; rtol=1e-5)
    end

    @testset "Metal GPU test" begin
        input_cpu = rand(StableRNG(42), Float32, 16, 16, 1, 2)
        grid_cpu = create_identity_grid_2d(16, 16, 2, Float32)

        input_mtl = MtlArray(input_cpu)
        grid_mtl = MtlArray(grid_cpu)

        output_mtl = grid_sample(input_mtl, grid_mtl)

        @test output_mtl isa MtlArray
        @test size(output_mtl) == (16, 16, 1, 2)

        # Compare GPU vs CPU result
        output_cpu = grid_sample(input_cpu, grid_cpu)
        @test isapprox(Array(output_mtl), output_cpu; rtol=1e-5)
    end

    @testset "Downsampling" begin
        input = MtlArray(rand(StableRNG(123), Float32, 32, 32, 3, 2))
        # Create grid that samples at half resolution
        grid_cpu = create_identity_grid_2d(16, 16, 2, Float32)
        grid = MtlArray(grid_cpu)

        output = grid_sample(input, grid)

        @test output isa MtlArray
        @test size(output) == (16, 16, 3, 2)
    end

    @testset "Upsampling" begin
        input = MtlArray(rand(StableRNG(124), Float32, 8, 8, 2, 1))
        grid_cpu = create_identity_grid_2d(16, 16, 1, Float32)
        grid = MtlArray(grid_cpu)

        output = grid_sample(input, grid)

        @test output isa MtlArray
        @test size(output) == (16, 16, 2, 1)
    end

    @testset "padding_mode=:border" begin
        input = MtlArray(rand(StableRNG(125), Float32, 8, 8, 1, 1))
        # Grid with some out-of-bounds coordinates
        grid_cpu = create_identity_grid_2d(8, 8, 1, Float32)
        grid_cpu[1, 1, :, :] .= -2.0f0  # x = -2 is out of bounds
        grid = MtlArray(grid_cpu)

        output = grid_sample(input, grid; padding_mode=:border)

        @test output isa MtlArray
        @test all(isfinite.(Array(output)))
    end

    @testset "padding_mode=:zeros" begin
        input = MtlArray(ones(Float32, 8, 8, 1, 1))
        # Grid with out-of-bounds coordinates
        grid_cpu = create_identity_grid_2d(8, 8, 1, Float32)
        grid_cpu[1, 1, :, :] .= -2.0f0  # x = -2 is out of bounds
        grid = MtlArray(grid_cpu)

        output = grid_sample(input, grid; padding_mode=:zeros)

        output_arr = Array(output)
        # The first column should have some zeros due to out-of-bounds x
        @test any(output_arr .< 1.0f0)
    end
end

@testset "grid_sample 3D forward" begin
    @testset "Metal GPU test" begin
        input_cpu = rand(StableRNG(200), Float32, 8, 8, 8, 1, 2)
        grid_cpu = create_identity_grid_3d(8, 8, 8, 2, Float32)

        input_mtl = MtlArray(input_cpu)
        grid_mtl = MtlArray(grid_cpu)

        output_mtl = grid_sample(input_mtl, grid_mtl)

        @test output_mtl isa MtlArray
        @test size(output_mtl) == (8, 8, 8, 1, 2)

        # With identity grid, should match input
        @test isapprox(Array(output_mtl), input_cpu; rtol=1e-4)
    end

    @testset "Downsampling 3D" begin
        input = MtlArray(rand(StableRNG(201), Float32, 16, 16, 16, 2, 1))
        grid_cpu = create_identity_grid_3d(8, 8, 8, 1, Float32)
        grid = MtlArray(grid_cpu)

        output = grid_sample(input, grid)

        @test output isa MtlArray
        @test size(output) == (8, 8, 8, 2, 1)
    end
end

# ============================================================================
# Gradient Tests
# ============================================================================

@testset "grid_sample 2D gradients" begin
    @testset "Mooncake rrule!! on CPU" begin
        input_cpu = rand(StableRNG(300), Float32, 8, 8, 1, 1)
        grid_cpu = create_identity_grid_2d(8, 8, 1, Float32)

        # Create fdata (gradient accumulators)
        input_fdata = zeros(Float32, size(input_cpu))
        grid_fdata = zeros(Float32, size(grid_cpu))

        # Create CoDuals
        input_codual = CoDual(input_cpu, input_fdata)
        grid_codual = CoDual(grid_cpu, grid_fdata)
        fn_codual = CoDual(grid_sample, NoFData())

        # Forward pass through rrule!!
        output_codual, pullback = Mooncake.rrule!!(fn_codual, input_codual, grid_codual)

        @test size(output_codual.x) == (8, 8, 1, 1)

        # Set upstream gradient to ones
        output_codual.dx .= 1.0f0

        # Run pullback
        pullback(NoRData())

        # Check gradients are non-zero
        @test any(input_fdata .!= 0)
        @test any(grid_fdata .!= 0)
    end

    @testset "Mooncake rrule!! on Metal" begin
        input_cpu = rand(StableRNG(301), Float32, 8, 8, 1, 1)
        grid_cpu = create_identity_grid_2d(8, 8, 1, Float32)

        input_mtl = MtlArray(input_cpu)
        grid_mtl = MtlArray(grid_cpu)

        # Create fdata on GPU
        input_fdata = Metal.zeros(Float32, size(input_mtl))
        grid_fdata = Metal.zeros(Float32, size(grid_mtl))

        # Create CoDuals
        input_codual = CoDual(input_mtl, input_fdata)
        grid_codual = CoDual(grid_mtl, grid_fdata)
        fn_codual = CoDual(grid_sample, NoFData())

        # Forward pass
        output_codual, pullback = Mooncake.rrule!!(fn_codual, input_codual, grid_codual)

        @test output_codual.x isa MtlArray
        @test size(output_codual.x) == (8, 8, 1, 1)

        # Set upstream gradient
        output_codual.dx .= 1.0f0

        # Run pullback
        pullback(NoRData())

        # Gradients should be non-zero
        @test any(Array(input_fdata) .!= 0)
        @test any(Array(grid_fdata) .!= 0)
    end

    @testset "Gradient correctness via finite differences" begin
        input_cpu = rand(StableRNG(302), Float32, 4, 4, 1, 1)
        grid_cpu = create_identity_grid_2d(4, 4, 1, Float32)

        ε = 1f-4

        # Test d_input gradient at a specific location
        test_i, test_j = 2, 2

        # Forward at (input + ε)
        input_plus = copy(input_cpu)
        input_plus[test_i, test_j, 1, 1] += ε
        out_plus = sum(grid_sample(input_plus, grid_cpu))

        # Forward at (input - ε)
        input_minus = copy(input_cpu)
        input_minus[test_i, test_j, 1, 1] -= ε
        out_minus = sum(grid_sample(input_minus, grid_cpu))

        # Finite difference gradient
        fd_grad = (out_plus - out_minus) / (2ε)

        # Analytical gradient via rrule!!
        input_fdata = zeros(Float32, size(input_cpu))
        grid_fdata = zeros(Float32, size(grid_cpu))

        input_codual = CoDual(input_cpu, input_fdata)
        grid_codual = CoDual(grid_cpu, grid_fdata)
        fn_codual = CoDual(grid_sample, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, input_codual, grid_codual)
        output_codual.dx .= 1.0f0
        pullback(NoRData())

        analytical_grad = input_fdata[test_i, test_j, 1, 1]

        @test isapprox(analytical_grad, fd_grad; rtol=1e-2)
    end

    @testset "Grid gradient correctness via finite differences" begin
        input_cpu = rand(StableRNG(303), Float32, 8, 8, 1, 1)
        grid_cpu = create_identity_grid_2d(8, 8, 1, Float32) .* 0.5f0

        ε = 1f-4

        # Test d_grid gradient at a specific location
        test_coord, test_i, test_j = 1, 4, 4

        # Forward at (grid + ε)
        grid_plus = copy(grid_cpu)
        grid_plus[test_coord, test_i, test_j, 1] += ε
        out_plus = sum(grid_sample(input_cpu, grid_plus))

        # Forward at (grid - ε)
        grid_minus = copy(grid_cpu)
        grid_minus[test_coord, test_i, test_j, 1] -= ε
        out_minus = sum(grid_sample(input_cpu, grid_minus))

        # Finite difference gradient
        fd_grad = (out_plus - out_minus) / (2ε)

        # Analytical gradient
        input_fdata = zeros(Float32, size(input_cpu))
        grid_fdata = zeros(Float32, size(grid_cpu))

        input_codual = CoDual(input_cpu, input_fdata)
        grid_codual = CoDual(grid_cpu, grid_fdata)
        fn_codual = CoDual(grid_sample, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, input_codual, grid_codual)
        output_codual.dx .= 1.0f0
        pullback(NoRData())

        analytical_grad = grid_fdata[test_coord, test_i, test_j, 1]

        @test isapprox(analytical_grad, fd_grad; rtol=1e-2)
    end
end

@testset "grid_sample 3D gradients" begin
    @testset "Mooncake rrule!! on Metal 3D" begin
        input_cpu = rand(StableRNG(400), Float32, 4, 4, 4, 1, 1)
        grid_cpu = create_identity_grid_3d(4, 4, 4, 1, Float32)

        input_mtl = MtlArray(input_cpu)
        grid_mtl = MtlArray(grid_cpu)

        input_fdata = Metal.zeros(Float32, size(input_mtl))
        grid_fdata = Metal.zeros(Float32, size(grid_mtl))

        input_codual = CoDual(input_mtl, input_fdata)
        grid_codual = CoDual(grid_mtl, grid_fdata)
        fn_codual = CoDual(grid_sample, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, input_codual, grid_codual)

        @test output_codual.x isa MtlArray
        @test size(output_codual.x) == (4, 4, 4, 1, 1)

        output_codual.dx .= 1.0f0
        pullback(NoRData())

        @test any(Array(input_fdata) .!= 0)
        @test any(Array(grid_fdata) .!= 0)
    end

    @testset "3D Gradient correctness via finite differences" begin
        input_cpu = rand(StableRNG(401), Float32, 4, 4, 4, 1, 1)
        grid_cpu = create_identity_grid_3d(4, 4, 4, 1, Float32)

        ε = 1f-4
        test_i, test_j, test_k = 2, 2, 2

        input_plus = copy(input_cpu)
        input_plus[test_i, test_j, test_k, 1, 1] += ε
        out_plus = sum(grid_sample(input_plus, grid_cpu))

        input_minus = copy(input_cpu)
        input_minus[test_i, test_j, test_k, 1, 1] -= ε
        out_minus = sum(grid_sample(input_minus, grid_cpu))

        fd_grad = (out_plus - out_minus) / (2ε)

        input_fdata = zeros(Float32, size(input_cpu))
        grid_fdata = zeros(Float32, size(grid_cpu))

        input_codual = CoDual(input_cpu, input_fdata)
        grid_codual = CoDual(grid_cpu, grid_fdata)
        fn_codual = CoDual(grid_sample, NoFData())

        output_codual, pullback = Mooncake.rrule!!(fn_codual, input_codual, grid_codual)
        output_codual.dx .= 1.0f0
        pullback(NoRData())

        analytical_grad = input_fdata[test_i, test_j, test_k, 1, 1]

        @test isapprox(analytical_grad, fd_grad; rtol=1e-2)
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

    # Helper to convert Julia array to PyTorch tensor
    function to_torch(arr::Array{Float32})
        return torch.from_numpy(np.array(arr))
    end

    @testset "2D bilinear align_corners=true" begin
        # Create test data
        input_jl = rand(StableRNG(500), Float32, 8, 8, 2, 2)  # (X, Y, C, N)
        grid_jl = (rand(StableRNG(501), Float32, 2, 6, 6, 2) .- 0.5f0) .* 1.8f0  # (2, X_out, Y_out, N)

        # Convert to PyTorch convention: (N, C, H, W) and (N, H, W, 2)
        # Julia (X, Y, C, N) -> PyTorch (N, C, Y, X) requires permutation
        input_pt = permutedims(input_jl, (4, 3, 2, 1))  # (N, C, Y, X)
        # Julia grid (2, X_out, Y_out, N) -> PyTorch (N, Y_out, X_out, 2)
        grid_pt = permutedims(grid_jl, (4, 3, 2, 1))  # (N, Y_out, X_out, 2)

        # Run Julia
        output_jl = grid_sample(input_jl, grid_jl; padding_mode=:zeros, align_corners=true)

        # Run PyTorch
        input_torch = to_torch(input_pt)
        grid_torch = to_torch(grid_pt)
        output_torch = F.grid_sample(input_torch, grid_torch, mode="bilinear",
                                     padding_mode="zeros", align_corners=true)
        output_pt = pyconvert(Array{Float32}, output_torch.detach().numpy())

        # Convert PyTorch output back to Julia convention
        output_pt_jl = permutedims(output_pt, (4, 3, 2, 1))  # (X, Y, C, N)

        @test size(output_jl) == size(output_pt_jl)
        @test isapprox(output_jl, output_pt_jl; rtol=1e-5)
    end

    @testset "2D bilinear align_corners=false" begin
        input_jl = rand(StableRNG(502), Float32, 8, 8, 1, 1)
        grid_jl = (rand(StableRNG(503), Float32, 2, 4, 4, 1) .- 0.5f0) .* 1.8f0

        input_pt = permutedims(input_jl, (4, 3, 2, 1))
        grid_pt = permutedims(grid_jl, (4, 3, 2, 1))

        output_jl = grid_sample(input_jl, grid_jl; padding_mode=:zeros, align_corners=false)

        input_torch = to_torch(input_pt)
        grid_torch = to_torch(grid_pt)
        output_torch = F.grid_sample(input_torch, grid_torch, mode="bilinear",
                                     padding_mode="zeros", align_corners=false)
        output_pt = pyconvert(Array{Float32}, output_torch.detach().numpy())
        output_pt_jl = permutedims(output_pt, (4, 3, 2, 1))

        @test isapprox(output_jl, output_pt_jl; rtol=1e-5)
    end

    @testset "2D border padding" begin
        input_jl = rand(StableRNG(504), Float32, 8, 8, 1, 1)
        # Grid with values outside [-1, 1]
        grid_jl = (rand(StableRNG(505), Float32, 2, 4, 4, 1) .- 0.5f0) .* 3.0f0

        input_pt = permutedims(input_jl, (4, 3, 2, 1))
        grid_pt = permutedims(grid_jl, (4, 3, 2, 1))

        output_jl = grid_sample(input_jl, grid_jl; padding_mode=:border, align_corners=true)

        input_torch = to_torch(input_pt)
        grid_torch = to_torch(grid_pt)
        output_torch = F.grid_sample(input_torch, grid_torch, mode="bilinear",
                                     padding_mode="border", align_corners=true)
        output_pt = pyconvert(Array{Float32}, output_torch.detach().numpy())
        output_pt_jl = permutedims(output_pt, (4, 3, 2, 1))

        @test isapprox(output_jl, output_pt_jl; rtol=1e-5)
    end

    @testset "3D trilinear align_corners=true" begin
        input_jl = rand(StableRNG(510), Float32, 6, 6, 6, 1, 1)  # (X, Y, Z, C, N)
        grid_jl = (rand(StableRNG(511), Float32, 3, 4, 4, 4, 1) .- 0.5f0) .* 1.8f0  # (3, X_out, Y_out, Z_out, N)

        # Julia (X, Y, Z, C, N) -> PyTorch (N, C, D, H, W) = (N, C, Z, Y, X)
        input_pt = permutedims(input_jl, (5, 4, 3, 2, 1))
        # Julia grid (3, X_out, Y_out, Z_out, N) -> PyTorch (N, D_out, H_out, W_out, 3)
        grid_pt = permutedims(grid_jl, (5, 4, 3, 2, 1))

        output_jl = grid_sample(input_jl, grid_jl; padding_mode=:zeros, align_corners=true)

        input_torch = to_torch(input_pt)
        grid_torch = to_torch(grid_pt)
        # PyTorch uses "bilinear" for 5D input (it automatically does trilinear)
        output_torch = F.grid_sample(input_torch, grid_torch, mode="bilinear",
                                     padding_mode="zeros", align_corners=true)
        output_pt = pyconvert(Array{Float32}, output_torch.detach().numpy())
        output_pt_jl = permutedims(output_pt, (5, 4, 3, 2, 1))

        @test size(output_jl) == size(output_pt_jl)
        @test isapprox(output_jl, output_pt_jl; rtol=1e-5)
    end
end
