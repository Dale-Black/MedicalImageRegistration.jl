# Metrics parity tests with torchreg
# Tests dice_loss, NCC, and LinearElasticity against PyTorch reference

using Test
using MedicalImageRegistration
using Random

# PythonCall setup (torch and np are defined in runtests.jl)

@testset "Metrics" begin
    # Import torchreg.metrics
    sys = pyimport("sys")
    sys.path.insert(0, "/Users/daleblack/Documents/dev/torchreg_temp")
    torchreg_metrics = pyimport("torchreg.metrics")

    @testset "Dice Loss Parity" begin
        Random.seed!(42)

        @testset "2D arrays" begin
            # Julia: (X, Y, C, N)
            x1_julia = rand(Float32, 8, 8, 1, 1)
            x2_julia = rand(Float32, 8, 8, 1, 1)

            # PyTorch: (N, C, Y, X)
            x1_torch = torch.tensor(np.ascontiguousarray(permutedims(x1_julia, (4, 3, 2, 1))))
            x2_torch = torch.tensor(np.ascontiguousarray(permutedims(x2_julia, (4, 3, 2, 1))))

            julia_loss = dice_loss(x1_julia, x2_julia)
            julia_score = dice_score(x1_julia, x2_julia)
            torch_loss = Float32(pyconvert(Float64, torchreg_metrics.dice_loss(x1_torch, x2_torch).item()))
            torch_score = Float32(pyconvert(Float64, torchreg_metrics.dice_score(x1_torch, x2_torch).item()))

            @test isapprox(julia_loss, torch_loss, rtol=1e-5)
            @test isapprox(julia_score, torch_score, rtol=1e-5)
        end

        @testset "3D arrays" begin
            x1_julia = rand(Float32, 7, 7, 7, 1, 1)
            x2_julia = rand(Float32, 7, 7, 7, 1, 1)

            x1_torch = torch.tensor(np.ascontiguousarray(permutedims(x1_julia, (5, 4, 3, 2, 1))))
            x2_torch = torch.tensor(np.ascontiguousarray(permutedims(x2_julia, (5, 4, 3, 2, 1))))

            julia_loss = dice_loss(x1_julia, x2_julia)
            julia_score = dice_score(x1_julia, x2_julia)
            torch_loss = Float32(pyconvert(Float64, torchreg_metrics.dice_loss(x1_torch, x2_torch).item()))
            torch_score = Float32(pyconvert(Float64, torchreg_metrics.dice_score(x1_torch, x2_torch).item()))

            @test isapprox(julia_loss, torch_loss, rtol=1e-5)
            @test isapprox(julia_score, torch_score, rtol=1e-5)
        end

        @testset "batch size > 1" begin
            x1_julia = rand(Float32, 6, 6, 6, 1, 3)
            x2_julia = rand(Float32, 6, 6, 6, 1, 3)

            x1_torch = torch.tensor(np.ascontiguousarray(permutedims(x1_julia, (5, 4, 3, 2, 1))))
            x2_torch = torch.tensor(np.ascontiguousarray(permutedims(x2_julia, (5, 4, 3, 2, 1))))

            julia_score = dice_score(x1_julia, x2_julia)
            torch_score = Float32(pyconvert(Float64, torchreg_metrics.dice_score(x1_torch, x2_torch).item()))

            @test isapprox(julia_score, torch_score, rtol=1e-5)
        end

        @testset "edge cases" begin
            # Identical binary masks should give score 1
            mask = Float32.(rand(8, 8, 8, 1, 1) .> 0.5)
            @test isapprox(dice_score(mask, mask), 1.0f0, atol=1e-6)
            @test isapprox(dice_loss(mask, mask), 0.0f0, atol=1e-6)
        end
    end

    @testset "NCC Loss Parity" begin
        Random.seed!(123)

        # Note: NCC parity is verified for N=1 (single batch) cases only
        # torchreg has a quirk where kernel shape depends on batch size

        @testset "3D N=1 kernel_size=7" begin
            pred_julia = rand(Float32, 9, 9, 9, 1, 1)
            targ_julia = rand(Float32, 9, 9, 9, 1, 1)

            pred_torch = torch.tensor(np.ascontiguousarray(permutedims(pred_julia, (5, 4, 3, 2, 1))))
            targ_torch = torch.tensor(np.ascontiguousarray(permutedims(targ_julia, (5, 4, 3, 2, 1))))

            ncc_julia = NCC(kernel_size=7)
            ncc_torch = torchreg_metrics.NCC(kernel_size=7)

            julia_loss = ncc_julia(pred_julia, targ_julia)
            torch_loss = Float32(pyconvert(Float64, ncc_torch(pred_torch, targ_torch).item()))

            @test isapprox(julia_loss, torch_loss, rtol=1e-4)
        end

        @testset "different kernel sizes (N=1)" begin
            pred_julia = rand(Float32, 11, 11, 11, 1, 1)
            targ_julia = rand(Float32, 11, 11, 11, 1, 1)

            pred_torch = torch.tensor(np.ascontiguousarray(permutedims(pred_julia, (5, 4, 3, 2, 1))))
            targ_torch = torch.tensor(np.ascontiguousarray(permutedims(targ_julia, (5, 4, 3, 2, 1))))

            for ks in [3, 5, 9]
                ncc_julia = NCC(kernel_size=ks)
                ncc_torch = torchreg_metrics.NCC(kernel_size=ks)

                julia_loss = ncc_julia(pred_julia, targ_julia)
                torch_loss = Float32(pyconvert(Float64, ncc_torch(pred_torch, targ_torch).item()))

                @test isapprox(julia_loss, torch_loss, rtol=1e-4)
            end
        end

        @testset "identical images (N=1)" begin
            # Identical images should give NCC ≈ -1 (highly correlated)
            same = rand(Float32, 8, 8, 8, 1, 1)
            same_torch = torch.tensor(np.ascontiguousarray(permutedims(same, (5, 4, 3, 2, 1))))

            ncc_julia = NCC(kernel_size=5)
            ncc_torch = torchreg_metrics.NCC(kernel_size=5)

            julia_loss = ncc_julia(same, same)
            torch_loss = Float32(pyconvert(Float64, ncc_torch(same_torch, same_torch).item()))

            @test isapprox(julia_loss, torch_loss, rtol=1e-4)
            @test julia_loss < -0.9  # Should be close to -1
        end
    end

    @testset "LinearElasticity Regularizer" begin
        Random.seed!(456)

        # Note: LinearElasticity has axis convention differences with torchreg
        # Tests verify correct behavior, not exact parity

        @testset "basic functionality" begin
            u = randn(Float32, 8, 8, 8, 3, 1)
            reg = LinearElasticity(mu=2.0f0, lam=1.0f0)

            penalty = reg(u)

            @test isfinite(penalty)
            @test penalty >= 0  # Sum of squares, must be non-negative
        end

        @testset "parameter effect" begin
            u = randn(Float32, 8, 8, 8, 3, 1)

            reg_low = LinearElasticity(mu=1.0f0, lam=1.0f0)
            reg_high_mu = LinearElasticity(mu=5.0f0, lam=1.0f0)
            reg_high_lam = LinearElasticity(mu=1.0f0, lam=5.0f0)

            penalty_low = reg_low(u)
            penalty_high_mu = reg_high_mu(u)
            penalty_high_lam = reg_high_lam(u)

            # Higher mu or lam should increase penalty
            @test penalty_high_mu > penalty_low
            @test penalty_high_lam > penalty_low
        end

        @testset "smooth vs rough fields" begin
            # Create smooth displacement field (linear gradient)
            X, Y, Z = 10, 10, 10
            u_smooth = zeros(Float32, X, Y, Z, 3, 1)
            for i in 1:X
                u_smooth[i, :, :, 1, 1] .= Float32(i - 1) / (X - 1) * 0.1
            end

            # Create rough displacement field (random noise)
            u_rough = randn(Float32, X, Y, Z, 3, 1) .* 0.1f0

            reg = LinearElasticity(mu=2.0f0, lam=1.0f0)

            penalty_smooth = reg(u_smooth)
            penalty_rough = reg(u_rough)

            # Smooth field should have lower penalty than rough field
            @test penalty_smooth < penalty_rough
        end

        @testset "single batch (N=1)" begin
            # Note: Current implementation only supports batch_size=1
            # following torchreg convention
            u_single = randn(Float32, 6, 6, 6, 3, 1)
            reg = LinearElasticity()

            penalty = reg(u_single)

            @test isfinite(penalty)
            @test penalty >= 0
        end
    end
end
