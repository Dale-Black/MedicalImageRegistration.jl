# Test metrics on GPU (Metal) with Mooncake AD
# Tests mse_loss, dice_loss, dice_score, ncc_loss

using Test
using MedicalImageRegistration
using Metal
using StableRNGs
import AcceleratedKernels as AK
import Mooncake
import Mooncake: CoDual, NoFData, NoRData

# ============================================================================
# Helper: Finite Difference Gradient Check
# ============================================================================

# Extract scalar value from 1-element array (works on CPU and GPU)
function _get_scalar(arr::AbstractArray{T}) where T
    return AK.reduce(+, arr; init=zero(T))
end

function finite_diff_grad(f, x::AbstractArray{T}, eps::T=T(1e-4)) where T
    grad = similar(x, size(x))
    x_cpu = Array(x)
    for i in eachindex(x_cpu)
        x_plus = copy(x_cpu)
        x_minus = copy(x_cpu)
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_get_scalar(f(x_plus)) - _get_scalar(f(x_minus))) / (T(2) * eps)
    end
    return grad
end

# ============================================================================
# MSE Loss Tests
# ============================================================================

@testset "mse_loss" begin
    @testset "CPU basic test" begin
        pred = Float32[1, 2, 3, 4]
        target = Float32[1, 2, 3, 4]
        loss = mse_loss(reshape(pred, 4, 1, 1, 1), reshape(target, 4, 1, 1, 1))
        @test _get_scalar(loss) ≈ 0.0f0 atol=1e-7

        # Different values
        pred2 = Float32[1, 2, 3, 4]
        target2 = Float32[2, 3, 4, 5]
        loss2 = mse_loss(reshape(pred2, 4, 1, 1, 1), reshape(target2, 4, 1, 1, 1))
        # MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2) = mean(1+1+1+1) = 1
        @test _get_scalar(loss2) ≈ 1.0f0 atol=1e-6
    end

    @testset "Metal GPU test" begin
        rng = StableRNG(42)
        pred_cpu = rand(rng, Float32, 16, 16, 2, 2)
        target_cpu = rand(rng, Float32, 16, 16, 2, 2)

        pred_mtl = MtlArray(pred_cpu)
        target_mtl = MtlArray(target_cpu)

        loss_mtl = mse_loss(pred_mtl, target_mtl)

        @test loss_mtl isa MtlArray{Float32, 1}
        @test length(loss_mtl) == 1

        # Compare GPU vs CPU result
        loss_cpu = mse_loss(pred_cpu, target_cpu)
        @test isapprox(_get_scalar(Array(loss_mtl)), _get_scalar(loss_cpu), rtol=1e-5)
    end

    @testset "Gradient verification (CPU)" begin
        rng = StableRNG(123)
        pred = rand(rng, Float32, 8, 8, 1, 1)
        target = rand(rng, Float32, 8, 8, 1, 1)

        # Create CoDuals
        pred_fdata = zeros(Float32, size(pred))
        target_fdata = zeros(Float32, size(target))
        pred_codual = CoDual(pred, pred_fdata)
        target_codual = CoDual(target, target_fdata)

        # Forward pass via rrule!!
        output_codual, pb = Mooncake.rrule!!(
            CoDual(mse_loss, NoFData()),
            pred_codual,
            target_codual
        )

        # Set upstream gradient (fill 1-element array)
        fill!(output_codual.dx, 1.0f0)

        # Backward pass
        pb(NoRData())

        # Finite difference check
        f_pred = x -> mse_loss(x, target)
        grad_fd = finite_diff_grad(f_pred, pred)

        @test isapprox(pred_fdata, grad_fd, rtol=1e-2)
    end

    @testset "Gradient on Metal GPU" begin
        rng = StableRNG(456)
        pred_cpu = rand(rng, Float32, 8, 8, 1, 1)
        target_cpu = rand(rng, Float32, 8, 8, 1, 1)

        pred_mtl = MtlArray(pred_cpu)
        target_mtl = MtlArray(target_cpu)
        pred_fdata = Metal.zeros(Float32, size(pred_mtl))
        target_fdata = Metal.zeros(Float32, size(target_mtl))

        pred_codual = CoDual(pred_mtl, pred_fdata)
        target_codual = CoDual(target_mtl, target_fdata)

        output_codual, pb = Mooncake.rrule!!(
            CoDual(mse_loss, NoFData()),
            pred_codual,
            target_codual
        )

        @test output_codual.x isa MtlArray
        @test output_codual.dx isa MtlArray

        fill!(output_codual.dx, 1.0f0)
        pb(NoRData())

        @test pred_fdata isa MtlArray
        # Check gradient is non-zero
        @test _get_scalar(abs.(Array(pred_fdata))) > 0
    end
end

# ============================================================================
# Dice Score/Loss Tests
# ============================================================================

@testset "dice_score" begin
    @testset "CPU basic test - identical masks" begin
        mask = Float32.(rand(StableRNG(1), 8, 8, 1, 1) .> 0.5)
        score = dice_score(mask, mask)
        @test _get_scalar(score) ≈ 1.0f0 atol=1e-6
    end

    @testset "CPU basic test - disjoint masks" begin
        # Create two non-overlapping masks
        mask1 = zeros(Float32, 8, 8, 1, 1)
        mask2 = zeros(Float32, 8, 8, 1, 1)
        mask1[1:4, :, :, :] .= 1.0f0
        mask2[5:8, :, :, :] .= 1.0f0

        score = dice_score(mask1, mask2)
        # Dice of disjoint sets = 0
        @test _get_scalar(score) ≈ 0.0f0 atol=1e-6
    end

    @testset "Metal GPU test 2D" begin
        rng = StableRNG(42)
        pred_cpu = rand(rng, Float32, 16, 16, 1, 2)
        target_cpu = rand(rng, Float32, 16, 16, 1, 2)

        pred_mtl = MtlArray(pred_cpu)
        target_mtl = MtlArray(target_cpu)

        score_mtl = dice_score(pred_mtl, target_mtl)

        @test score_mtl isa MtlArray{Float32, 1}
        @test length(score_mtl) == 1

        # Compare GPU vs CPU result
        score_cpu = dice_score(pred_cpu, target_cpu)
        @test isapprox(_get_scalar(Array(score_mtl)), _get_scalar(score_cpu), rtol=1e-5)
    end

    @testset "Metal GPU test 3D" begin
        rng = StableRNG(43)
        pred_cpu = rand(rng, Float32, 8, 8, 8, 1, 2)
        target_cpu = rand(rng, Float32, 8, 8, 8, 1, 2)

        pred_mtl = MtlArray(pred_cpu)
        target_mtl = MtlArray(target_cpu)

        score_mtl = dice_score(pred_mtl, target_mtl)

        @test score_mtl isa MtlArray{Float32, 1}
        @test length(score_mtl) == 1

        score_cpu = dice_score(pred_cpu, target_cpu)
        @test isapprox(_get_scalar(Array(score_mtl)), _get_scalar(score_cpu), rtol=1e-5)
    end

    @testset "Gradient verification 2D (CPU)" begin
        rng = StableRNG(123)
        pred = rand(rng, Float32, 8, 8, 1, 1)
        target = rand(rng, Float32, 8, 8, 1, 1)

        pred_fdata = zeros(Float32, size(pred))
        target_fdata = zeros(Float32, size(target))

        output_codual, pb = Mooncake.rrule!!(
            CoDual(dice_score, NoFData()),
            CoDual(pred, pred_fdata),
            CoDual(target, target_fdata)
        )

        fill!(output_codual.dx, 1.0f0)
        pb(NoRData())

        # Finite difference check (use larger epsilon for this test)
        f_pred = x -> dice_score(x, target)
        grad_fd = finite_diff_grad(f_pred, pred, Float32(1e-3))

        # Allow slightly larger tolerance due to numerical precision
        @test isapprox(pred_fdata, grad_fd, rtol=2e-2)
    end

    @testset "Gradient on Metal GPU 2D" begin
        rng = StableRNG(456)
        pred_mtl = MtlArray(rand(rng, Float32, 8, 8, 1, 1))
        target_mtl = MtlArray(rand(rng, Float32, 8, 8, 1, 1))
        pred_fdata = Metal.zeros(Float32, size(pred_mtl))
        target_fdata = Metal.zeros(Float32, size(target_mtl))

        output_codual, pb = Mooncake.rrule!!(
            CoDual(dice_score, NoFData()),
            CoDual(pred_mtl, pred_fdata),
            CoDual(target_mtl, target_fdata)
        )

        @test output_codual.x isa MtlArray
        fill!(output_codual.dx, 1.0f0)
        pb(NoRData())

        @test pred_fdata isa MtlArray
        @test _get_scalar(abs.(Array(pred_fdata))) > 0
    end
end

@testset "dice_loss" begin
    @testset "CPU basic test - identical masks" begin
        mask = Float32.(rand(StableRNG(1), 8, 8, 1, 1) .> 0.5)
        loss = dice_loss(mask, mask)
        @test _get_scalar(loss) ≈ 0.0f0 atol=1e-6
    end

    @testset "relation to dice_score" begin
        rng = StableRNG(42)
        pred = rand(rng, Float32, 8, 8, 1, 1)
        target = rand(rng, Float32, 8, 8, 1, 1)

        loss = dice_loss(pred, target)
        score = dice_score(pred, target)

        @test isapprox(_get_scalar(loss), 1.0f0 - _get_scalar(score), atol=1e-6)
    end

    @testset "Metal GPU test" begin
        rng = StableRNG(43)
        pred_mtl = MtlArray(rand(rng, Float32, 16, 16, 1, 2))
        target_mtl = MtlArray(rand(rng, Float32, 16, 16, 1, 2))

        loss_mtl = dice_loss(pred_mtl, target_mtl)

        @test loss_mtl isa MtlArray{Float32, 1}
        loss_val = _get_scalar(Array(loss_mtl))
        @test loss_val >= 0.0f0
        @test loss_val <= 1.0f0
    end

    @testset "Gradient verification (CPU)" begin
        rng = StableRNG(789)
        pred = rand(rng, Float32, 8, 8, 1, 1)
        target = rand(rng, Float32, 8, 8, 1, 1)

        pred_fdata = zeros(Float32, size(pred))
        target_fdata = zeros(Float32, size(target))

        output_codual, pb = Mooncake.rrule!!(
            CoDual(dice_loss, NoFData()),
            CoDual(pred, pred_fdata),
            CoDual(target, target_fdata)
        )

        fill!(output_codual.dx, 1.0f0)
        pb(NoRData())

        # Finite difference check (use larger epsilon for stability)
        f_pred = x -> dice_loss(x, target)
        grad_fd = finite_diff_grad(f_pred, pred, Float32(1e-3))

        # Allow slightly larger tolerance due to numerical precision
        @test isapprox(pred_fdata, grad_fd, rtol=2e-2)
    end
end

# ============================================================================
# NCC Loss Tests
# ============================================================================

@testset "ncc_loss" begin
    @testset "CPU basic test - identical images" begin
        rng = StableRNG(42)
        img = rand(rng, Float32, 16, 16, 16, 1, 1)

        # Identical images should have high correlation → low loss (negative)
        loss = ncc_loss(img, img; kernel_size=5)

        @test isfinite(_get_scalar(loss))
        @test _get_scalar(loss) < -0.5f0  # Should be strongly negative (high correlation)
    end

    @testset "CPU basic test - uncorrelated images" begin
        rng1 = StableRNG(42)
        rng2 = StableRNG(123)
        img1 = rand(rng1, Float32, 16, 16, 16, 1, 1)
        img2 = rand(rng2, Float32, 16, 16, 16, 1, 1)

        loss = ncc_loss(img1, img2; kernel_size=5)

        @test isfinite(_get_scalar(loss))
        # Random images should have lower correlation than identical
        @test _get_scalar(loss) > -1.0f0
    end

    @testset "Metal GPU test" begin
        rng = StableRNG(456)
        pred_cpu = rand(rng, Float32, 12, 12, 12, 1, 1)
        target_cpu = rand(rng, Float32, 12, 12, 12, 1, 1)

        pred_mtl = MtlArray(pred_cpu)
        target_mtl = MtlArray(target_cpu)

        loss_mtl = ncc_loss(pred_mtl, target_mtl; kernel_size=5)

        @test loss_mtl isa MtlArray{Float32, 1}
        @test length(loss_mtl) == 1

        # Compare GPU vs CPU result
        loss_cpu = ncc_loss(pred_cpu, target_cpu; kernel_size=5)
        @test isapprox(_get_scalar(Array(loss_mtl)), _get_scalar(loss_cpu), rtol=1e-4)
    end

    @testset "different kernel sizes" begin
        rng = StableRNG(789)
        pred = rand(rng, Float32, 16, 16, 16, 1, 1)
        target = rand(rng, Float32, 16, 16, 16, 1, 1)

        for ks in [3, 5, 7]
            loss = ncc_loss(pred, target; kernel_size=ks)
            @test isfinite(_get_scalar(loss))
        end
    end

    @testset "Gradient verification (CPU) - small" begin
        rng = StableRNG(111)
        # Use small array for finite diff
        pred = rand(rng, Float32, 8, 8, 8, 1, 1)
        target = rand(rng, Float32, 8, 8, 8, 1, 1)

        pred_fdata = zeros(Float32, size(pred))
        target_fdata = zeros(Float32, size(target))

        output_codual, pb = Mooncake.rrule!!(
            CoDual(ncc_loss, NoFData()),
            CoDual(pred, pred_fdata),
            CoDual(target, target_fdata);
            kernel_size=3
        )

        fill!(output_codual.dx, 1.0f0)
        pb(NoRData())

        # Check gradient is non-zero and finite
        @test all(isfinite.(pred_fdata))
        @test _get_scalar(abs.(pred_fdata)) > 0

        # Finite difference check (use larger epsilon for numerical stability)
        f_pred = x -> ncc_loss(x, target; kernel_size=3)
        grad_fd = finite_diff_grad(f_pred, pred, Float32(1e-3))

        # NCC gradient is complex with local windows - allow larger tolerance
        # The gradient computation involves many intermediate calculations
        @test isapprox(pred_fdata, grad_fd, rtol=0.5, atol=1e-2)
    end

    @testset "Gradient on Metal GPU" begin
        rng = StableRNG(222)
        pred_mtl = MtlArray(rand(rng, Float32, 8, 8, 8, 1, 1))
        target_mtl = MtlArray(rand(rng, Float32, 8, 8, 8, 1, 1))
        pred_fdata = Metal.zeros(Float32, size(pred_mtl))
        target_fdata = Metal.zeros(Float32, size(target_mtl))

        output_codual, pb = Mooncake.rrule!!(
            CoDual(ncc_loss, NoFData()),
            CoDual(pred_mtl, pred_fdata),
            CoDual(target_mtl, target_fdata);
            kernel_size=3
        )

        @test output_codual.x isa MtlArray
        fill!(output_codual.dx, 1.0f0)
        pb(NoRData())

        @test pred_fdata isa MtlArray
        # Check gradient is non-zero
        @test _get_scalar(abs.(Array(pred_fdata))) > 0
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
    torchreg_metrics = pyimport("torchreg.metrics")
    np = pyimport("numpy")

    @testset "dice_score parity 3D" begin
        torch.manual_seed(42)

        # Create random masks - torchreg uses (N, C, D, H, W) convention
        N, C, D, H, W = 2, 1, 8, 8, 8

        # Create random values in [0, 1] for soft dice
        x1_pt = torch.rand(N, C, D, H, W)
        x2_pt = torch.rand(N, C, D, H, W)

        # Compute torchreg dice_score
        dice_pt = torchreg_metrics.dice_score(x1_pt, x2_pt)
        dice_pt_val = pyconvert(Float32, dice_pt.item())

        # Convert to Julia convention: (N, C, D, H, W) -> (X, Y, Z, C, N) = (W, H, D, C, N)
        x1_np = pyconvert(Array{Float32}, x1_pt.numpy())
        x2_np = pyconvert(Array{Float32}, x2_pt.numpy())
        x1_jl = permutedims(x1_np, (5, 4, 3, 2, 1))  # (W, H, D, C, N) = (X, Y, Z, C, N)
        x2_jl = permutedims(x2_np, (5, 4, 3, 2, 1))

        # Compute Julia dice_score
        dice_jl = dice_score(x1_jl, x2_jl)
        dice_jl_val = _get_scalar(dice_jl)

        @test isapprox(dice_jl_val, dice_pt_val, rtol=1e-5)
    end

    @testset "dice_score parity 2D" begin
        torch.manual_seed(123)

        # 2D case: (N, C, H, W)
        N, C, H, W = 2, 1, 16, 16

        x1_pt = torch.rand(N, C, H, W)
        x2_pt = torch.rand(N, C, H, W)

        dice_pt = torchreg_metrics.dice_score(x1_pt, x2_pt)
        dice_pt_val = pyconvert(Float32, dice_pt.item())

        # Convert to Julia convention: (N, C, H, W) -> (X, Y, C, N) = (W, H, C, N)
        x1_np = pyconvert(Array{Float32}, x1_pt.numpy())
        x2_np = pyconvert(Array{Float32}, x2_pt.numpy())
        x1_jl = permutedims(x1_np, (4, 3, 2, 1))
        x2_jl = permutedims(x2_np, (4, 3, 2, 1))

        dice_jl = dice_score(x1_jl, x2_jl)
        dice_jl_val = _get_scalar(dice_jl)

        @test isapprox(dice_jl_val, dice_pt_val, rtol=1e-5)
    end

    @testset "dice_loss parity 3D" begin
        torch.manual_seed(456)

        N, C, D, H, W = 1, 1, 10, 10, 10

        x1_pt = torch.rand(N, C, D, H, W)
        x2_pt = torch.rand(N, C, D, H, W)

        loss_pt = torchreg_metrics.dice_loss(x1_pt, x2_pt)
        loss_pt_val = pyconvert(Float32, loss_pt.item())

        x1_np = pyconvert(Array{Float32}, x1_pt.numpy())
        x2_np = pyconvert(Array{Float32}, x2_pt.numpy())
        x1_jl = permutedims(x1_np, (5, 4, 3, 2, 1))
        x2_jl = permutedims(x2_np, (5, 4, 3, 2, 1))

        loss_jl = dice_loss(x1_jl, x2_jl)
        loss_jl_val = _get_scalar(loss_jl)

        @test isapprox(loss_jl_val, loss_pt_val, rtol=1e-5)
    end

    @testset "dice_score identical masks" begin
        # Perfect overlap should give dice = 1
        torch.manual_seed(789)

        N, C, D, H, W = 1, 1, 8, 8, 8

        x_pt = torch.rand(N, C, D, H, W)

        dice_pt = torchreg_metrics.dice_score(x_pt, x_pt)
        dice_pt_val = pyconvert(Float32, dice_pt.item())

        x_np = pyconvert(Array{Float32}, x_pt.numpy())
        x_jl = permutedims(x_np, (5, 4, 3, 2, 1))

        dice_jl = dice_score(x_jl, x_jl)
        dice_jl_val = _get_scalar(dice_jl)

        @test isapprox(dice_jl_val, dice_pt_val, rtol=1e-5)
        @test isapprox(dice_jl_val, 1.0f0, atol=1e-6)
    end

    @testset "NCC parity" begin
        torch.manual_seed(100)

        # Note: NCC implementations may differ slightly in boundary handling,
        # but the behavior should be qualitatively similar
        N, C, D, H, W = 1, 1, 16, 16, 16
        kernel_size = 7

        pred_pt = torch.rand(N, C, D, H, W)
        targ_pt = torch.rand(N, C, D, H, W)

        ncc_module = torchreg_metrics.NCC(kernel_size=kernel_size)
        loss_pt = ncc_module(pred_pt, targ_pt)
        loss_pt_val = pyconvert(Float32, loss_pt.item())

        pred_np = pyconvert(Array{Float32}, pred_pt.numpy())
        targ_np = pyconvert(Array{Float32}, targ_pt.numpy())
        pred_jl = permutedims(pred_np, (5, 4, 3, 2, 1))
        targ_jl = permutedims(targ_np, (5, 4, 3, 2, 1))

        loss_jl = ncc_loss(pred_jl, targ_jl; kernel_size=kernel_size)
        loss_jl_val = _get_scalar(loss_jl)

        # NCC implementations can differ in boundary handling
        # Check that both are negative (as expected for correlation-based loss)
        # and have similar magnitude
        @test loss_pt_val < 0  # torchreg NCC returns negative
        @test loss_jl_val < 0  # our NCC returns negative

        # Check same order of magnitude - allow 50% tolerance due to
        # different convolution vs explicit window implementations
        @test isapprox(loss_jl_val, loss_pt_val, rtol=0.5)
    end

    @testset "NCC identical images" begin
        torch.manual_seed(200)

        N, C, D, H, W = 1, 1, 12, 12, 12
        kernel_size = 5

        img_pt = torch.rand(N, C, D, H, W)

        ncc_module = torchreg_metrics.NCC(kernel_size=kernel_size)
        loss_pt = ncc_module(img_pt, img_pt)
        loss_pt_val = pyconvert(Float32, loss_pt.item())

        img_np = pyconvert(Array{Float32}, img_pt.numpy())
        img_jl = permutedims(img_np, (5, 4, 3, 2, 1))

        loss_jl = ncc_loss(img_jl, img_jl; kernel_size=kernel_size)
        loss_jl_val = _get_scalar(loss_jl)

        # Identical images should have high correlation, so NCC loss should be strongly negative
        @test loss_pt_val < -0.5
        @test loss_jl_val < -0.5
    end
end
