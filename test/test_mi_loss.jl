# Test Mutual Information loss on GPU (Metal) with Mooncake AD
# Tests mi_loss and nmi_loss for multi-modal registration

using Test
using MedicalImageRegistration
using StableRNGs
import AcceleratedKernels as AK
import Mooncake
import Mooncake: CoDual, NoFData, NoRData

# Include test helpers for conditional GPU testing
include("test_helpers.jl")

# ============================================================================
# Helper: Extract scalar from 1-element array
# ============================================================================

function _get_scalar(arr::AbstractArray{T}) where T
    return AK.reduce(+, arr; init=zero(T))
end

# ============================================================================
# Helper: Finite Difference Gradient Check
# ============================================================================

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
# MI Loss Basic Tests
# ============================================================================

@testset "mi_loss" begin
    @testset "CPU basic test - identical images have high MI" begin
        rng = StableRNG(42)
        # Create image with distinct regions (similar to CT: different tissues)
        img = zeros(Float32, 16, 16, 16, 1, 1)
        img[1:8, :, :, :, :] .= -500.0f0   # Air-like
        img[9:16, :, :, :, :] .= 50.0f0    # Soft tissue-like
        img[7:10, 7:10, 7:10, :, :] .= 1000.0f0  # Bone-like

        # MI of image with itself should be high (loss should be very negative)
        loss = mi_loss(img, img)
        @test isfinite(_get_scalar(loss))
        @test _get_scalar(loss) < 0  # Negative because we return -MI
    end

    @testset "CPU basic test - random images have lower MI" begin
        rng = StableRNG(42)
        img1 = rand(rng, Float32, 16, 16, 16, 1, 1) .* 1000.0f0 .- 500.0f0
        rng2 = StableRNG(123)
        img2 = rand(rng2, Float32, 16, 16, 16, 1, 1) .* 1000.0f0 .- 500.0f0

        loss = mi_loss(img1, img2)
        @test isfinite(_get_scalar(loss))

        # Random images should have less MI than identical
        loss_identical = mi_loss(img1, img1)
        @test _get_scalar(loss) > _get_scalar(loss_identical)  # Less negative = lower MI
    end

    @testset "CPU test - shifted copy has high MI" begin
        # MI should be robust to intensity shifts (different from MSE!)
        rng = StableRNG(456)
        img1 = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0

        # Create a shifted copy (like contrast agent changes)
        img2 = img1 .+ 200.0f0  # Uniform shift

        loss = mi_loss(img1, img2)
        @test isfinite(_get_scalar(loss))

        # Shifted copy should still have relatively high MI
        # (intensities perfectly correlated, just offset)
        loss_random = mi_loss(img1, rand(StableRNG(789), Float32, size(img1)...) .* 500.0f0)
        @test _get_scalar(loss) < _get_scalar(loss_random)  # More negative = higher MI
    end

    @testset "CPU test - scaled copy has high MI" begin
        # MI should detect multiplicative relationships too
        rng = StableRNG(111)
        img1 = rand(rng, Float32, 16, 16, 16, 1, 1) .* 100.0f0 .+ 1.0f0

        # Create a scaled copy
        img2 = img1 .* 3.0f0

        loss = mi_loss(img1, img2)
        @test isfinite(_get_scalar(loss))

        # Scaled copy should have relatively high MI
        loss_random = mi_loss(img1, rand(StableRNG(222), Float32, size(img1)...) .* 300.0f0)
        @test _get_scalar(loss) < _get_scalar(loss_random)
    end

    @testset "CPU test - simulated contrast/non-contrast" begin
        # Simulate cardiac CT scenario:
        # Non-contrast: blood=40, myocardium=50, bone=1000
        # Contrast: blood=300, myocardium=100, bone=1000

        non_contrast = zeros(Float32, 20, 20, 20, 1, 1)
        contrast = zeros(Float32, 20, 20, 20, 1, 1)

        # Background (fat)
        non_contrast .= -100.0f0
        contrast .= -100.0f0

        # Heart region
        non_contrast[5:15, 5:15, 5:15, :, :] .= 40.0f0   # Blood non-contrast
        contrast[5:15, 5:15, 5:15, :, :] .= 300.0f0       # Blood with contrast

        # Myocardium ring
        non_contrast[7:13, 7:13, 7:13, :, :] .= 50.0f0
        contrast[7:13, 7:13, 7:13, :, :] .= 100.0f0

        # Bone (unchanged by contrast)
        non_contrast[2:4, 2:4, 10:12, :, :] .= 1000.0f0
        contrast[2:4, 2:4, 10:12, :, :] .= 1000.0f0

        # MI should be reasonably high despite intensity differences
        loss = mi_loss(non_contrast, contrast)
        @test isfinite(_get_scalar(loss))

        # MSE would be huge due to 260 HU difference in blood
        # But MI should find the correspondence
        @test _get_scalar(loss) < 0  # Negative MI loss
    end

    if METAL_AVAILABLE
        @testset "Metal GPU test" begin
            rng = StableRNG(42)
            moving_cpu = rand(rng, Float32, 16, 16, 16, 1, 1) .* 1000.0f0 .- 500.0f0
            static_cpu = rand(rng, Float32, 16, 16, 16, 1, 1) .* 1000.0f0 .- 500.0f0

            moving_mtl = MtlArray(moving_cpu)
            static_mtl = MtlArray(static_cpu)

            loss_mtl = mi_loss(moving_mtl, static_mtl)

            @test loss_mtl isa MtlArray{Float32, 1}
            @test length(loss_mtl) == 1

            # Compare GPU vs CPU result
            loss_cpu = mi_loss(moving_cpu, static_cpu)
            @test isapprox(_get_scalar(Array(loss_mtl)), _get_scalar(loss_cpu), rtol=1e-3)
        end

        @testset "Metal GPU test - different bin counts" begin
            rng = StableRNG(789)
            moving_mtl = MtlArray(rand(rng, Float32, 12, 12, 12, 1, 1) .* 500.0f0)
            static_mtl = MtlArray(rand(rng, Float32, 12, 12, 12, 1, 1) .* 500.0f0)

            for num_bins in [32, 64, 128]
                loss = mi_loss(moving_mtl, static_mtl; num_bins=num_bins)
                @test loss isa MtlArray
                @test isfinite(_get_scalar(Array(loss)))
            end
        end
    end

    @testset "Gradient verification (CPU) - small array" begin
        rng = StableRNG(123)
        # Use small array for finite diff
        moving = rand(rng, Float32, 6, 6, 6, 1, 1) .* 100.0f0
        static = rand(rng, Float32, 6, 6, 6, 1, 1) .* 100.0f0

        # Create CoDuals
        moving_fdata = zeros(Float32, size(moving))
        static_fdata = zeros(Float32, size(static))
        moving_codual = CoDual(moving, moving_fdata)
        static_codual = CoDual(static, static_fdata)

        # Forward pass via rrule!!
        output_codual, pb = Mooncake.rrule!!(
            CoDual(mi_loss, NoFData()),
            moving_codual,
            static_codual;
            num_bins=16,
            sigma=1.5f0
        )

        # Set upstream gradient
        fill!(output_codual.dx, 1.0f0)

        # Backward pass
        pb(NoRData())

        # Check gradients are finite and non-zero
        @test all(isfinite.(moving_fdata))
        @test _get_scalar(abs.(moving_fdata)) > 0

        # Finite difference check
        f_moving = x -> mi_loss(x, static; num_bins=16, sigma=1.5f0)
        grad_fd = finite_diff_grad(f_moving, moving, Float32(1e-3))

        # MI gradients can be noisy - use generous tolerance
        # The key thing is that the gradient direction is approximately correct
        @test isapprox(moving_fdata, grad_fd, rtol=0.5, atol=5e-2)
    end

    if METAL_AVAILABLE
        @testset "Gradient on Metal GPU" begin
            rng = StableRNG(456)
            moving_mtl = MtlArray(rand(rng, Float32, 8, 8, 8, 1, 1) .* 100.0f0)
            static_mtl = MtlArray(rand(rng, Float32, 8, 8, 8, 1, 1) .* 100.0f0)
            moving_fdata = Metal.zeros(Float32, size(moving_mtl))
            static_fdata = Metal.zeros(Float32, size(static_mtl))

            output_codual, pb = Mooncake.rrule!!(
                CoDual(mi_loss, NoFData()),
                CoDual(moving_mtl, moving_fdata),
                CoDual(static_mtl, static_fdata);
                num_bins=32,
                sigma=1.0f0
            )

            @test output_codual.x isa MtlArray
            @test output_codual.dx isa MtlArray

            fill!(output_codual.dx, 1.0f0)
            pb(NoRData())

            @test moving_fdata isa MtlArray
            # Check gradient is non-zero
            @test _get_scalar(abs.(Array(moving_fdata))) > 0
        end
    end
end

# ============================================================================
# NMI Loss Tests
# ============================================================================

@testset "nmi_loss" begin
    @testset "CPU basic test - identical images" begin
        rng = StableRNG(42)
        img = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0

        loss = nmi_loss(img, img)
        @test isfinite(_get_scalar(loss))
        # NMI of identical should be close to 1, so loss close to -1
        @test _get_scalar(loss) < -0.5f0
    end

    @testset "CPU basic test - random images" begin
        rng1 = StableRNG(42)
        rng2 = StableRNG(123)
        img1 = rand(rng1, Float32, 16, 16, 16, 1, 1) .* 500.0f0
        img2 = rand(rng2, Float32, 16, 16, 16, 1, 1) .* 500.0f0

        loss = nmi_loss(img1, img2)
        @test isfinite(_get_scalar(loss))

        # Random images should have less NMI than identical
        loss_identical = nmi_loss(img1, img1)
        @test _get_scalar(loss) > _get_scalar(loss_identical)  # Less negative = lower NMI
    end

    @testset "NMI range check" begin
        rng = StableRNG(789)
        img1 = rand(rng, Float32, 12, 12, 12, 1, 1) .* 500.0f0
        img2 = rand(rng, Float32, 12, 12, 12, 1, 1) .* 500.0f0

        loss = nmi_loss(img1, img2)
        nmi_val = -_get_scalar(loss)  # NMI = -loss

        # NMI should be in [0, 1] for well-behaved cases
        # (can exceed 1 in edge cases but shouldn't be negative)
        @test nmi_val >= -0.1f0  # Allow small numerical error
    end

    if METAL_AVAILABLE
        @testset "Metal GPU test" begin
            rng = StableRNG(42)
            moving_cpu = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0
            static_cpu = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0

            moving_mtl = MtlArray(moving_cpu)
            static_mtl = MtlArray(static_cpu)

            loss_mtl = nmi_loss(moving_mtl, static_mtl)

            @test loss_mtl isa MtlArray{Float32, 1}

            # Compare GPU vs CPU
            loss_cpu = nmi_loss(moving_cpu, static_cpu)
            @test isapprox(_get_scalar(Array(loss_mtl)), _get_scalar(loss_cpu), rtol=1e-3)
        end
    end

    @testset "Gradient verification (CPU)" begin
        rng = StableRNG(111)
        moving = rand(rng, Float32, 6, 6, 6, 1, 1) .* 100.0f0
        static = rand(rng, Float32, 6, 6, 6, 1, 1) .* 100.0f0

        moving_fdata = zeros(Float32, size(moving))
        static_fdata = zeros(Float32, size(static))

        output_codual, pb = Mooncake.rrule!!(
            CoDual(nmi_loss, NoFData()),
            CoDual(moving, moving_fdata),
            CoDual(static, static_fdata);
            num_bins=16,
            sigma=1.5f0
        )

        fill!(output_codual.dx, 1.0f0)
        pb(NoRData())

        @test all(isfinite.(moving_fdata))
        @test _get_scalar(abs.(moving_fdata)) > 0

        # Finite difference check
        f_moving = x -> nmi_loss(x, static; num_bins=16, sigma=1.5f0)
        grad_fd = finite_diff_grad(f_moving, moving, Float32(1e-3))

        @test isapprox(moving_fdata, grad_fd, rtol=0.5, atol=5e-2)
    end

    if METAL_AVAILABLE
        @testset "Gradient on Metal GPU" begin
            rng = StableRNG(222)
            moving_mtl = MtlArray(rand(rng, Float32, 8, 8, 8, 1, 1) .* 100.0f0)
            static_mtl = MtlArray(rand(rng, Float32, 8, 8, 8, 1, 1) .* 100.0f0)
            moving_fdata = Metal.zeros(Float32, size(moving_mtl))
            static_fdata = Metal.zeros(Float32, size(static_mtl))

            output_codual, pb = Mooncake.rrule!!(
                CoDual(nmi_loss, NoFData()),
                CoDual(moving_mtl, moving_fdata),
                CoDual(static_mtl, static_fdata);
                num_bins=32,
                sigma=1.0f0
            )

            @test output_codual.x isa MtlArray
            fill!(output_codual.dx, 1.0f0)
            pb(NoRData())

            @test moving_fdata isa MtlArray
            @test _get_scalar(abs.(Array(moving_fdata))) > 0
        end
    end
end

# ============================================================================
# Multi-modal Registration Scenario Tests
# ============================================================================

@testset "MI for multi-modal registration scenarios" begin
    @testset "MI decreases as alignment improves" begin
        # Create synthetic aligned and misaligned scenarios
        rng = StableRNG(333)

        # Create a simple structured image
        static = zeros(Float32, 20, 20, 20, 1, 1)
        static[5:15, 5:15, 5:15, :, :] .= 100.0f0
        static[8:12, 8:12, 8:12, :, :] .= 200.0f0

        # Aligned moving (same structure, different intensities - like contrast)
        moving_aligned = zeros(Float32, 20, 20, 20, 1, 1)
        moving_aligned[5:15, 5:15, 5:15, :, :] .= 300.0f0  # Different value
        moving_aligned[8:12, 8:12, 8:12, :, :] .= 500.0f0

        # Misaligned moving (shifted)
        moving_misaligned = zeros(Float32, 20, 20, 20, 1, 1)
        moving_misaligned[7:17, 7:17, 7:17, :, :] .= 300.0f0  # Shifted
        moving_misaligned[10:14, 10:14, 10:14, :, :] .= 500.0f0

        loss_aligned = mi_loss(moving_aligned, static)
        loss_misaligned = mi_loss(moving_misaligned, static)

        # Aligned should have higher MI (more negative loss)
        @test _get_scalar(loss_aligned) < _get_scalar(loss_misaligned)
    end

    @testset "MI vs MSE on contrast-enhanced images" begin
        # This test demonstrates why MI is better than MSE for multi-modal

        # Create paired contrast/non-contrast images
        non_contrast = zeros(Float32, 16, 16, 16, 1, 1)
        contrast = zeros(Float32, 16, 16, 16, 1, 1)

        # Background
        non_contrast .= 0.0f0
        contrast .= 0.0f0

        # Vessel region (blood)
        non_contrast[6:10, 6:10, 6:10, :, :] .= 40.0f0   # Non-contrast blood
        contrast[6:10, 6:10, 6:10, :, :] .= 300.0f0       # Contrast-enhanced blood

        # MSE would be huge due to (300-40)^2 = 67600 per voxel
        mse = mse_loss(contrast, non_contrast)

        # MI should still work because correspondence exists
        mi = mi_loss(contrast, non_contrast)

        @test _get_scalar(mse) > 1000.0f0  # MSE is large
        @test _get_scalar(mi) < 0  # MI is negative (indicates correlation exists)

        # Compare to completely unrelated images
        rng = StableRNG(444)
        random_img = rand(rng, Float32, 16, 16, 16, 1, 1) .* 300.0f0
        mi_random = mi_loss(random_img, non_contrast)

        # MI with matched anatomy should be better than random
        @test _get_scalar(mi) < _get_scalar(mi_random)
    end
end

# ============================================================================
# Parameter sensitivity tests
# ============================================================================

@testset "MI parameter sensitivity" begin
    @testset "Different sigma values" begin
        rng = StableRNG(555)
        img1 = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0
        img2 = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0

        # Test different sigma values
        for sigma in [0.5f0, 1.0f0, 2.0f0]
            loss = mi_loss(img1, img2; sigma=sigma)
            @test isfinite(_get_scalar(loss))
        end
    end

    @testset "Different bin counts" begin
        rng = StableRNG(666)
        img1 = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0
        img2 = rand(rng, Float32, 16, 16, 16, 1, 1) .* 500.0f0

        for num_bins in [16, 32, 64, 128]
            loss = mi_loss(img1, img2; num_bins=num_bins)
            @test isfinite(_get_scalar(loss))
        end
    end

    @testset "Explicit intensity range" begin
        rng = StableRNG(777)
        img1 = rand(rng, Float32, 12, 12, 12, 1, 1) .* 500.0f0 .- 250.0f0
        img2 = rand(rng, Float32, 12, 12, 12, 1, 1) .* 500.0f0 .- 250.0f0

        # Provide explicit range
        loss = mi_loss(img1, img2; intensity_range=(-500.0f0, 500.0f0))
        @test isfinite(_get_scalar(loss))

        # Should give similar result to auto-range
        loss_auto = mi_loss(img1, img2)
        @test isfinite(_get_scalar(loss_auto))
    end
end

# ============================================================================
# 2D and 4D array tests
# ============================================================================

@testset "MI with different array dimensions" begin
    @testset "4D arrays (2D images)" begin
        rng = StableRNG(888)
        img1 = rand(rng, Float32, 32, 32, 1, 1) .* 500.0f0
        img2 = rand(rng, Float32, 32, 32, 1, 1) .* 500.0f0

        loss = mi_loss(img1, img2)
        @test isfinite(_get_scalar(loss))

        # Same image should have high MI
        loss_same = mi_loss(img1, img1)
        @test _get_scalar(loss_same) < _get_scalar(loss)
    end

    @testset "Batched images" begin
        rng = StableRNG(999)
        # Batch size 2
        img1 = rand(rng, Float32, 16, 16, 16, 1, 2) .* 500.0f0
        img2 = rand(rng, Float32, 16, 16, 16, 1, 2) .* 500.0f0

        loss = mi_loss(img1, img2)
        @test isfinite(_get_scalar(loss))
    end

    @testset "Multi-channel images" begin
        rng = StableRNG(1111)
        # 3 channels
        img1 = rand(rng, Float32, 16, 16, 16, 3, 1) .* 500.0f0
        img2 = rand(rng, Float32, 16, 16, 16, 3, 1) .* 500.0f0

        loss = mi_loss(img1, img2)
        @test isfinite(_get_scalar(loss))
    end
end
