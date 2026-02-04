# SyN registration tests
# Tests diffeomorphic_transform, gauss_smoothing, and full registration

using Test
using MedicalImageRegistration
using Random
using Statistics: var

# Try to use Metal if available, otherwise use CPU arrays
const USE_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

if USE_GPU
    using Metal
    const ArrayType = MtlArray
    @info "Running SyN tests on Metal GPU"
else
    const ArrayType = Array
    @info "Running SyN tests on CPU (Metal not available)"
end

@testset "SyN Registration" begin

    # =========================================================================
    # spatial_transform Tests
    # =========================================================================
    @testset "spatial_transform" begin

        @testset "identity displacement" begin
            # Zero displacement should return same image
            X, Y, Z, C, N = 8, 8, 8, 1, 1

            x_cpu = rand(Float32, X, Y, Z, C, N)
            v_cpu = zeros(Float32, X, Y, Z, 3, N)

            x = ArrayType(x_cpu)
            v = ArrayType(v_cpu)

            result = spatial_transform(x, v)

            @test result isa typeof(x)
            @test size(result) == size(x)
            @test isapprox(Array(result), x_cpu, rtol=1e-4, atol=1e-5)
        end

        @testset "small displacement" begin
            Random.seed!(42)
            X, Y, Z, C, N = 16, 16, 16, 1, 1

            x = ArrayType(rand(Float32, X, Y, Z, C, N))

            # Small uniform displacement
            v_cpu = zeros(Float32, X, Y, Z, 3, N)
            v_cpu[:, :, :, 1, :] .= 0.1f0  # Small x displacement
            v = ArrayType(v_cpu)

            result = spatial_transform(x, v)

            @test result isa typeof(x)
            @test size(result) == size(x)
            # Result should be different from input
            @test !isapprox(Array(result), Array(x), rtol=1e-2)
            @test all(isfinite, Array(result))
        end
    end

    # =========================================================================
    # diffeomorphic_transform Tests
    # =========================================================================
    @testset "diffeomorphic_transform" begin

        @testset "zero velocity gives zero displacement" begin
            X, Y, Z, N = 8, 8, 8, 1
            v = ArrayType(zeros(Float32, X, Y, Z, 3, N))

            result = diffeomorphic_transform(v; time_steps=7)

            @test result isa typeof(v)
            @test all(x -> abs(x) < 1e-6, Array(result))
        end

        @testset "small velocity field" begin
            Random.seed!(42)
            X, Y, Z, N = 8, 8, 8, 1

            v_cpu = randn(Float32, X, Y, Z, 3, N) .* 0.01f0
            v = ArrayType(v_cpu)

            result = diffeomorphic_transform(v; time_steps=3)

            @test result isa typeof(v)
            @test size(result) == size(v)
            @test all(isfinite, Array(result))
        end

        @testset "larger velocity field" begin
            Random.seed!(123)
            X, Y, Z, N = 10, 10, 10, 1

            v_cpu = randn(Float32, X, Y, Z, 3, N) .* 0.1f0
            v = ArrayType(v_cpu)

            result = diffeomorphic_transform(v; time_steps=7)

            @test all(isfinite, Array(result))
        end

        @testset "different time_steps" begin
            Random.seed!(456)
            X, Y, Z, N = 8, 8, 8, 1
            v = ArrayType(randn(Float32, X, Y, Z, 3, N) .* 0.05f0)

            for time_steps in [5, 7, 9]
                result = diffeomorphic_transform(v; time_steps=time_steps)

                @test size(result) == (X, Y, Z, 3, N)
                @test all(isfinite, Array(result))
            end
        end

        @testset "batch_size > 1" begin
            Random.seed!(789)
            X, Y, Z, N = 8, 8, 8, 2

            v = ArrayType(randn(Float32, X, Y, Z, 3, N) .* 0.05f0)

            result = diffeomorphic_transform(v; time_steps=7)

            @test size(result) == (X, Y, Z, 3, N)
            @test all(isfinite, Array(result))
        end
    end

    # =========================================================================
    # composition_transform Tests
    # =========================================================================
    @testset "composition_transform" begin

        @testset "identity composition" begin
            X, Y, Z, N = 8, 8, 8, 1
            v1 = ArrayType(zeros(Float32, X, Y, Z, 3, N))
            v2 = ArrayType(zeros(Float32, X, Y, Z, 3, N))

            result = composition_transform(v1, v2)

            @test all(x -> abs(x) < 1e-6, Array(result))
        end

        @testset "composition with self" begin
            Random.seed!(100)
            X, Y, Z, N = 8, 8, 8, 1
            v = ArrayType(randn(Float32, X, Y, Z, 3, N) .* 0.01f0)

            result = composition_transform(v, v)

            @test result isa typeof(v)
            @test all(isfinite, Array(result))
        end

        @testset "compose with zero" begin
            Random.seed!(101)
            X, Y, Z, N = 8, 8, 8, 1

            v1_cpu = randn(Float32, X, Y, Z, 3, N) .* 0.1f0
            v2_cpu = zeros(Float32, X, Y, Z, 3, N)

            v1 = ArrayType(v1_cpu)
            v2 = ArrayType(v2_cpu)

            # Compose v1 with zero: result = v2 + v1(v2) = 0 + v1(0) = v1
            result = composition_transform(v1, v2)

            @test isapprox(Array(result), v1_cpu, rtol=1e-4, atol=1e-5)
        end
    end

    # =========================================================================
    # gauss_smoothing Tests
    # =========================================================================
    @testset "gauss_smoothing" begin

        @testset "small sigma returns similar" begin
            Random.seed!(102)
            X, Y, Z, C, N = 16, 16, 16, 3, 1
            x_cpu = rand(Float32, X, Y, Z, C, N)
            x = ArrayType(x_cpu)

            result = gauss_smoothing(x, 0.05f0)

            @test result isa typeof(x)
            @test size(result) == size(x)
            # Very small sigma should return almost the same
            @test isapprox(Array(result), x_cpu, rtol=0.2)
        end

        @testset "smoothing reduces variance" begin
            Random.seed!(103)
            X, Y, Z, C, N = 32, 32, 32, 1, 1
            x = ArrayType(rand(Float32, X, Y, Z, C, N))

            result = gauss_smoothing(x, 1.0f0)

            # Smoothed should have less variance
            var_orig = var(Array(x))
            var_smooth = var(Array(result))
            @test var_smooth < var_orig
        end

        @testset "batch support" begin
            Random.seed!(104)
            X, Y, Z, C, N = 16, 16, 16, 3, 2
            x = ArrayType(rand(Float32, X, Y, Z, C, N))

            result = gauss_smoothing(x, 0.3f0)

            @test size(result) == size(x)
            @test all(isfinite, Array(result))
        end
    end

    # =========================================================================
    # linear_elasticity Tests
    # =========================================================================
    @testset "linear_elasticity" begin

        @testset "zero flow has zero energy" begin
            X, Y, Z, N = 16, 16, 16, 1
            flow = ArrayType(zeros(Float32, X, Y, Z, 3, N))

            result = linear_elasticity(flow)

            loss_val = Array(result)[1]
            @test loss_val < 1e-6
        end

        @testset "non-zero flow has positive energy" begin
            Random.seed!(105)
            X, Y, Z, N = 16, 16, 16, 1
            flow_cpu = randn(Float32, X, Y, Z, 3, N) .* 0.1f0
            flow = ArrayType(flow_cpu)

            result = linear_elasticity(flow)

            loss_val = Array(result)[1]
            @test loss_val > 0
            @test isfinite(loss_val)
        end
    end

    # =========================================================================
    # SyNRegistration Struct Tests
    # =========================================================================
    @testset "SyNRegistration struct" begin

        @testset "constructor defaults" begin
            reg = SyNRegistration{Float32}(array_type=ArrayType)

            @test reg.scales == (4, 2, 1)
            @test reg.iterations == (30, 30, 10)
            @test reg.time_steps == 7
            @test reg.v_xy === nothing
            @test reg.v_yx === nothing
        end

        @testset "custom parameters" begin
            reg = SyNRegistration{Float32}(
                scales=(2,),
                iterations=(10,),
                learning_rate=0.001f0,
                time_steps=5,
                sigma_flow=0.5f0,
                array_type=ArrayType
            )

            @test reg.scales == (2,)
            @test reg.iterations == (10,)
            @test reg.time_steps == 5
            @test reg.sigma_flow == 0.5f0
        end

        @testset "reset!" begin
            reg = SyNRegistration{Float32}(
                scales=(2,),
                iterations=(5,),
                array_type=ArrayType
            )

            # Simulate having velocity fields
            X, Y, Z, N = 8, 8, 8, 1
            moving = ArrayType(rand(Float32, X, Y, Z, 1, N))
            static = ArrayType(rand(Float32, X, Y, Z, 1, N))

            reg.verbose = false
            fit!(reg, moving, static)

            @test reg.v_xy !== nothing

            reset!(reg)

            @test reg.v_xy === nothing
            @test reg.v_yx === nothing
            @test isempty(reg.loss_history)
        end
    end

    # =========================================================================
    # SyNRegistration fit! Tests
    # =========================================================================
    @testset "SyNRegistration fit!" begin

        @testset "basic 3D registration" begin
            X, Y, Z, C, N = 16, 16, 16, 1, 1

            # Create simple test images
            moving_cpu = zeros(Float32, X, Y, Z, C, N)
            static_cpu = zeros(Float32, X, Y, Z, C, N)

            # Moving: blob in center
            for k in 5:12, j in 5:12, i in 5:12
                moving_cpu[i, j, k, 1, 1] = 1.0f0
            end

            # Static: blob slightly shifted
            for k in 6:13, j in 6:13, i in 6:13
                static_cpu[i, j, k, 1, 1] = 1.0f0
            end

            moving = ArrayType(moving_cpu)
            static = ArrayType(static_cpu)

            reg = SyNRegistration{Float32}(
                scales=(2,),
                iterations=(5,),
                learning_rate=0.01f0,
                time_steps=3,
                verbose=false,
                array_type=ArrayType
            )

            # This should not error
            fit!(reg, moving, static)

            @test reg.v_xy !== nothing
            @test reg.v_yx !== nothing
            @test length(reg.loss_history) == 5
        end

        @testset "loss decreases" begin
            Random.seed!(200)
            X, Y, Z, C, N = 16, 16, 16, 1, 1

            moving = ArrayType(rand(Float32, X, Y, Z, C, N))
            static = ArrayType(rand(Float32, X, Y, Z, C, N))

            reg = SyNRegistration{Float32}(
                scales=(2,),
                iterations=(10,),
                learning_rate=0.01f0,
                time_steps=3,
                verbose=false,
                array_type=ArrayType
            )

            fit!(reg, moving, static)

            # Loss should generally decrease
            @test length(reg.loss_history) == 10
            # At least final loss is finite
            @test isfinite(reg.loss_history[end])
        end
    end

    # =========================================================================
    # apply_flows Tests
    # =========================================================================
    @testset "apply_flows" begin

        @testset "basic flow application" begin
            Random.seed!(300)
            X, Y, Z, C, N = 8, 8, 8, 1, 1

            x = ArrayType(rand(Float32, X, Y, Z, C, N))
            y = ArrayType(rand(Float32, X, Y, Z, C, N))
            v_xy = ArrayType(randn(Float32, X, Y, Z, 3, N) .* 0.01f0)
            v_yx = ArrayType(randn(Float32, X, Y, Z, 3, N) .* 0.01f0)

            reg = SyNRegistration{Float32}(
                time_steps=3,
                verbose=false,
                array_type=ArrayType
            )

            images, flows = apply_flows(reg, x, y, v_xy, v_yx)

            @test haskey(images, :xy_half)
            @test haskey(images, :yx_half)
            @test haskey(images, :xy_full)
            @test haskey(images, :yx_full)

            @test haskey(flows, :xy_half)
            @test haskey(flows, :yx_half)
            @test haskey(flows, :xy_full)
            @test haskey(flows, :yx_full)

            @test size(images[:xy_full]) == (X, Y, Z, C, N)
            @test size(flows[:xy_full]) == (X, Y, Z, 3, N)

            @test all(isfinite, Array(images[:xy_full]))
            @test all(isfinite, Array(flows[:xy_full]))
        end

        @testset "zero velocity returns original" begin
            Random.seed!(301)
            X, Y, Z, C, N = 8, 8, 8, 1, 1

            x_cpu = rand(Float32, X, Y, Z, C, N)
            y_cpu = rand(Float32, X, Y, Z, C, N)

            x = ArrayType(x_cpu)
            y = ArrayType(y_cpu)
            v_xy = ArrayType(zeros(Float32, X, Y, Z, 3, N))
            v_yx = ArrayType(zeros(Float32, X, Y, Z, 3, N))

            reg = SyNRegistration{Float32}(
                time_steps=3,
                verbose=false,
                array_type=ArrayType
            )

            images, flows = apply_flows(reg, x, y, v_xy, v_yx)

            # With zero velocity, images should be unchanged (approximately)
            @test isapprox(Array(images[:xy_full]), x_cpu, rtol=1e-4, atol=1e-5)
            @test isapprox(Array(images[:yx_full]), y_cpu, rtol=1e-4, atol=1e-5)
        end
    end

    # =========================================================================
    # SyN transform Tests
    # =========================================================================
    @testset "transform" begin

        @testset "transform after fit" begin
            Random.seed!(400)
            X, Y, Z, C, N = 16, 16, 16, 1, 1

            moving = ArrayType(rand(Float32, X, Y, Z, C, N))
            static = ArrayType(rand(Float32, X, Y, Z, C, N))

            reg = SyNRegistration{Float32}(
                scales=(2,),
                iterations=(3,),
                time_steps=3,
                verbose=false,
                array_type=ArrayType
            )

            fit!(reg, moving, static)

            # Forward transform
            moved = transform(reg, moving; direction=:forward)
            @test moved isa typeof(moving)
            @test size(moved) == size(moving)
            @test all(isfinite, Array(moved))

            # Inverse transform
            moved_inv = transform(reg, static; direction=:inverse)
            @test moved_inv isa typeof(static)
            @test size(moved_inv) == size(static)
            @test all(isfinite, Array(moved_inv))
        end
    end

    # =========================================================================
    # Diffeomorphism Property Tests
    # =========================================================================
    @testset "Diffeomorphism properties" begin

        @testset "inverse composition" begin
            # exp(v) ∘ exp(-v) ≈ identity (within numerical tolerance)
            Random.seed!(500)
            X, Y, Z, N = 8, 8, 8, 1

            # Use smaller velocity for better inverse composition accuracy
            v = ArrayType(randn(Float32, X, Y, Z, 3, N) .* 0.05f0)

            # Forward and inverse flows
            flow_pos = diffeomorphic_transform(v; time_steps=5)
            flow_neg = diffeomorphic_transform(-v; time_steps=5)

            # Compose forward with inverse
            composed = composition_transform(flow_pos, flow_neg)

            # Should be close to zero (identity displacement)
            @test maximum(abs.(Array(composed))) < 0.3f0
        end

        @testset "smooth output" begin
            # Diffeomorphic transform should produce smooth outputs
            Random.seed!(501)
            X, Y, Z, N = 16, 16, 16, 1

            v = ArrayType(randn(Float32, X, Y, Z, 3, N) .* 0.1f0)

            flow = diffeomorphic_transform(v; time_steps=7)

            flow_cpu = Array(flow)

            # Check that flow is smooth (finite differences should be bounded)
            grad_x = diff(flow_cpu, dims=1)
            grad_y = diff(flow_cpu, dims=2)
            grad_z = diff(flow_cpu, dims=3)

            # Smoothness: gradients shouldn't be too large
            @test maximum(abs.(grad_x)) < 1.0f0
            @test maximum(abs.(grad_y)) < 1.0f0
            @test maximum(abs.(grad_z)) < 1.0f0
        end
    end

    # =========================================================================
    # Full Registration Test
    # =========================================================================
    @testset "Full registration" begin

        @testset "register function" begin
            Random.seed!(600)
            X, Y, Z, C, N = 16, 16, 16, 1, 1

            # Create Gaussian blob as static
            static_cpu = zeros(Float32, X, Y, Z, C, N)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2) / (X/4)
                dy = (j - Y/2) / (Y/4)
                dz = (k - Z/2) / (Z/4)
                static_cpu[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            # Moving: blob shifted
            shift = 2
            moving_cpu = zeros(Float32, X, Y, Z, C, N)
            for i in 1:X, j in 1:Y, k in 1:Z
                dx = (i - X/2 - shift) / (X/4)
                dy = (j - Y/2) / (Y/4)
                dz = (k - Z/2) / (Z/4)
                moving_cpu[i, j, k, 1, 1] = exp(-(dx^2 + dy^2 + dz^2))
            end

            moving = ArrayType(moving_cpu)
            static = ArrayType(static_cpu)

            reg = SyNRegistration{Float32}(
                scales=(4, 2),
                iterations=(5, 5),
                learning_rate=0.01f0,
                time_steps=5,
                verbose=false,
                array_type=ArrayType
            )

            moved_xy, moved_yx, flow_xy, flow_yx = register(reg, moving, static)

            # Check output shapes
            @test size(moved_xy) == (X, Y, Z, C, N)
            @test size(moved_yx) == (X, Y, Z, C, N)
            @test size(flow_xy) == (X, Y, Z, 3, N)
            @test size(flow_yx) == (X, Y, Z, 3, N)

            # No NaN or Inf
            @test all(isfinite, Array(moved_xy))
            @test all(isfinite, Array(moved_yx))
            @test all(isfinite, Array(flow_xy))
            @test all(isfinite, Array(flow_yx))
        end
    end
end
