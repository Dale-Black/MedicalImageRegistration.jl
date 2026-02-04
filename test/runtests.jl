using MedicalImageRegistration
using Test
using Random

# Check if PythonCall and torch are available for parity testing
const PYTHON_AVAILABLE = try
    using PythonCall
    torch = pyimport("torch")
    np = pyimport("numpy")
    true
catch e
    @warn "PythonCall/PyTorch not available - parity tests will be skipped" exception=e
    false
end

if PYTHON_AVAILABLE
    using PythonCall
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

    # Include the dedicated test files that don't require torchreg
    # These use PyTorch F.grid_sample and F.affine_grid for parity testing
    include("test_grid_sample.jl")
    include("test_affine_grid.jl")
    include("test_compose_affine.jl")

    # HU preservation tests (don't require torchreg)
    include("test_hu_preservation.jl")

    # Transform resampling tests
    include("test_resample_transform.jl")

    # Clinical registration workflow tests
    include("test_clinical.jl")
else
    @warn "Skipping all Python-based parity tests"

    # Run tests that don't require Python (basic functionality tests)
    @testset "Basic functionality (no Python)" begin
        @testset "grid_sample CPU" begin
            input = randn(Float32, 4, 4, 1, 1)
            grid = zeros(Float32, 2, 4, 4, 1)
            for j in 1:4, i in 1:4
                grid[1, i, j, 1] = Float32(2) * Float32(i - 1) / Float32(3) - Float32(1)
                grid[2, i, j, 1] = Float32(2) * Float32(j - 1) / Float32(3) - Float32(1)
            end
            output = grid_sample(input, grid)
            @test size(output) == (4, 4, 1, 1)
            @test isapprox(output, input; rtol=1e-5)
        end

        @testset "affine_grid CPU" begin
            theta = zeros(Float32, 2, 3, 1)
            theta[1, 1, 1] = 1.0f0
            theta[2, 2, 1] = 1.0f0
            grid = affine_grid(theta, (4, 4))
            @test size(grid) == (2, 4, 4, 1)
        end

        @testset "compose_affine CPU" begin
            translation = zeros(Float32, 2, 1)
            rotation = Float32.(reshape([1 0; 0 1], 2, 2, 1))
            zoom = ones(Float32, 2, 1)
            shear = zeros(Float32, 2, 1)
            theta = compose_affine(translation, rotation, zoom, shear)
            @test size(theta) == (2, 3, 1)
        end

        @testset "mse_loss CPU" begin
            pred = randn(Float32, 8, 8, 1, 1)
            loss = mse_loss(pred, pred)
            # Convert 1-element array to scalar
            import AcceleratedKernels as AK
            loss_val = AK.reduce(+, loss; init=0.0f0)
            @test loss_val ≈ 0.0f0 atol=1e-7
        end
    end
end
