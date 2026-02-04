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

# Include the dedicated test files that don't require torchreg
# These use PyTorch F.grid_sample and F.affine_grid for parity testing
include("test_grid_sample.jl")
include("test_affine_grid.jl")
include("test_compose_affine.jl")

# Note: Full integration tests with the register() API require the fit! function
# to work on the target GPU array type. These are better tested in the individual
# test files (test_affine.jl, test_syn.jl) with proper setup.
