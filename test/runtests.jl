using MedicalImageRegistration
using Test
using PythonCall

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
