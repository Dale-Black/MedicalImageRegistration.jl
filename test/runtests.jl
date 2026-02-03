using MedicalImageRegistration
using Test
using PythonCall

# Import torchreg for parity testing
const torchreg = pyimport("torchreg")
const torch = pyimport("torch")
const np = pyimport("numpy")

include("test_utils.jl")
include("test_affine.jl")
include("test_syn.jl")
include("test_metrics.jl")
