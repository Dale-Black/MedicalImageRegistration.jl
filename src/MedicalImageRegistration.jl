module MedicalImageRegistration

# Dependencies
using NNlib
using Optimisers
using Statistics

# Core types and utilities
include("types.jl")
include("utils.jl")

# Registration algorithms
include("affine.jl")
include("syn.jl")

# Metrics and loss functions
include("metrics.jl")

# Exports
export AffineRegistration, SyNRegistration
export AffineParameters
export register, transform, get_affine
export dice_loss, dice_score, NCC, LinearElasticity
export mse_loss, init_parameters

end # module
