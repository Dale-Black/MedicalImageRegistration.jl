module MedicalImageRegistration

# Dependencies
using NNlib
using Optimisers
using Statistics
# Note: Manual gradient computation used instead of AD library
# This avoids Zygote (prohibited) and works around Enzyme/Mooncake limitations
# with NNlib.grid_sample threading

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
export register, transform, get_affine, compose_affine, affine_transform
export dice_loss, dice_score, NCC, LinearElasticity
export mse_loss, init_parameters

end # module
