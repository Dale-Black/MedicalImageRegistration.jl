module MedicalImageRegistration

# Dependencies
using NNlib
using Optimisers
using Statistics
import AcceleratedKernels as AK

# Note: Manual gradient computation used instead of AD library
# This avoids Zygote (prohibited) and works around Enzyme/Mooncake limitations
# with NNlib.grid_sample threading
#
# GPU Acceleration: AcceleratedKernels.jl is used for cross-platform GPU support
# (CPU, CUDA, Metal, AMD ROCm). Operations transparently run on whatever array
# backend is provided - pass CuArray for CUDA, MtlArray for Metal, or regular
# Array for CPU multithreading.

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
