module MedicalImageRegistration

# Dependencies
using NNlib  # Still needed for batched_mul, conv operations
using Optimisers
using Statistics
import AcceleratedKernels as AK

# Note: We use our own pure Julia grid_sample implementation instead of NNlib.grid_sample.
# NNlib.grid_sample uses internal threading which breaks ALL Julia AD systems
# (Enzyme, Mooncake, Zygote). Our implementation is pure Julia and fully differentiable.
#
# GPU Acceleration: AcceleratedKernels.jl is used for cross-platform GPU support
# (CPU, CUDA, Metal, AMD ROCm). Operations transparently run on whatever array
# backend is provided - pass CuArray for CUDA, MtlArray for Metal, or regular
# Array for CPU multithreading.

# Core types and utilities
include("types.jl")
include("grid_sample.jl")  # Pure Julia grid_sample (before utils.jl which may use it)
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
