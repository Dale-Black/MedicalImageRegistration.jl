module MedicalImageRegistration

# GPU-first image registration for Julia
# Architecture: AcceleratedKernels.jl + Mooncake rrule!!
# No CPU fallbacks. No nested for loops.

# Dependencies
import AcceleratedKernels as AK
import Mooncake
import Mooncake: CoDual, NoFData, NoRData, @is_primitive, MinimalCtx
using Atomix

# Core operations
include("grid_sample.jl")
include("affine_grid.jl")
include("compose_affine.jl")
include("metrics.jl")

# Registration types and algorithms
include("types.jl")
include("affine.jl")

# Exports - Core operations
export grid_sample
export affine_grid
export compose_affine
export mse_loss, dice_loss, dice_score, ncc_loss

# Exports - Registration
export AffineRegistration
export register, transform, fit!, reset!, get_affine
export affine_transform

end # module
