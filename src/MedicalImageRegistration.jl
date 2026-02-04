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

# Exports
export grid_sample
export affine_grid
export compose_affine

end # module
