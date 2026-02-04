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
include("mi_loss.jl")

# Physical coordinates and anisotropic voxel support
include("physical.jl")

# Transform resampling for multi-resolution workflows
include("resample_transform.jl")

# Registration types and algorithms
include("types.jl")
include("affine.jl")
include("syn.jl")

# Exports - Core operations
export grid_sample
export affine_grid
export compose_affine
export mse_loss, dice_loss, dice_score, ncc_loss
export mi_loss, nmi_loss

# Exports - Physical coordinates
export PhysicalImage
export affine_grid_physical
export resample

# Exports - Transform resampling
export resample_displacement, resample_velocity
export upsample_affine_transform, upsample_affine_transform_physical
export invert_displacement

# Exports - SyN operations
export spatial_transform, diffeomorphic_transform, composition_transform
export gauss_smoothing, linear_elasticity

# Exports - Registration
export AffineRegistration, SyNRegistration
export register, transform, fit!, reset!, get_affine
export affine_transform, apply_flows

end # module
