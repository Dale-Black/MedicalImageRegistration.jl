# MedicalImageRegistration.jl

A minimal Julia library for 2D and 3D medical image registration, inspired by [torchreg](https://github.com/codingfisch/torchreg).

## Features

- **Affine Registration**: Translation, rotation, zoom, and shear optimization
- **SyN Registration**: Symmetric diffeomorphic (deformable) registration
- **Multiresolution**: Coarse-to-fine optimization for speed and robustness
- **2D and 3D**: Full support for both image dimensions
- **Automatic Differentiation**: Enzyme.jl integration for gradients

## Installation

```julia
using Pkg
Pkg.add("MedicalImageRegistration")
```

## Quick Start

### Affine Registration

```julia
using MedicalImageRegistration

# Load images as arrays (X, Y, Z, C, N) - Julia convention
moving = ...  # Array{Float32, 5}
static = ...  # Array{Float32, 5}

# Create registration object
reg = AffineRegistration(; ndims=3, scales=(4, 2), iterations=(500, 100))

# Run registration
moved = register(moving, static, reg)

# Access the affine matrix
affine = get_affine(reg)

# Apply transform to another image
another_moved = transform(another_image, reg)
```

### SyN (Diffeomorphic) Registration

```julia
reg = SyNRegistration(; ndims=3, scales=(4, 2, 1), iterations=(30, 30, 10))
moved_xy, moved_yx, flow_xy, flow_yx = register(moving, static, reg)
```

### Custom Loss Functions

```julia
reg = AffineRegistration(;
    ndims=3,
    dissimilarity_fn=dice_loss,
    optimizer=Optimisers.Adam(1e-2)
)
```

## Array Conventions

Julia uses column-major order. This package follows Julia conventions:

| Dimension | Julia (this package) | PyTorch (torchreg) |
|-----------|---------------------|-------------------|
| Spatial   | (X, Y) or (X, Y, Z) | (Y, X) or (Z, Y, X) |
| Full 2D   | (X, Y, C, N)        | (N, C, Y, X) |
| Full 3D   | (X, Y, Z, C, N)     | (N, C, Z, Y, X) |

## Dependencies

- [NNlib.jl](https://github.com/FluxML/NNlib.jl) - `grid_sample` for spatial transforms
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) - Automatic differentiation
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) - Optimization algorithms

## License

MIT
