# MedicalImageRegistration.jl

A minimal Julia library for 2D and 3D medical image registration, inspired by [torchreg](https://github.com/codingfisch/torchreg).

## Demo

![Registration Demo](examples/output/registration_demo.gif)

*The animation shows: static target → misaligned moving image → registration in progress → aligned result → checkerboard overlay comparison*

## Features

- **Affine Registration**: Translation, rotation, zoom, and shear optimization
- **SyN Registration**: Symmetric diffeomorphic (deformable) registration
- **Multiresolution**: Coarse-to-fine optimization for speed and robustness
- **2D and 3D**: Full support for both image dimensions
- **GPU Acceleration**: Transparent CPU/CUDA/Metal support via AcceleratedKernels.jl
- **Automatic Differentiation**: Mooncake.jl for gradient computation

## Installation

```julia
using Pkg
Pkg.add("MedicalImageRegistration")
```

## Quick Start

### Affine Registration

```julia
using MedicalImageRegistration
using Metal  # or CUDA for NVIDIA GPUs

# Load images as arrays (X, Y, Z, C, N) - Julia convention
moving = MtlArray(rand(Float32, 64, 64, 64, 1, 1))  # GPU array
static = MtlArray(rand(Float32, 64, 64, 64, 1, 1))

# Create registration object
reg = AffineRegistration{Float32}(
    is_3d=true,
    scales=(4, 2),
    iterations=(500, 100),
    array_type=MtlArray  # Use MtlArray for Metal GPU
)

# Run registration
moved = register(reg, moving, static)

# Access the affine matrix
affine = get_affine(reg)

# Apply transform to another image
another_moved = transform(reg, another_image)
```

### SyN (Diffeomorphic) Registration

```julia
reg = SyNRegistration{Float32}(
    scales=(4, 2, 1),
    iterations=(30, 30, 10),
    array_type=MtlArray
)
moved = register(reg, moving, static)
```

### Custom Loss Functions

```julia
reg = AffineRegistration{Float32}(
    is_3d=true,
    learning_rate=0.01f0
)

# Use dice_loss instead of default mse_loss
moved = register(reg, moving, static; loss_fn=dice_loss)
```

## Running the Demo

To run the interactive demo with TestImages.jl:

```bash
cd examples
julia demo.jl
```

This will:
1. Automatically detect and use Metal GPU (Apple Silicon) if available
2. Load a test image (cameraman)
3. Create a synthetically misaligned version
4. Run affine registration to recover alignment
5. Generate a GIF animation showing the process
6. Save output images to `examples/output/`

**GPU Acceleration**: The demo automatically uses Metal GPU on macOS with Apple Silicon. CPU fallback is used when GPU is not available.

## Array Conventions

Julia uses column-major order. This package follows Julia conventions:

| Dimension | Julia (this package) | PyTorch (torchreg) |
|-----------|---------------------|-------------------|
| Spatial   | (X, Y) or (X, Y, Z) | (Y, X) or (Z, Y, X) |
| Full 2D   | (X, Y, C, N)        | (N, C, Y, X) |
| Full 3D   | (X, Y, Z, C, N)     | (N, C, Z, Y, X) |

## Dependencies

- [Mooncake.jl](https://github.com/compintell/Mooncake.jl) - Automatic differentiation
- [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl) - GPU acceleration (CPU/CUDA/Metal)
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) - Optimization algorithms
- [NNlib.jl](https://github.com/FluxML/NNlib.jl) - Batched matrix operations

## License

MIT
