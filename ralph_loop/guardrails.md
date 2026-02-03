# MedicalImageRegistration.jl Guardrails

Conventions, settled decisions, and rules learned from past iterations.

## Julia Coding Standards

### Naming Conventions
```julia
# Functions: snake_case
register(moving, static, reg)
get_affine(reg)
affine_transform(x, affine)

# Types: PascalCase
struct AffineRegistration <: AbstractRegistration
struct SyNRegistration <: AbstractRegistration

# Constants: SCREAMING_SNAKE
const DEFAULT_SCALES = (4, 2)
const DEFAULT_ITERATIONS = (500, 100)
```

### Type Annotations
- Use type annotations on struct fields
- Use type annotations on function arguments when it aids dispatch or clarity
- Avoid over-constraining (use `AbstractArray` not `Array{Float32, 5}`)

```julia
# Good
function register(moving::AbstractArray, static::AbstractArray, reg::AffineRegistration)

# Avoid (too restrictive)
function register(moving::Array{Float32, 5}, static::Array{Float32, 5}, reg::AffineRegistration)
```

### Keyword Arguments
- Use keyword arguments for optional parameters
- Provide sensible defaults matching torchreg where applicable

```julia
# Good
function AffineRegistration(;
    ndims::Int = 3,
    scales = (4, 2),
    iterations = (500, 100),
    learning_rate = 1e-2,
    verbose = true,
    loss_fn = mse_loss,
    with_translation = true,
    with_rotation = true,
    with_zoom = true,
    with_shear = false
)
```

## Array Conventions

### Axis Order (CRITICAL)
```
Julia (this package):
  2D: (X, Y, C, N)      where X=width, Y=height, C=channels, N=batch
  3D: (X, Y, Z, C, N)   where Z=depth

PyTorch (torchreg):
  2D: (N, C, Y, X)
  3D: (N, C, Z, Y, X)

Note: Julia is column-major, PyTorch is row-major
```

### Conversion Functions
```julia
# Julia → PyTorch (for testing)
function julia_to_torch(arr::AbstractArray{T, 4}) where T  # 2D
    permutedims(arr, (4, 3, 2, 1))  # (X,Y,C,N) → (N,C,Y,X)
end

function julia_to_torch(arr::AbstractArray{T, 5}) where T  # 3D
    permutedims(arr, (5, 4, 3, 2, 1))  # (X,Y,Z,C,N) → (N,C,Z,Y,X)
end

# PyTorch → Julia
function torch_to_julia(arr::AbstractArray{T, 4}) where T  # 2D
    permutedims(arr, (4, 3, 2, 1))  # (N,C,Y,X) → (X,Y,C,N)
end

function torch_to_julia(arr::AbstractArray{T, 5}) where T  # 3D
    permutedims(arr, (5, 4, 3, 2, 1))  # (N,C,Z,Y,X) → (X,Y,Z,C,N)
end
```

## NNlib.grid_sample Usage

### Signature
```julia
NNlib.grid_sample(input, grid; padding_mode=:zeros)
```

### Grid Format
- Grid should have shape matching input spatial dims
- Grid values in range [-1, 1] for normalized coordinates
- `padding_mode` options: `:zeros`, `:border`, `:reflection`

### Differences from PyTorch
- Check `align_corners` behavior carefully
- May need to handle batch dimension differently

## Automatic Differentiation (CRITICAL)

### REQUIRED: Use Enzyme.jl or Mooncake.jl

**DO NOT USE ZYGOTE.JL** - This is a hard requirement.

The project MUST use either Enzyme.jl or Mooncake.jl for all gradient computations. Zygote.jl is NOT compatible with project requirements.

**Which one to use?** See RESEARCH-AD-001 story - the ralph loop will evaluate both and choose the best option based on:
- Compatibility with NNlib.grid_sample
- Performance on our specific use case
- Stability and ease of use
- Error messages and debuggability

### Enzyme.jl Basic Pattern
```julia
using Enzyme

# For gradient of scalar loss with respect to parameters
function compute_loss(params, moving, static)
    moved = transform_with_params(moving, params)
    return loss_fn(moved, static)
end

# Using Enzyme.autodiff for reverse-mode AD
function compute_gradients(params, moving, static)
    dparams = Enzyme.make_zero(params)
    Enzyme.autodiff(Reverse, compute_loss, Active, Duplicated(params, dparams), Const(moving), Const(static))
    return dparams
end
```

### Enzyme with Tuples of Parameters
```julia
# When optimizing multiple parameter arrays
function loss_fn(translation, rotation, zoom, shear, moving, static)
    affine = compose_affine(translation, rotation, zoom, shear)
    moved = affine_transform(moving, affine)
    return mse_loss(moved, static)
end

# Get gradients for each parameter
dt = zero(translation)
dr = zero(rotation)
dz = zero(zoom)
ds = zero(shear)

Enzyme.autodiff(Reverse, loss_fn, Active,
    Duplicated(translation, dt),
    Duplicated(rotation, dr),
    Duplicated(zoom, dz),
    Duplicated(shear, ds),
    Const(moving),
    Const(static)
)
```

### Mooncake.jl Fallback Pattern
```julia
using Mooncake

# If Enzyme doesn't work, try Mooncake
rule = Mooncake.build_rrule(loss_fn, params, moving, static)
loss, (_, dparams, _, _) = Mooncake.value_and_pullback!!(rule, 1.0, loss_fn, params, moving, static)
```

### Mutable State
- Enzyme works best with immutable data
- If using mutable structs, may need `Duplicated` wrapper
- Consider using immutable parameter tuples for optimization

## Optimisers.jl Usage

### Setup Pattern
```julia
using Optimisers

# Create optimizer state
opt_state = Optimisers.setup(Adam(learning_rate), params)

# Update step
grads = compute_gradients(params, ...)
opt_state, params = Optimisers.update(opt_state, params, grads)
```

## Testing Conventions

### Tolerances
```julia
# Utility functions (exact math)
rtol = 1e-5
atol = 1e-8

# Metrics (may have numerical differences)
rtol = 1e-4
atol = 1e-6

# Registration results (optimization is stochastic)
rtol = 1e-2
atol = 1e-4
```

### Test Structure
```julia
@testset "FeatureName" begin
    @testset "2D" begin
        # Test with 2D arrays
    end

    @testset "3D" begin
        # Test with 3D arrays
    end

    @testset "batch_size > 1" begin
        # Test with batch_size = 2
    end
end
```

### Random Seeds
```julia
using Random
Random.seed!(42)  # Use consistent seeds for reproducibility
```

## Git Commit Conventions

### Message Format
```
[STORY-ID] Brief description

Examples:
[RESEARCH-001] Document torchreg affine and SyN algorithms
[IMPL-AFFINE-002] Implement compose_affine for 2D and 3D
[TEST-METRICS-001] Add parity tests for dice and NCC losses
[WIP] Partial progress on IMPL-SYN-001 (timeout)
```

### What to Commit
- Always commit `ralph_loop/progress.md` and `ralph_loop/prd.json`
- Commit source files that were modified
- Commit test files that were added/modified

## Mistakes Log

*This section will be populated as we learn from errors*

### Template
```
### Mistake N: [Brief title]
**What happened**: Description of the error
**Root cause**: Why it happened
**Fix**: How it was resolved
**Lesson**: Rule to prevent recurrence
```
