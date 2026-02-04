# MedicalImageRegistration.jl Progress Log

## GPU-First Rewrite

Starting fresh with a clean architecture:
- **AcceleratedKernels.jl** for ALL parallel operations (AK.foreachindex)
- **Mooncake rrule!!** for ALL gradients (custom backward passes)
- **MtlArrays** for testing locally on Metal GPU
- **NO CPU fallbacks, NO nested for loops, NO hacks**

---

## Architecture

```
Forward Pass: AK.foreachindex → runs on GPU
Backward Pass: AK.foreachindex → runs on GPU
AD Integration: Mooncake rrule!! connects them
```

Every differentiable function follows this pattern:
1. `my_function()` - forward pass with AK.foreachindex
2. `∇my_function()` - backward pass with AK.foreachindex
3. `rrule!!(::CoDual{typeof(my_function)}, ...)` - tells Mooncake to use our backward

---

## Story Log

### [NUKE-001] Delete all existing src/ code and start fresh

**Status:** DONE
**Date:** 2026-02-03

**What was deleted:**
- `src/affine.jl` (28KB) - old AffineRegistration with CPU fallbacks
- `src/grid_sample.jl` (24KB) - old grid_sample implementation
- `src/metrics.jl` (14KB) - old metrics with mixed patterns
- `src/syn.jl` (38KB) - old SyN registration
- `src/types.jl` (4KB) - old type definitions
- `src/utils.jl` (18KB) - old utility functions

**What remains:**
- `src/MedicalImageRegistration.jl` - empty module declaration only

The old code had:
- NNlib dependency (uses threading, breaks AD)
- Mixed CPU/GPU patterns
- Nested for loops in some places
- Manual gradient hacks

Starting fresh with GPU-first architecture: every function will use AK.foreachindex with Mooncake rrule!!.

---

### [RESEARCH-MOONCAKE-001] Document Mooncake rrule!! pattern for GPU functions

**Status:** DONE
**Date:** 2026-02-03

#### Key Finding: Mooncake CANNOT autodiff through AK.foreachindex

When Mooncake tries to automatically differentiate through `AK.foreachindex`, it fails with:
```
No rrule!! available for foreigncall with primal argument types Tuple{Val{:jl_new_task}, ...}
```

This is because AcceleratedKernels.jl uses Julia's task-based parallelism (`jl_new_task`) internally, which Mooncake cannot differentiate through.

**Solution:** Every function using `AK.foreachindex` MUST have a hand-written `rrule!!`.

#### Mooncake rrule!! Signature

```julia
import Mooncake
import Mooncake: CoDual, NoFData, NoRData, @is_primitive, MinimalCtx

function Mooncake.rrule!!(
    ::CoDual{typeof(my_function)},
    x::CoDual{A, F}
) where {A<:AbstractArray, F}
    # Access primal and fdata
    x_primal = x.x      # The actual array values
    x_fdata = x.dx      # Gradient accumulator (same shape as x)

    # Forward pass
    output = my_function(x_primal)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    # Define pullback (called during backward pass)
    function my_function_pullback(_rdata)
        # CRITICAL: For arrays, gradient is in output_fdata (captured via closure)
        # _rdata is NoRData() for arrays!
        ∇my_function!(x_fdata, output_fdata, x_primal)  # Accumulate into x_fdata
        return NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), my_function_pullback
end
```

#### CoDual, NoFData, NoRData Explained

- **CoDual{T, F}**: Wraps primal value (`.x`) with forward data (`.dx`)
  - For arrays: `.dx` is another array of same shape (gradient accumulator)
  - For scalars: `.dx` is `NoFData()`

- **NoFData()**: Marker for types with no forward-pass data (bits types like Float64)

- **NoRData()**: Marker for types whose gradient is handled via fdata (arrays)
  - The pullback receives `NoRData()` for arrays
  - Gradients accumulate in the fdata captured via closure

#### Critical Pattern: Array Gradients Use FData

For array-valued operations:
1. Forward pass creates `output_fdata` (zeros)
2. Next op in chain accumulates its gradient into `output_fdata`
3. Our pullback reads `output_fdata` (via closure) and propagates to `x_fdata`
4. Pullback receives `NoRData()` argument (NOT the gradient!)

```julia
function my_function_pullback(_rdata)
    # _rdata is NoRData() for arrays - IGNORE IT
    # The gradient is in output_fdata (captured)
    ∇my_function!(x_fdata, output_fdata, x_primal)
    return NoRData(), NoRData()
end
```

#### Working Example: Element-wise Square

```julia
import Mooncake
import Mooncake: CoDual, NoFData, NoRData, @is_primitive, MinimalCtx
import AcceleratedKernels as AK

# Forward pass using AK.foreachindex
function my_square(x::AbstractArray{T}) where T
    out = similar(x)
    AK.foreachindex(out) do i
        @inbounds out[i] = x[i]^2
    end
    return out
end

# Mark as primitive (prevents Mooncake from trying to autodiff)
@is_primitive MinimalCtx Tuple{typeof(my_square), AbstractArray}

# Backward pass using AK.foreachindex
function ∇my_square!(d_x, d_out, x::AbstractArray{T}) where T
    AK.foreachindex(d_x) do i
        @inbounds d_x[i] += d_out[i] * 2 * x[i]
    end
    return nothing
end

# Mooncake rrule!!
function Mooncake.rrule!!(
    ::CoDual{typeof(my_square)},
    x::CoDual{A, F}
) where {A<:AbstractArray, F}
    x_primal = x.x
    x_fdata = x.dx

    output = my_square(x_primal)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function my_square_pullback(_rdata)
        ∇my_square!(x_fdata, output_fdata, x_primal)
        return NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), my_square_pullback
end
```

#### Testing with Mooncake.TestUtils.test_rule

```julia
using Mooncake.TestUtils: test_rule
using Mooncake: ReverseMode
using StableRNGs

# Test correctness against finite differences
test_rule(StableRNG(1), my_square, x_cpu; is_primitive=true, mode=ReverseMode)
```

This verifies:
- Forward pass correctness
- Gradient correctness via finite differences
- Interface compliance

#### Verified on MtlArrays

The rrule!! pattern works directly on Metal GPU arrays:

```julia
x_mtl = MtlArray(Float32[1.0, 2.0, 3.0, 4.0])
x_fdata = Metal.zeros(Float32, 4)
x_codual = CoDual(x_mtl, x_fdata)

output_codual, pb = Mooncake.rrule!!(CoDual(my_square, NoFData()), x_codual)
# output_codual.x is MtlArray (forward result)
# output_codual.dx is MtlArray (zeros, for gradient accumulation)

output_codual.dx .= 1.0f0  # Simulate upstream gradient
pb(NoRData())              # Propagate back
# x_fdata now contains gradients
```

**Note:** Mooncake's high-level `prepare_gradient_cache` has issues with MtlArray storage types. Use the direct `rrule!!` interface or composition.

#### Metal Array Support (Additional Definitions)

To fully integrate with Mooncake's gradient interface, add:

```julia
Mooncake.tangent_type(::Type{<:MtlArray{T,N}}) where {T,N} = MtlArray{T,N}

function Mooncake.zero_tangent(x::MtlArray{T,N}) where {T,N}
    return Metal.zeros(T, size(x)...)
end
```

---

### [IMPL-GRID-001] Implement grid_sample with AK.jl + Mooncake rrule!!

**Status:** DONE
**Date:** 2026-02-03

#### Implementation Summary

Implemented `grid_sample` function that samples from an input array at locations specified by a grid, using bilinear (2D) or trilinear (3D) interpolation. Fully GPU-accelerated with custom Mooncake rrule!!.

#### Key Files
- `src/grid_sample.jl` - Main implementation (686 lines)
- `test/test_grid_sample.jl` - Comprehensive test suite

#### Features Implemented
- **2D Bilinear interpolation**: (X, Y, C, N) input + (2, X_out, Y_out, N) grid → (X_out, Y_out, C, N) output
- **3D Trilinear interpolation**: (X, Y, Z, C, N) input + (3, X_out, Y_out, Z_out, N) grid → (X_out, Y_out, Z_out, C, N) output
- **padding_mode=:zeros**: Out-of-bounds samples return 0
- **padding_mode=:border**: Out-of-bounds samples clamped to border
- **align_corners=true/false**: Control corner alignment behavior

#### Architecture
```julia
# Forward pass using AK.foreachindex
function _grid_sample_2d(input, grid, pm, align_corners)
    output = similar(input, X_out, Y_out, C, N)
    AK.foreachindex(output) do idx
        # Convert linear index to (i_out, j_out, c, n)
        # Unnormalize grid coords to pixel coords (1-indexed)
        # Bilinear interpolation with 4 corners
    end
    return output
end

# Backward pass - gradients for input
function _∇grid_sample_input_2d!(d_input, d_output, grid, pm, align_corners)
    AK.foreachindex(d_output) do idx
        # Scatter gradients to 4 corners using Atomix.@atomic
    end
end

# Backward pass - gradients for grid
function _∇grid_sample_grid_2d!(d_grid, d_output, input, grid, pm, align_corners)
    AK.foreachindex(d_output) do idx
        # Compute spatial gradients and chain rule
    end
end
```

#### Critical Fix: 1-Indexed Coordinate Conversion

Original unnormalize code produced 0-indexed pixel coords, but Julia arrays are 1-indexed:
```julia
# WRONG (0-indexed)
x_pix = (x_norm + 1) / 2 * (X - 1)  # -1 → 0, +1 → X-1

# CORRECT (1-indexed for Julia)
x_pix = (x_norm + 1) / 2 * (X - 1) + 1  # -1 → 1, +1 → X
```

#### Gradient Handling

Uses Atomix.jl for atomic operations on GPU to handle race conditions when multiple output positions scatter gradients to the same input position:
```julia
Atomix.@atomic d_input[i, j, c, n] += grad
```

#### Mooncake Integration

Both 2D and 3D versions registered as primitives:
```julia
@is_primitive MinimalCtx Tuple{typeof(grid_sample), AbstractArray{<:Any,4}, AbstractArray{<:Any,4}}
@is_primitive MinimalCtx Tuple{typeof(grid_sample), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}
```

#### Test Results

All acceptance criteria verified:
- ✓ Forward pass works on MtlArrays (Metal GPU)
- ✓ Backward pass works on MtlArrays
- ✓ Gradients verified against finite differences (rtol=1e-2)
- ✓ Matches PyTorch F.grid_sample within rtol=1e-5
- ✓ Both padding modes work
- ✓ Both align_corners modes work
- ✓ 2D and 3D interpolation work

#### PyTorch Parity

Achieved numerical parity with PyTorch F.grid_sample:
- Maximum difference: ~1.8e-7 (well within rtol=1e-5)
- Tested 2D bilinear (align_corners true/false)
- Tested 2D border padding
- Tested 3D trilinear

---

### [IMPL-AFFINE-GRID-001] Implement affine_grid with AK.jl + Mooncake rrule!!

**Status:** DONE
**Date:** 2026-02-03

#### Implementation Summary

Implemented `affine_grid` function that generates a sampling grid from an affine transformation matrix. Fully GPU-accelerated with custom Mooncake rrule!!.

#### Key Files
- `src/affine_grid.jl` - Main implementation (~300 lines)
- `test/test_affine_grid.jl` - Comprehensive test suite

#### Features Implemented
- **2D affine grid**: theta (2, 3, N) → grid (2, X_out, Y_out, N)
- **3D affine grid**: theta (3, 4, N) → grid (3, X_out, Y_out, Z_out, N)
- **align_corners=true/false** support
- Multiple size tuple variants: (X, Y), (X, Y, C, N), (X, Y, Z), (X, Y, Z, C, N)

#### Affine Matrix Convention
```
2D: [a b tx]    3D: [a b c tx]
    [c d ty]        [d e f ty]
                    [g h i tz]
```

The output grid coordinates are computed as:
- x_out = a*x_base + b*y_base + tx
- y_out = c*x_base + d*y_base + ty

Where x_base, y_base are the normalized input coordinates in [-1, 1].

#### Architecture
```julia
# Forward pass
function _affine_grid_2d(theta, size, align_corners)
    grid = similar(theta, 2, X_out, Y_out, N)
    AK.foreachindex(grid) do idx
        coord, i, j, n = _linear_to_cartesian_4d_affine(idx, X_out, Y_out)
        x_base, y_base = _generate_base_coord_2d(i, j, X_out, Y_out, T, align_corners)
        # Apply affine: grid[coord] = theta[coord, 1]*x + theta[coord, 2]*y + theta[coord, 3]
    end
    return grid
end

# Backward pass
function _∇affine_grid_theta_2d!(d_theta, d_grid, align_corners)
    AK.foreachindex(d_grid) do idx
        # d_theta[coord, 1] += d_grid[coord, i, j] * x_base
        # d_theta[coord, 2] += d_grid[coord, i, j] * y_base
        # d_theta[coord, 3] += d_grid[coord, i, j]
        Atomix.@atomic d_theta[coord, col, n] += ...
    end
end
```

#### Mooncake Integration
Registered as primitives for all size tuple variants:
```julia
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{2,Int}}
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{3,Int}}
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{4,Int}}
@is_primitive MinimalCtx Tuple{typeof(affine_grid), AbstractArray{<:Any,3}, NTuple{5,Int}}
```

#### Test Results

All acceptance criteria verified:
- ✓ Forward pass works on MtlArrays (Metal GPU)
- ✓ Backward pass works on MtlArrays
- ✓ Gradients verified via finite differences
- ✓ Matches PyTorch F.affine_grid within rtol=1e-5
- ✓ 2D and 3D support working
- ✓ Multiple size tuple variants supported

#### PyTorch Parity

Achieved numerical parity with PyTorch F.affine_grid:
- Maximum difference: ~6e-8 (well within rtol=1e-5)
- Tested 2D identity and random transforms
- Tested 3D identity and random transforms

---

### [IMPL-COMPOSE-001] Implement compose_affine with AK.jl + Mooncake rrule!!

**Status:** DONE
**Date:** 2026-02-03

#### Implementation Summary

Implemented `compose_affine` function that builds an affine transformation matrix from individual components (translation, rotation, zoom, shear). Fully GPU-accelerated with custom Mooncake rrule!!.

#### Key Files
- `src/compose_affine.jl` - Main implementation (~300 lines)
- `test/test_compose_affine.jl` - Comprehensive test suite

#### Features Implemented
- **2D affine composition**: translation (2,N), rotation (2,2,N), zoom (2,N), shear (2,N) → theta (2,3,N)
- **3D affine composition**: translation (3,N), rotation (3,3,N), zoom (3,N), shear (3,N) → theta (3,4,N)

#### Algorithm
The affine matrix is computed as: `[R @ S | t]` where:
- R is the rotation matrix
- S is the upper-triangular scale+shear matrix: `S = diag(zoom) + shear_upper_triangular`
- t is the translation vector

For 2D:
```
S = [sx  sxy]
    [0   sy ]
```

For 3D:
```
S = [sx  sxy sxz]
    [0   sy  syz]
    [0   0   sz ]
```

#### Test Results

All acceptance criteria verified:
- ✓ Forward pass works on MtlArrays (Metal GPU)
- ✓ Backward pass works on MtlArrays
- ✓ Gradients verified via finite differences
- ✓ Matches torchreg compose_affine exactly
- ✓ 2D and 3D support working

#### torchreg Parity

Achieved perfect numerical parity with torchreg compose_affine:
- Maximum difference: 0.0 (exact match)
- Tested 2D and 3D with random parameters

---

### [IMPL-METRICS-001] Implement loss functions with AK.jl + Mooncake rrule!!

**Status:** DONE
**Date:** 2026-02-03

#### Implementation Summary

Implemented GPU-accelerated loss functions for image registration:
- **mse_loss**: Mean Squared Error
- **dice_score**: Dice coefficient (Sørensen–Dice)
- **dice_loss**: 1 - dice_score
- **ncc_loss**: Normalized Cross Correlation (local, windowed)

All functions are fully GPU-accelerated with AcceleratedKernels.jl and have custom Mooncake rrule!! for automatic differentiation.

#### Key Files
- `src/metrics.jl` - Main implementation (~650 lines)
- `test/test_metrics.jl` - Comprehensive GPU test suite

#### Architecture Pattern

All metrics follow the GPU-first pattern:

```julia
# Forward pass - GPU reduction
function loss_fn(pred, target)
    # Element-wise computation via AK.foreachindex
    # Reduction via AK.reduce (no scalar indexing!)
    return result  # Returns 1-element array
end

# Backward pass - GPU gradient computation
function _∇loss_fn!(d_pred, d_target, d_output_arr, ...)
    # Extract scalar via AK.reduce (no scalar indexing!)
    d_output = _extract_scalar(d_output_arr)
    # Gradient computation via AK.foreachindex
end

# Mooncake rrule!!
@is_primitive MinimalCtx Tuple{typeof(loss_fn), ...}
function Mooncake.rrule!!(...) ... end
```

#### Key Implementation Details

1. **No Scalar Indexing**: All GPU arrays return 1-element arrays instead of scalars. Values extracted via `AK.reduce(+, arr; init=zero(T))`.

2. **Type Safety in Closures**: Avoid capturing `T = eltype(...)` types in AK.foreachindex closures - GPU kernels require bits types only.

3. **NCC Local Windows**: Uses explicit loops within AK.foreachindex for box filtering (O(n * k³) but GPU-compatible).

#### Features Implemented

**mse_loss**:
- Computes mean((pred - target)²)
- Works on any array shape
- Gradients: d_pred = 2*(pred - target)/n

**dice_score / dice_loss**:
- Dice = 2*sum(pred*target) / sum(pred + target)
- Works on 2D (4D arrays) and 3D (5D arrays)
- Soft dice for probability masks
- Gradients: uses quotient rule

**ncc_loss**:
- Local NCC in windows of kernel_size³
- CC = (cross² + ε) / (var_p * var_t + ε)
- Returns negative mean CC (loss to minimize)
- Gradients: chain rule through local sums

#### Test Results

All acceptance criteria verified:
- ✓ mse_loss: forward + backward on MtlArrays
- ✓ dice_score: forward + backward on MtlArrays (2D and 3D)
- ✓ dice_loss: forward + backward on MtlArrays
- ✓ ncc_loss: forward + backward on MtlArrays
- ✓ All gradients verified against finite differences
- ✓ GPU/CPU parity verified

#### Test Summary
```
Test Summary: | Pass  Total
mse_loss      |   10     10
dice_score    |   12     12
dice_loss     |    6      6
ncc_loss      |   16     16
```

---

### [IMPL-AFFINE-REG-001] Implement AffineRegistration with GPU optimization loop

**Status:** DONE
**Date:** 2026-02-03

#### Implementation Summary

Implemented full affine image registration with GPU-accelerated optimization loop. The entire registration pipeline runs on GPU without any CPU fallbacks.

#### Key Files
- `src/types.jl` - AffineRegistration struct (~180 lines)
- `src/affine.jl` - Registration algorithm (~550 lines)

#### Architecture

The registration follows torchreg's architecture:
1. Multi-resolution pyramid (coarse-to-fine)
2. Adam optimizer for parameter updates
3. Composition: `compose_affine` → `affine_grid` → `grid_sample` → `loss`

```julia
# Full registration pipeline
reg = AffineRegistration{Float32}(
    is_3d=false,
    scales=(4, 2),          # Multi-resolution scales
    iterations=(500, 100),  # Iterations per scale
    learning_rate=0.01f0,
    with_translation=true,
    with_rotation=true,
    with_zoom=true,
    with_shear=false,
    array_type=MtlArray     # GPU arrays
)

moved = register(reg, moving, static)  # Returns moved image
```

#### GPU-Compatible Design

Key challenges solved:

1. **No Scalar Indexing**: All array initialization done on CPU then transferred:
   ```julia
   theta_cpu = zeros(T, 2, 3, N)
   for n in 1:N
       theta_cpu[1, 1, n] = one(T)
       theta_cpu[2, 2, n] = one(T)
   end
   theta = similar(image, 2, 3, N)
   copyto!(theta, theta_cpu)
   ```

2. **Bits-Type Closures**: GPU kernels require bits types. Extract arrays from structs before closure:
   ```julia
   # Wrong: captures non-bits AdamState struct
   AK.foreachindex(param) do idx
       state.m[idx] = ...  # Fails on GPU
   end

   # Correct: capture array directly
   m_arr = state.m
   AK.foreachindex(param) do idx
       m_arr[idx] = ...  # Works on GPU
   end
   ```

3. **Manual Gradient Chain**: Since Mooncake cannot autodiff through AK.foreachindex, we manually chain the pullbacks:
   ```julia
   # Forward: compose_affine → affine_grid → grid_sample → loss
   # Backward: d_loss → d_moved → d_grid → d_theta → d_params
   ```

#### Features Implemented

**AffineRegistration struct**:
- Configuration: scales, iterations, learning_rate, align_corners, padding_mode
- Parameter flags: with_translation, with_rotation, with_zoom, with_shear
- Parameters: translation (D,N), rotation (D,D,N), zoom (D,N), shear (D,N)
- Loss history tracking

**Functions**:
- `AffineRegistration{T}(; kwargs...)` - Constructor with sensible defaults
- `reset!(reg)` - Reset parameters to identity
- `get_affine(reg)` - Get current affine matrix
- `fit!(reg, moving, static)` - Run optimization
- `register(reg, moving, static)` - Convenience function (fit + transform)
- `transform(reg, image)` - Apply current transform to image
- `affine_transform(image, theta)` - Apply explicit affine matrix

**Adam Optimizer**:
- GPU-compatible implementation using AK.foreachindex
- Per-parameter learning rate
- Standard hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8

**Multi-Resolution Pyramid**:
- Coarse-to-fine optimization
- Image resampling via grid_sample with identity affine
- Configurable scales and iterations per scale

#### Test Results

All acceptance criteria verified:

**2D Registration on MtlArray**:
```
Scale 1/1: scale=2, shape=(16, 16), iters=50
  Iter 1/50: loss = 0.067637
  Iter 50/50: loss = 0.000425
Loss decreased: true
```

**3D Registration on MtlArray**:
```
Scale 1/1: scale=2, shape=(8, 8, 8), iters=30
  Iter 1/30: loss = 0.003846
  Iter 30/30: loss = 0.002603
Loss decreased: true
```

**Multi-Resolution Pyramid on MtlArray**:
```
Scale 1/2: scale=4, shape=(16, 16), iters=30
  Iter 1/30: loss = 0.117188
  Iter 30/30: loss = 0.001056
Scale 2/2: scale=2, shape=(32, 32), iters=20
  Iter 1/20: loss = 0.00142
  Iter 20/20: loss = 0.000606
Total iterations: 50
Loss decreased: true
```

#### Acceptance Criteria Status
- ✓ src/affine.jl with AffineRegistration struct
- ✓ fit! function runs entirely on GPU
- ✓ Manual gradient computation through entire forward pass (compose_affine → affine_grid → grid_sample → loss)
- ✓ Multi-resolution pyramid support
- ✓ register() and transform() API
- ✓ Converges on test cases with MtlArrays (2D and 3D)

---

### [IMPL-SYN-001] Implement SyN diffeomorphic registration on GPU

**Status:** DONE
**Date:** 2026-02-03

#### Implementation Summary

Implemented full SyN (Symmetric Normalization) diffeomorphic registration. The entire registration pipeline runs on GPU without any CPU fallbacks, using AcceleratedKernels.jl and Mooncake rrule!! for AD.

#### Key Files
- `src/syn.jl` - Main implementation (~1400 lines)
- `test/test_syn.jl` - Comprehensive GPU test suite (84 tests)

#### Features Implemented

**Core Operations:**
- **spatial_transform**: Warp image using displacement field (identity grid + displacement)
- **diffeomorphic_transform**: Scaling-and-squaring algorithm to convert velocity to diffeomorphic displacement
- **composition_transform**: Compose two displacement fields
- **gauss_smoothing**: Separable Gaussian smoothing for flow regularization
- **linear_elasticity**: Linear elasticity regularization loss

**SyNRegistration struct:**
- Multi-resolution pyramid (scales, iterations per scale)
- Symmetric registration (forward and inverse flows)
- Velocity field smoothing (sigma_flow)
- Image smoothing (sigma_img)
- Regularization weight (lambda_)
- Configurable time_steps for scaling-and-squaring (default: 7)

**API:**
- `SyNRegistration{T}(; kwargs...)` - Constructor with sensible defaults
- `reset!(reg)` - Reset velocity fields
- `fit!(reg, moving, static)` - Run optimization
- `register(reg, moving, static)` - Convenience function (fit + transform)
- `transform(reg, image; direction=:forward/:inverse)` - Apply transform
- `apply_flows(reg, x, y, v_xy, v_yx)` - Compute half and full flows/images

#### Architecture

SyN registration works by:
1. Optimizing velocity fields `v_xy` (moving→static) and `v_yx` (static→moving)
2. Converting velocities to diffeomorphic displacements via scaling-and-squaring
3. Computing half-way images via warping with half-flows
4. Computing full-flows via composition of half-flows
5. Minimizing dissimilarity + regularization loss

```julia
# Scaling-and-squaring
function diffeomorphic_transform(v; time_steps=7)
    v_scaled = v / 2^time_steps
    result = v_scaled
    for _ in 1:time_steps
        result = composition_transform(result, result)
    end
    return result
end

# Composition: result[p] = v2[p] + v1[p + v2[p]]
function composition_transform(v1, v2)
    v1_warped = spatial_transform_displacement(v1, v2)
    return v2 + v1_warped
end
```

#### GPU Compatibility Fixes

1. **Non-bits type in closures**: Dict captured in AK.foreachindex fails on GPU. Fix: extract arrays before closure.
   ```julia
   # Wrong
   AK.foreachindex(arr) do idx
       arr[idx] = images[:xy_full][idx]  # Dict is not bits type!
   end

   # Correct
   xy_full = images[:xy_full]  # Extract first
   AK.foreachindex(arr) do idx
       arr[idx] = xy_full[idx]
   end
   ```

2. **Permutation for grid_sample**: Velocity field is (X, Y, Z, 3, N), but grid_sample expects grid as (3, X, Y, Z, N). Implemented `_permute_velocity_to_grid!` and `_permute_grid_to_velocity!`.

3. **Identity grid creation**: Created on CPU then copied to GPU to avoid scalar indexing.

#### Mooncake Integration

All differentiable functions registered as primitives:
```julia
@is_primitive MinimalCtx Tuple{typeof(spatial_transform), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}
@is_primitive MinimalCtx Tuple{typeof(diffeomorphic_transform), AbstractArray{<:Any,5}}
@is_primitive MinimalCtx Tuple{typeof(composition_transform), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}
@is_primitive MinimalCtx Tuple{typeof(linear_elasticity), AbstractArray{<:Any,5}}
```

Each has corresponding `rrule!!` with GPU-compatible backward pass.

#### Test Results

All 84 tests passed on Metal GPU:
```
Test Summary:    | Pass  Total     Time
SyN Registration |   84     84  1m23.7s
```

Test coverage:
- ✓ spatial_transform: identity, small displacement
- ✓ diffeomorphic_transform: zero velocity, small/large velocity, different time_steps, batch support
- ✓ composition_transform: identity, self-composition, compose with zero
- ✓ gauss_smoothing: small sigma, variance reduction, batch support
- ✓ linear_elasticity: zero/non-zero flow
- ✓ SyNRegistration: constructor, custom params, reset!
- ✓ fit!: basic 3D, loss tracking
- ✓ apply_flows: output shapes, zero velocity
- ✓ transform: forward/inverse after fit
- ✓ Diffeomorphism properties: inverse composition, smooth output
- ✓ Full registration: register function

#### Acceptance Criteria Status
- ✓ src/syn.jl with SyNRegistration struct
- ✓ diffeomorphic_transform using AK.jl
- ✓ Mooncake rrule!! for all custom ops
- ✓ Converges on test cases with MtlArrays

---
