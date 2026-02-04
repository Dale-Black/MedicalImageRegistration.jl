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
