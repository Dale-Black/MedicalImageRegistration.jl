# Utility functions for MedicalImageRegistration.jl
#
# GPU Acceleration: This package supports GPU computation via NNlib.jl's GPU backends.
# When you pass GPU arrays (CuArray for CUDA, MtlArray for Metal), the heavy operations
# (grid_sample, convolution, batched matrix multiply) automatically run on GPU.
#
# For CPU multithreading, AcceleratedKernels.jl is used for parallel loop execution.

using LinearAlgebra
import AcceleratedKernels as AK

# Helper function to avoid recomputation of constant values in gradient computation
# This replaces Zygote's ignore_derivatives - in manual gradient computation,
# these values are naturally constants that don't require gradients
@inline function _constant(f::Function)
    return f()
end

# ============================================================================
# GPU Utility Functions
# ============================================================================

"""
    to_device(arr::AbstractArray, device_array::AbstractArray)

Convert `arr` to the same array type and device as `device_array`.

This is useful for ensuring auxiliary arrays (like identity grids) are on the
same device as the input data.

# Example
```julia
using CUDA
moving = CuArray(randn(Float32, 64, 64, 64, 1, 1))
grid = create_identity_grid((64, 64, 64), Float32)
gpu_grid = to_device(grid, moving)  # Now on GPU
```
"""
function to_device(arr::AbstractArray, device_array::AbstractArray)
    # If device_array is not a regular Array, convert arr to same type
    if typeof(device_array) <: Array
        return arr
    else
        # Use similar to get the right array type, then copy
        return typeof(device_array)(arr)
    end
end

"""
    get_array_type(arr::AbstractArray)

Get the array constructor type for creating arrays on the same device.

# Example
```julia
using CUDA
x = CuArray(randn(Float32, 10))
ArrayType = get_array_type(x)  # Returns CuArray
y = ArrayType{Float32}(undef, 10)  # Creates CuArray
```
"""
function get_array_type(arr::AbstractArray)
    # For standard arrays, return Array
    return typeof(arr).name.wrapper
end

"""
    create_identity_grid(spatial_size::NTuple{2, Int}, ::Type{T}=Float32) where T

Create a 2D identity grid with normalized coordinates from -1 to 1.

Returns grid of shape `(2, X, Y)` where dim 1 contains (x, y) coordinates.

# GPU Support
The returned grid is a standard Array. For GPU computation, the grid will be
transferred to GPU when used with GPU arrays in operations like grid_sample.
To create a grid directly on GPU, convert after creation:
```julia
using CUDA  # or Metal
grid = create_identity_grid((64, 64), Float32)
gpu_grid = CuArray(grid)  # or MtlArray(grid)
```
"""
function create_identity_grid(spatial_size::NTuple{2, Int}, ::Type{T}=Float32) where T
    X, Y = spatial_size
    # Create 1D coordinate arrays from -1 to 1
    xs = collect(T, range(T(-1), T(1), length=X))
    ys = collect(T, range(T(-1), T(1), length=Y))

    # Create grid using broadcasting (efficient vectorized operation)
    grid = zeros(T, 2, X, Y)
    @inbounds for j in 1:Y, i in 1:X
        grid[1, i, j] = xs[i]  # x coordinate
        grid[2, i, j] = ys[j]  # y coordinate
    end
    return grid
end

"""
    create_identity_grid(spatial_size::NTuple{3, Int}, ::Type{T}=Float32) where T

Create a 3D identity grid with normalized coordinates from -1 to 1.

Returns grid of shape `(3, X, Y, Z)` where dim 1 contains (x, y, z) coordinates.

# GPU Support
The returned grid is a standard Array. For GPU computation, the grid will be
transferred to GPU when used with GPU arrays in operations like grid_sample.
"""
function create_identity_grid(spatial_size::NTuple{3, Int}, ::Type{T}=Float32) where T
    X, Y, Z = spatial_size
    # Create 1D coordinate arrays from -1 to 1
    xs = collect(T, range(T(-1), T(1), length=X))
    ys = collect(T, range(T(-1), T(1), length=Y))
    zs = collect(T, range(T(-1), T(1), length=Z))

    # Create grid using loops (efficient with @inbounds)
    grid = zeros(T, 3, X, Y, Z)
    @inbounds for k in 1:Z, j in 1:Y, i in 1:X
        grid[1, i, j, k] = xs[i]  # x coordinate
        grid[2, i, j, k] = ys[j]  # y coordinate
        grid[3, i, j, k] = zs[k]  # z coordinate
    end
    return grid
end

"""
    affine_grid(theta::AbstractArray{T, 3}, spatial_size::NTuple{2, Int}) where T

Create a 2D sampling grid by applying affine transformation to identity grid.

# Arguments
- `theta`: Affine transformation matrix of shape `(2, 3, N)` where N is batch size.
           Each slice `theta[:, :, n]` is a 2×3 affine matrix.
- `spatial_size`: Tuple `(X, Y)` specifying output grid spatial dimensions.

# Returns
- Grid of shape `(2, X, Y, N)` suitable for `NNlib.grid_sample`.

# Notes
- Coordinates are normalized to [-1, 1] range (align_corners=true semantics).
- For identity transformation: `theta = [1 0 0; 0 1 0]` (per batch element).
"""
function affine_grid(theta::AbstractArray{T, 3}, spatial_size::NTuple{2, Int}) where T
    ndim = 2
    @assert size(theta, 1) == ndim "Expected theta to have $ndim rows, got $(size(theta, 1))"
    @assert size(theta, 2) == ndim + 1 "Expected theta to have $(ndim+1) columns, got $(size(theta, 2))"

    N = size(theta, 3)
    X, Y = spatial_size

    # Create identity grid: (2, X, Y)
    # Use ignore_derivatives since grid creation doesn't depend on theta
    id_grid = _constant() do
        create_identity_grid(spatial_size, T)
    end

    # Flatten spatial dims: (2, X*Y)
    num_points = X * Y
    flat_grid = reshape(id_grid, ndim, num_points)

    # Add homogeneous coordinate: (3, X*Y)
    ones_row = _constant() do
        ones(T, 1, num_points)
    end
    homogeneous_grid = vcat(flat_grid, ones_row)

    # Batched matrix multiply using einsum-like operation
    # theta: (ndim, ndim+1, N), homogeneous_grid: (ndim+1, num_points)
    # result: (ndim, num_points, N)
    # Use NNlib.batched_mul for Zygote compatibility
    # Reshape theta to (ndim, ndim+1, N) and expand homogeneous to (ndim+1, num_points, N)
    homogeneous_batch = repeat(reshape(homogeneous_grid, ndim + 1, num_points, 1), 1, 1, N)
    output_grid = NNlib.batched_mul(theta, homogeneous_batch)

    # Reshape to (2, X, Y, N) for NNlib.grid_sample
    return reshape(output_grid, ndim, X, Y, N)
end

"""
    affine_grid(theta::AbstractArray{T, 3}, spatial_size::NTuple{3, Int}) where T

Create a 3D sampling grid by applying affine transformation to identity grid.

# Arguments
- `theta`: Affine transformation matrix of shape `(3, 4, N)` where N is batch size.
           Each slice `theta[:, :, n]` is a 3×4 affine matrix.
- `spatial_size`: Tuple `(X, Y, Z)` specifying output grid spatial dimensions.

# Returns
- Grid of shape `(3, X, Y, Z, N)` suitable for `NNlib.grid_sample`.

# Notes
- Coordinates are normalized to [-1, 1] range (align_corners=true semantics).
- For identity transformation: `theta = [1 0 0 0; 0 1 0 0; 0 0 1 0]` (per batch element).
"""
function affine_grid(theta::AbstractArray{T, 3}, spatial_size::NTuple{3, Int}) where T
    ndim = 3
    @assert size(theta, 1) == ndim "Expected theta to have $ndim rows, got $(size(theta, 1))"
    @assert size(theta, 2) == ndim + 1 "Expected theta to have $(ndim+1) columns, got $(size(theta, 2))"

    N = size(theta, 3)
    X, Y, Z = spatial_size

    # Create identity grid: (3, X, Y, Z)
    # Use ignore_derivatives since grid creation doesn't depend on theta
    id_grid = _constant() do
        create_identity_grid(spatial_size, T)
    end

    # Flatten spatial dims: (3, X*Y*Z)
    num_points = X * Y * Z
    flat_grid = reshape(id_grid, ndim, num_points)

    # Add homogeneous coordinate: (4, X*Y*Z)
    ones_row = _constant() do
        ones(T, 1, num_points)
    end
    homogeneous_grid = vcat(flat_grid, ones_row)

    # Batched matrix multiply using NNlib.batched_mul for Zygote compatibility
    # theta: (ndim, ndim+1, N), homogeneous_grid: (ndim+1, num_points)
    # result: (ndim, num_points, N)
    homogeneous_batch = repeat(reshape(homogeneous_grid, ndim + 1, num_points, 1), 1, 1, N)
    output_grid = NNlib.batched_mul(theta, homogeneous_batch)

    # Reshape to (3, X, Y, Z, N) for NNlib.grid_sample
    return reshape(output_grid, ndim, X, Y, Z, N)
end

"""
    identity_affine(ndim::Int, batch_size::Int, ::Type{T}=Float32) where T

Create identity affine transformation matrices.

# Returns
- For 2D: Array of shape `(2, 3, batch_size)` with identity transforms.
- For 3D: Array of shape `(3, 4, batch_size)` with identity transforms.
"""
function identity_affine(ndim::Int, batch_size::Int, ::Type{T}=Float32) where T
    theta = zeros(T, ndim, ndim + 1, batch_size)
    @inbounds for n in 1:batch_size
        for i in 1:ndim
            theta[i, i, n] = one(T)
        end
    end
    return theta
end

# ============================================================================
# Gaussian Smoothing Kernel
# ============================================================================

"""
    smooth_kernel(kernel_size::NTuple{2, Int}, sigma::NTuple{2, T}) where T

Create a 2D Gaussian smoothing kernel (separable).

# Arguments
- `kernel_size`: Tuple `(kx, ky)` specifying kernel size in each dimension.
- `sigma`: Tuple `(σx, σy)` specifying standard deviation in each dimension.

# Returns
- Normalized kernel of shape `(kx, ky)` that sums to 1.0.
"""
function smooth_kernel(kernel_size::NTuple{2, Int}, sigma::NTuple{2, T}) where T
    kx, ky = kernel_size
    σx, σy = sigma

    # Create 1D Gaussian arrays
    mean_x = (kx - 1) / T(2)
    mean_y = (ky - 1) / T(2)

    gx = [exp(-((i - 1 - mean_x) / σx)^2 / 2) / (σx * sqrt(T(2π))) for i in 1:kx]
    gy = [exp(-((j - 1 - mean_y) / σy)^2 / 2) / (σy * sqrt(T(2π))) for j in 1:ky]

    # Outer product for 2D kernel
    kernel = zeros(T, kx, ky)
    @inbounds for j in 1:ky, i in 1:kx
        kernel[i, j] = gx[i] * gy[j]
    end

    # Normalize to sum to 1
    return kernel ./ sum(kernel)
end

"""
    smooth_kernel(kernel_size::NTuple{3, Int}, sigma::NTuple{3, T}) where T

Create a 3D Gaussian smoothing kernel (separable).

# Arguments
- `kernel_size`: Tuple `(kx, ky, kz)` specifying kernel size in each dimension.
- `sigma`: Tuple `(σx, σy, σz)` specifying standard deviation in each dimension.

# Returns
- Normalized kernel of shape `(kx, ky, kz)` that sums to 1.0.
"""
function smooth_kernel(kernel_size::NTuple{3, Int}, sigma::NTuple{3, T}) where T
    kx, ky, kz = kernel_size
    σx, σy, σz = sigma

    # Create 1D Gaussian arrays
    mean_x = (kx - 1) / T(2)
    mean_y = (ky - 1) / T(2)
    mean_z = (kz - 1) / T(2)

    gx = [exp(-((i - 1 - mean_x) / σx)^2 / 2) / (σx * sqrt(T(2π))) for i in 1:kx]
    gy = [exp(-((j - 1 - mean_y) / σy)^2 / 2) / (σy * sqrt(T(2π))) for j in 1:ky]
    gz = [exp(-((k - 1 - mean_z) / σz)^2 / 2) / (σz * sqrt(T(2π))) for k in 1:kz]

    # Outer product for 3D kernel
    kernel = zeros(T, kx, ky, kz)
    @inbounds for k in 1:kz, j in 1:ky, i in 1:kx
        kernel[i, j, k] = gx[i] * gy[j] * gz[k]
    end

    # Normalize to sum to 1
    return kernel ./ sum(kernel)
end

"""
    smooth_kernel(kernel_size::NTuple{N, Int}, sigma::AbstractVector{T}) where {N, T}

Create an N-dimensional Gaussian smoothing kernel from a vector sigma.

Convenience method that converts vector sigma to tuple.
"""
function smooth_kernel(kernel_size::NTuple{N, Int}, sigma::AbstractVector{T}) where {N, T}
    @assert length(sigma) == N "sigma must have $N elements, got $(length(sigma))"
    return smooth_kernel(kernel_size, NTuple{N, T}(sigma))
end

# ============================================================================
# Jacobian Gradient and Determinant
# ============================================================================

"""
    jacobi_gradient(u::AbstractArray{T, 5}, id_grid::Union{Nothing, AbstractArray}=nothing) where T

Compute spatial gradients of a displacement field using central differences.

# Arguments
- `u`: Displacement field of shape `(X, Y, Z, 3, N)` in Julia convention.
       The 4th dimension contains (u_x, u_y, u_z) displacement components.
- `id_grid`: Optional identity grid. If not provided, creates one internally.

# Returns
- Gradient tensor of shape `(3, 3, X, Y, Z, N)` where:
  - First index: output direction (∂u_x, ∂u_y, ∂u_z)
  - Second index: derivative direction (∂/∂x, ∂/∂y, ∂/∂z)

# Notes
- Uses central differences with [-0.5, 0, 0.5] kernel.
- Boundary values are replicated.
- Coordinates are scaled from normalized [-1,1] to voxel coordinates.
- Uses AcceleratedKernels.jl for CPU multithreading (GPU arrays not yet supported).
"""
function jacobi_gradient(u::AbstractArray{T, 5}, id_grid::Union{Nothing, AbstractArray}=nothing) where T
    X, Y, Z, C, N = size(u)
    @assert C == 3 "Expected 3 displacement components, got $C"

    # Create identity grid if not provided
    if id_grid === nothing
        id_grid = create_identity_grid((X, Y, Z), T)
    end

    # Scale displacement from normalized to voxel coordinates
    # x_voxel = 0.5 * (u + id_grid) * (size - 1)
    scale_factors = T.([X - 1, Y - 1, Z - 1])

    # Output: (3, 3, X, Y, Z, N) - [output_comp, deriv_dir, spatial..., batch]
    gradients = zeros(T, 3, 3, X, Y, Z, N)

    # Use AcceleratedKernels for parallel iteration over spatial indices
    # Create a linear index array for parallel processing
    linear_indices = [(i, j, k, n, c) for n in 1:N, k in 1:Z, j in 1:Y, i in 1:X, c in 1:3]

    # Use AK.foreachindex for multithreaded CPU execution
    AK.foreachindex(linear_indices) do idx
        i, j, k, n, c = linear_indices[idx]

        # Get scaled value at current position
        x_curr = T(0.5) * (u[i, j, k, c, n] + id_grid[c, i, j, k]) * scale_factors[c]

        # Gradient in x direction (∂/∂x)
        if i == 1
            x_next = T(0.5) * (u[2, j, k, c, n] + id_grid[c, 2, j, k]) * scale_factors[c]
            gradients[c, 1, i, j, k, n] = x_next - x_curr
        elseif i == X
            x_prev = T(0.5) * (u[X-1, j, k, c, n] + id_grid[c, X-1, j, k]) * scale_factors[c]
            gradients[c, 1, i, j, k, n] = x_curr - x_prev
        else
            x_prev = T(0.5) * (u[i-1, j, k, c, n] + id_grid[c, i-1, j, k]) * scale_factors[c]
            x_next = T(0.5) * (u[i+1, j, k, c, n] + id_grid[c, i+1, j, k]) * scale_factors[c]
            gradients[c, 1, i, j, k, n] = T(0.5) * (x_next - x_prev)
        end

        # Gradient in y direction (∂/∂y)
        if j == 1
            y_next = T(0.5) * (u[i, 2, k, c, n] + id_grid[c, i, 2, k]) * scale_factors[c]
            gradients[c, 2, i, j, k, n] = y_next - x_curr
        elseif j == Y
            y_prev = T(0.5) * (u[i, Y-1, k, c, n] + id_grid[c, i, Y-1, k]) * scale_factors[c]
            gradients[c, 2, i, j, k, n] = x_curr - y_prev
        else
            y_prev = T(0.5) * (u[i, j-1, k, c, n] + id_grid[c, i, j-1, k]) * scale_factors[c]
            y_next = T(0.5) * (u[i, j+1, k, c, n] + id_grid[c, i, j+1, k]) * scale_factors[c]
            gradients[c, 2, i, j, k, n] = T(0.5) * (y_next - y_prev)
        end

        # Gradient in z direction (∂/∂z)
        if k == 1
            z_next = T(0.5) * (u[i, j, 2, c, n] + id_grid[c, i, j, 2]) * scale_factors[c]
            gradients[c, 3, i, j, k, n] = z_next - x_curr
        elseif k == Z
            z_prev = T(0.5) * (u[i, j, Z-1, c, n] + id_grid[c, i, j, Z-1]) * scale_factors[c]
            gradients[c, 3, i, j, k, n] = x_curr - z_prev
        else
            z_prev = T(0.5) * (u[i, j, k-1, c, n] + id_grid[c, i, j, k-1]) * scale_factors[c]
            z_next = T(0.5) * (u[i, j, k+1, c, n] + id_grid[c, i, j, k+1]) * scale_factors[c]
            gradients[c, 3, i, j, k, n] = T(0.5) * (z_next - z_prev)
        end
    end

    return gradients
end

"""
    jacobi_determinant(u::AbstractArray{T, 5}, id_grid::Union{Nothing, AbstractArray}=nothing) where T

Compute the determinant of the Jacobian of a displacement field.

# Arguments
- `u`: Displacement field of shape `(X, Y, Z, 3, N)` in Julia convention.
- `id_grid`: Optional identity grid.

# Returns
- Determinant field of shape `(X, Y, Z, N)`.

# Notes
- Uses the Jacobian gradient to compute the 3×3 determinant at each point.
- det(J) > 0 required for diffeomorphism (no folding).
- det(J) ≈ 1 means volume-preserving.
- Uses AcceleratedKernels.jl for CPU multithreading.
"""
function jacobi_determinant(u::AbstractArray{T, 5}, id_grid::Union{Nothing, AbstractArray}=nothing) where T
    X, Y, Z, _, N = size(u)

    # Get gradient tensor: (3, 3, X, Y, Z, N)
    gradient = jacobi_gradient(u, id_grid)

    # Compute determinant at each point
    det_J = zeros(T, X, Y, Z, N)

    # Create linear indices for parallel processing
    linear_indices = [(i, j, k, n) for n in 1:N, k in 1:Z, j in 1:Y, i in 1:X]

    AK.foreachindex(linear_indices) do idx
        i, j, k, n = linear_indices[idx]

        # Extract 3×3 Jacobian matrix at this point
        # J[row, col] = gradient[row, col, i, j, k, n]
        a = gradient[1, 1, i, j, k, n]  # ∂u_x/∂x
        b = gradient[1, 2, i, j, k, n]  # ∂u_x/∂y
        c = gradient[1, 3, i, j, k, n]  # ∂u_x/∂z
        d = gradient[2, 1, i, j, k, n]  # ∂u_y/∂x
        e = gradient[2, 2, i, j, k, n]  # ∂u_y/∂y
        f = gradient[2, 3, i, j, k, n]  # ∂u_y/∂z
        g = gradient[3, 1, i, j, k, n]  # ∂u_z/∂x
        h = gradient[3, 2, i, j, k, n]  # ∂u_z/∂y
        l = gradient[3, 3, i, j, k, n]  # ∂u_z/∂z

        # 3×3 determinant: a(el - fh) - b(dl - fg) + c(dh - eg)
        det_J[i, j, k, n] = a * (e * l - f * h) - b * (d * l - f * g) + c * (d * h - e * g)
    end

    return det_J
end
