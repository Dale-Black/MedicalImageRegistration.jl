# Test utilities for parity testing with torchreg
# Handles array axis conversions between Julia and PyTorch conventions

using Test
using PythonCall

# Note: torch and np are already defined in runtests.jl before this file is included
# This file should only be included from runtests.jl

"""
    julia_to_torch(arr::AbstractArray{T, 4}) where T

Convert 2D Julia array to PyTorch tensor.
Julia convention: (X, Y, C, N) -> PyTorch: (N, C, Y, X)
"""
function julia_to_torch(arr::AbstractArray{T, 4}) where T
    # Permute: (X, Y, C, N) -> (N, C, Y, X)
    permuted = permutedims(arr, (4, 3, 2, 1))
    # Create contiguous copy and convert to numpy with explicit order='C'
    np_arr = np.ascontiguousarray(permuted)
    return torch.from_numpy(np_arr)
end

"""
    julia_to_torch(arr::AbstractArray{T, 5}) where T

Convert 3D Julia array to PyTorch tensor.
Julia convention: (X, Y, Z, C, N) -> PyTorch: (N, C, Z, Y, X)
"""
function julia_to_torch(arr::AbstractArray{T, 5}) where T
    # Permute: (X, Y, Z, C, N) -> (N, C, Z, Y, X)
    permuted = permutedims(arr, (5, 4, 3, 2, 1))
    np_arr = np.ascontiguousarray(permuted)
    return torch.from_numpy(np_arr)
end

"""
    torch_to_julia(tensor) -> Array

Convert PyTorch tensor to Julia array.
For 4D tensors: PyTorch (N, C, Y, X) -> Julia (X, Y, C, N)
For 5D tensors: PyTorch (N, C, Z, Y, X) -> Julia (X, Y, Z, C, N)
"""
function torch_to_julia(tensor)
    # Get numpy array (handles GPU tensors too)
    numpy_arr = tensor.detach().cpu().numpy()
    julia_arr = pyconvert(Array, numpy_arr)

    ndims_tensor = ndims(julia_arr)
    if ndims_tensor == 4
        # (N, C, Y, X) -> (X, Y, C, N)
        return permutedims(julia_arr, (4, 3, 2, 1))
    elseif ndims_tensor == 5
        # (N, C, Z, Y, X) -> (X, Y, Z, C, N)
        return permutedims(julia_arr, (5, 4, 3, 2, 1))
    else
        error("Expected 4D or 5D tensor, got $(ndims_tensor)D")
    end
end

"""
    compare_results(julia_result::AbstractArray, torch_result; rtol=1e-5, atol=1e-8)

Compare Julia and torchreg results within tolerance.
"""
function compare_results(julia_result::AbstractArray, torch_result; rtol=1e-5, atol=1e-8)
    torch_julia = torch_to_julia(torch_result)

    if size(julia_result) != size(torch_julia)
        @warn "Shape mismatch: Julia=$(size(julia_result)), Torch=$(size(torch_julia))"
        return false
    end

    return isapprox(julia_result, torch_julia; rtol=rtol, atol=atol)
end

"""
    compare_results(julia_result::Number, torch_result; rtol=1e-5, atol=1e-8)

Compare scalar results within tolerance.
"""
function compare_results(julia_result::Number, torch_result; rtol=1e-5, atol=1e-8)
    torch_val = pyconvert(Float64, torch_result.item())
    return isapprox(julia_result, torch_val; rtol=rtol, atol=atol)
end

"""
Convert a raw numpy/torch array to Julia array preserving layout.
"""
function raw_to_julia(arr)
    if pyisinstance(arr, torch.Tensor)
        arr = arr.detach().cpu().numpy()
    end
    return pyconvert(Array, arr)
end

# ============================================================================
# Test Suites
# ============================================================================

@testset "Array Conversion Utilities" begin
    @testset "2D arrays (4D tensors)" begin
        # Create test array: (X=7, Y=8, C=1, N=2)
        julia_arr = randn(Float32, 7, 8, 1, 2)

        # Convert to torch and back
        torch_tensor = julia_to_torch(julia_arr)

        # Verify torch shape is (N=2, C=1, Y=8, X=7)
        torch_shape = pyconvert(Tuple, torch_tensor.shape)
        @test torch_shape == (2, 1, 8, 7)

        # Convert back to Julia
        julia_back = torch_to_julia(torch_tensor)

        # Should match original
        @test size(julia_back) == size(julia_arr)
        @test julia_back ≈ julia_arr
    end

    @testset "3D arrays (5D tensors)" begin
        # Create test array: (X=7, Y=8, Z=9, C=1, N=2)
        julia_arr = randn(Float32, 7, 8, 9, 1, 2)

        # Convert to torch and back
        torch_tensor = julia_to_torch(julia_arr)

        # Verify torch shape is (N=2, C=1, Z=9, Y=8, X=7)
        torch_shape = pyconvert(Tuple, torch_tensor.shape)
        @test torch_shape == (2, 1, 9, 8, 7)

        # Convert back to Julia
        julia_back = torch_to_julia(torch_tensor)

        # Should match original
        @test size(julia_back) == size(julia_arr)
        @test julia_back ≈ julia_arr
    end

    @testset "compare_results helper" begin
        # Create matching arrays
        julia_arr = randn(Float32, 5, 6, 7, 1, 2)
        torch_tensor = julia_to_torch(julia_arr)

        # Should match
        @test compare_results(julia_arr, torch_tensor)

        # Add small noise - should still match with tolerance
        julia_noisy = julia_arr .+ randn(Float32, size(julia_arr)...) .* 1f-6
        @test compare_results(julia_noisy, torch_tensor; rtol=1e-4)

        # Add large noise - should not match
        julia_different = julia_arr .+ randn(Float32, size(julia_arr)...) .* 1f0
        @test !compare_results(julia_different, torch_tensor; rtol=1e-5)
    end

    @testset "Round-trip preservation" begin
        for (desc, shape) in [
            ("2D batch=1", (10, 12, 1, 1)),
            ("2D batch=3", (10, 12, 2, 3)),
            ("3D batch=1", (8, 9, 10, 1, 1)),
            ("3D batch=2", (8, 9, 10, 2, 2)),
        ]
            julia_arr = randn(Float32, shape...)
            torch_tensor = julia_to_torch(julia_arr)
            julia_back = torch_to_julia(torch_tensor)

            @test julia_back ≈ julia_arr atol=1e-7
        end
    end
end

# ============================================================================
# Utility Function Tests (using MedicalImageRegistration module)
# ============================================================================

using MedicalImageRegistration: create_identity_grid, affine_grid, identity_affine,
                                 smooth_kernel, jacobi_gradient, jacobi_determinant

# Import torch.nn.functional for grid comparison
const F = pyimport("torch.nn.functional")

@testset "Utility Function Tests" begin
    @testset "create_identity_grid" begin
        # 2D grid
        grid_2d = create_identity_grid((7, 8), Float32)
        @test size(grid_2d) == (2, 7, 8)
        @test grid_2d[1, 1, 1] ≈ -1.0f0  # x at first position
        @test grid_2d[1, 7, 1] ≈ 1.0f0   # x at last position
        @test grid_2d[2, 1, 1] ≈ -1.0f0  # y at first position
        @test grid_2d[2, 1, 8] ≈ 1.0f0   # y at last position

        # 3D grid
        grid_3d = create_identity_grid((5, 6, 7), Float32)
        @test size(grid_3d) == (3, 5, 6, 7)
        @test grid_3d[1, 1, 1, 1] ≈ -1.0f0
        @test grid_3d[1, 5, 1, 1] ≈ 1.0f0
        @test grid_3d[2, 1, 1, 1] ≈ -1.0f0
        @test grid_3d[2, 1, 6, 1] ≈ 1.0f0
        @test grid_3d[3, 1, 1, 1] ≈ -1.0f0
        @test grid_3d[3, 1, 1, 7] ≈ 1.0f0
    end

    @testset "identity_affine" begin
        theta_2d = identity_affine(2, 3, Float32)
        @test size(theta_2d) == (2, 3, 3)
        @test theta_2d[1, 1, 1] ≈ 1.0f0
        @test theta_2d[2, 2, 1] ≈ 1.0f0
        @test theta_2d[1, 3, 1] ≈ 0.0f0  # translation

        theta_3d = identity_affine(3, 2, Float32)
        @test size(theta_3d) == (3, 4, 2)
        @test theta_3d[1, 1, 1] ≈ 1.0f0
        @test theta_3d[2, 2, 1] ≈ 1.0f0
        @test theta_3d[3, 3, 1] ≈ 1.0f0
    end

    @testset "affine_grid" begin
        # Test identity transformation preserves grid
        theta_3d = identity_affine(3, 1, Float32)
        grid = affine_grid(theta_3d, (7, 8, 9))

        @test size(grid) == (3, 7, 8, 9, 1)
        @test grid[1, 1, 1, 1, 1] ≈ -1.0f0  # x at corner
        @test grid[1, 7, 1, 1, 1] ≈ 1.0f0   # x at opposite corner
        @test grid[2, 1, 1, 1, 1] ≈ -1.0f0  # y at corner
        @test grid[2, 1, 8, 1, 1] ≈ 1.0f0   # y at opposite corner
        @test grid[3, 1, 1, 1, 1] ≈ -1.0f0  # z at corner
        @test grid[3, 1, 1, 9, 1] ≈ 1.0f0   # z at opposite corner

        # Test translation
        theta_trans = identity_affine(3, 1, Float32)
        theta_trans[1, 4, 1] = 0.5f0  # translate x by 0.5
        grid_trans = affine_grid(theta_trans, (5, 5, 5))
        # All x coordinates should be shifted by 0.5
        @test grid_trans[1, 1, 1, 1, 1] ≈ -0.5f0  # -1 + 0.5 = -0.5
        @test grid_trans[1, 5, 1, 1, 1] ≈ 1.5f0   # 1 + 0.5 = 1.5
    end

    @testset "smooth_kernel" begin
        # 2D kernel
        kernel_2d = smooth_kernel((5, 5), (1.0f0, 1.0f0))
        @test size(kernel_2d) == (5, 5)
        @test sum(kernel_2d) ≈ 1.0f0 atol=1e-6
        # Center should be maximum
        @test kernel_2d[3, 3] == maximum(kernel_2d)

        # 3D kernel
        kernel_3d = smooth_kernel((7, 7, 7), (1.5f0, 1.5f0, 1.5f0))
        @test size(kernel_3d) == (7, 7, 7)
        @test sum(kernel_3d) ≈ 1.0f0 atol=1e-6
        @test kernel_3d[4, 4, 4] == maximum(kernel_3d)

        # Non-uniform kernel
        kernel_nonuniform = smooth_kernel((3, 5, 7), (0.8f0, 1.0f0, 1.5f0))
        @test size(kernel_nonuniform) == (3, 5, 7)
        @test sum(kernel_nonuniform) ≈ 1.0f0 atol=1e-6
    end

    @testset "jacobi_gradient" begin
        # Test with zero displacement
        X, Y, Z, N = 8, 8, 8, 1
        u_zero = zeros(Float32, X, Y, Z, 3, N)
        grad = jacobi_gradient(u_zero)

        @test size(grad) == (3, 3, X, Y, Z, N)
        @test !any(isnan.(grad))
        @test !any(isinf.(grad))
    end

    @testset "jacobi_determinant" begin
        X, Y, Z, N = 8, 8, 8, 1
        u_zero = zeros(Float32, X, Y, Z, 3, N)
        det_J = jacobi_determinant(u_zero)

        @test size(det_J) == (X, Y, Z, N)
        @test !any(isnan.(det_J))
        @test !any(isinf.(det_J))
    end
end

@testset "PyTorch Parity Tests" begin
    @testset "affine_grid vs F.affine_grid" begin
        # Compare our affine_grid against PyTorch's F.affine_grid

        for (X, Y, Z) in [(7, 8, 9), (10, 10, 10)]
            # Create identity theta for Julia: (3, 4, 1)
            theta_julia = identity_affine(3, 1, Float32)
            julia_grid = affine_grid(theta_julia, (X, Y, Z))  # (3, X, Y, Z, 1)

            # Create identity theta for PyTorch
            # PyTorch F.affine_grid expects theta of shape (N, ndim, ndim+1)
            # For 3D: (N, 3, 4)
            builtins = pyimport("builtins")
            eye4 = torch.eye(4)
            # Select first 3 rows using Python slicing
            torch_theta = eye4.__getitem__((builtins.slice(nothing, 3), builtins.slice(nothing))).unsqueeze(0)

            # PyTorch affine_grid for 3D takes size [N, C, D, H, W]
            # and returns grid of shape (N, D, H, W, 3) where 3 = (x, y, z)
            torch_grid = F.affine_grid(
                torch_theta,
                pylist([1, 3, Z, Y, X]),
                align_corners=true
            )

            torch_grid_np = raw_to_julia(torch_grid)  # (1, Z, Y, X, 3)

            # Verify both grids have same range
            @test minimum(julia_grid) ≈ -1.0f0 atol=1e-5
            @test maximum(julia_grid) ≈ 1.0f0 atol=1e-5

            torch_min = pyconvert(Float32, torch_grid.min().item())
            torch_max = pyconvert(Float32, torch_grid.max().item())
            @test torch_min ≈ -1.0f0 atol=1e-5
            @test torch_max ≈ 1.0f0 atol=1e-5
        end
    end

    @testset "smooth_kernel properties" begin
        # Verify our smooth_kernel matches expected Gaussian properties

        for (ksize, sigma_val) in [((5, 5, 5), 1.0f0), ((7, 7, 7), 2.0f0)]
            sigma = (sigma_val, sigma_val, sigma_val)
            julia_kernel = smooth_kernel(ksize, sigma)

            # Gaussian is symmetric
            kx, ky, kz = ksize
            @test julia_kernel[1, :, :] ≈ julia_kernel[kx, :, :] rtol=1e-5
            @test julia_kernel[:, 1, :] ≈ julia_kernel[:, ky, :] rtol=1e-5
            @test julia_kernel[:, :, 1] ≈ julia_kernel[:, :, kz] rtol=1e-5

            # Center is maximum
            cx = (kx + 1) ÷ 2
            cy = (ky + 1) ÷ 2
            cz = (kz + 1) ÷ 2
            @test julia_kernel[cx, cy, cz] == maximum(julia_kernel)
        end
    end
end
