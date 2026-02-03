# Test utilities for parity testing with torchreg
# Handles array axis conversions between Julia and PyTorch conventions

using Test
using PythonCall

# Note: torch and np are defined in runtests.jl and used globally
# These functions use the global torch and np variables

"""
    julia_to_torch(arr::AbstractArray{T, 4}) where T

Convert 2D Julia array to PyTorch tensor.
Julia convention: (X, Y, C, N) -> PyTorch: (N, C, Y, X)
"""
function julia_to_torch(arr::AbstractArray{T, 4}) where T
    # Permute: (X, Y, C, N) -> (N, C, Y, X)
    permuted = permutedims(arr, (4, 3, 2, 1))
    # Create contiguous copy and convert to numpy with explicit order='C'
    # Using np.ascontiguousarray ensures proper C-order layout
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
    # Create contiguous copy and convert to numpy with explicit order='C'
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
    # Convert to Julia array
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
Automatically converts torch_result to Julia array if needed.
Returns true if arrays match within tolerance.
"""
function compare_results(julia_result::AbstractArray, torch_result; rtol=1e-5, atol=1e-8)
    # Convert torch result to Julia array
    torch_julia = torch_to_julia(torch_result)

    # Check shapes match
    if size(julia_result) != size(torch_julia)
        @warn "Shape mismatch: Julia=$(size(julia_result)), Torch=$(size(torch_julia))"
        return false
    end

    # Compare values
    return isapprox(julia_result, torch_julia; rtol=rtol, atol=atol)
end

"""
    compare_results(julia_result::Number, torch_result::Number; rtol=1e-5, atol=1e-8)

Compare scalar results within tolerance.
"""
function compare_results(julia_result::Number, torch_result; rtol=1e-5, atol=1e-8)
    torch_val = pyconvert(Float64, torch_result.item())
    return isapprox(julia_result, torch_val; rtol=rtol, atol=atol)
end

# Test the conversion functions
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
        # Test that round-trip preserves data exactly
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
