# Test utilities for parity testing with torchreg
# This file will be populated by the ralph loop

"""
    julia_to_torch(arr::AbstractArray)

Convert Julia array to PyTorch tensor.
Note: Julia uses column-major (Fortran) order, PyTorch uses row-major (C) order.
Julia convention: (X, Y, Z, C, N) -> PyTorch: (N, C, Z, Y, X)
"""
function julia_to_torch(arr::AbstractArray)
    # TODO: Implement with proper axis permutation
end

"""
    torch_to_julia(tensor)

Convert PyTorch tensor to Julia array.
PyTorch: (N, C, Z, Y, X) -> Julia: (X, Y, Z, C, N)
"""
function torch_to_julia(tensor)
    # TODO: Implement with proper axis permutation
end

"""
    compare_results(julia_result, torch_result; rtol=1e-5, atol=1e-8)

Compare Julia and torchreg results within tolerance.
"""
function compare_results(julia_result, torch_result; rtol=1e-5, atol=1e-8)
    # TODO: Implement comparison
end
