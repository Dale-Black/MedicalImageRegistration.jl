# Test helpers - Metal availability checking and conditional GPU arrays

using Test

# Guard against multiple includes
if !@isdefined(METAL_AVAILABLE)

# Check Metal availability at load time
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch
    false
end

if METAL_AVAILABLE
    @info "Metal GPU available - running GPU tests"
else
    @info "Metal GPU not available - GPU tests will be skipped"
end

# Helper to create GPU array if Metal is available, otherwise return CPU array
function maybe_gpu(arr::AbstractArray)
    if METAL_AVAILABLE
        return MtlArray(arr)
    else
        return arr
    end
end

# Helper to create zeros on GPU if available
function maybe_gpu_zeros(T::Type, dims...)
    if METAL_AVAILABLE
        return Metal.zeros(T, dims...)
    else
        return zeros(T, dims...)
    end
end

# Macro to conditionally run GPU tests
macro gpu_testset(name, body)
    quote
        if METAL_AVAILABLE
            @testset $(esc(name)) $(esc(body))
        else
            @testset $(esc(name)) begin
                @info "Skipping GPU test: Metal not available"
                @test_skip false
            end
        end
    end
end

# Check if array is a GPU array
function is_gpu_array(arr)
    if METAL_AVAILABLE
        return arr isa MtlArray
    else
        return false
    end
end

end # if !@isdefined(METAL_AVAILABLE)
