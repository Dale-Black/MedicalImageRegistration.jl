# GPU-accelerated loss functions with Mooncake AD support
# Uses AcceleratedKernels.jl for cross-platform GPU execution
# Dependencies imported by parent module: AK, Mooncake, CoDual, NoFData, NoRData, @is_primitive, MinimalCtx, Atomix

# ============================================================================
# Helper: GPU-compatible scalar extraction
# ============================================================================

# Extract scalar from 1-element GPU array without scalar indexing
@inline function _extract_scalar(arr::AbstractArray{T}) where T
    # Use reduce to extract the single value - works on GPU
    return AK.reduce(+, arr; init=zero(T))
end

# ============================================================================
# MSE Loss - Mean Squared Error
# ============================================================================

"""
    mse_loss(pred, target)

Compute the mean squared error between `pred` and `target`.

# Arguments
- `pred`: Predicted array of any shape
- `target`: Target array of same shape as pred

# Returns
- Scalar MSE value (as 1-element array for GPU compatibility)

# Example
```julia
pred = MtlArray(rand(Float32, 64, 64, 1, 1))
target = MtlArray(rand(Float32, 64, 64, 1, 1))
loss = mse_loss(pred, target)
```
"""
function mse_loss(pred::AbstractArray{T}, target::AbstractArray{T}) where T
    @assert size(pred) == size(target) "Size mismatch: pred $(size(pred)) vs target $(size(target))"
    return _mse_loss_impl(pred, target)
end

function _mse_loss_impl(pred::AbstractArray{T}, target::AbstractArray{T}) where T
    # Compute squared differences
    n = length(pred)
    sq_diff = similar(pred)

    AK.foreachindex(sq_diff) do idx
        diff = @inbounds pred[idx] - target[idx]
        @inbounds sq_diff[idx] = diff * diff
    end

    # GPU-compatible reduction for sum
    total = AK.reduce(+, sq_diff; init=zero(T))

    # Return as 1-element array to stay on GPU
    result = similar(pred, 1)
    fill!(result, total / T(n))
    return result
end

# Backward pass for MSE loss
function _∇mse_loss!(d_pred, d_target, d_output_arr, pred, target)
    T = eltype(pred)
    n = T(length(pred))

    # Extract scalar from d_output array using GPU-compatible method
    d_output = _extract_scalar(d_output_arr)
    scale = T(2) * d_output / n

    AK.foreachindex(d_pred) do idx
        diff = @inbounds pred[idx] - target[idx]
        @inbounds d_pred[idx] += scale * diff
        @inbounds d_target[idx] += -scale * diff
    end
    return nothing
end

# Mooncake rrule!! for mse_loss
@is_primitive MinimalCtx Tuple{typeof(mse_loss), AbstractArray, AbstractArray}

function Mooncake.rrule!!(
    ::CoDual{typeof(mse_loss)},
    pred::CoDual{P, FP},
    target::CoDual{G, FG}
) where {P<:AbstractArray, FP, G<:AbstractArray, FG}
    pred_primal = pred.x
    target_primal = target.x
    pred_fdata = pred.dx
    target_fdata = target.dx
    T = eltype(pred_primal)

    # Forward pass
    output = mse_loss(pred_primal, target_primal)
    output_fdata = similar(output)
    fill!(output_fdata, zero(T))

    function mse_loss_pullback(_rdata)
        # Pass the whole fdata array - we'll extract scalar inside
        _∇mse_loss!(pred_fdata, target_fdata, output_fdata, pred_primal, target_primal)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), mse_loss_pullback
end


# ============================================================================
# Dice Score and Dice Loss
# ============================================================================

"""
    dice_score(pred, target)

Compute the Dice coefficient (Sørensen–Dice coefficient) between `pred` and `target`.

The Dice score is computed as:
    2 * sum(pred * target) / sum(pred + target)

Summed over all elements.

# Arguments
- `pred`: Predicted array of shape (X, Y, [Z], C, N)
- `target`: Target array of same shape

# Returns
- Scalar Dice score value (between 0 and 1), as 1-element array

# Note
For binary segmentation masks, pred and target should contain values in [0, 1].
For soft masks (probabilities), this computes the soft Dice score.
"""
function dice_score(pred::AbstractArray{T,4}, target::AbstractArray{T,4}) where T
    @assert size(pred) == size(target) "Size mismatch: pred $(size(pred)) vs target $(size(target))"
    return _dice_score_impl(pred, target)
end

function dice_score(pred::AbstractArray{T,5}, target::AbstractArray{T,5}) where T
    @assert size(pred) == size(target) "Size mismatch: pred $(size(pred)) vs target $(size(target))"
    return _dice_score_impl(pred, target)
end

function _dice_score_impl(pred::AbstractArray{T}, target::AbstractArray{T}) where T
    # Compute intersection and union element-wise
    inter = similar(pred)
    union_arr = similar(pred)

    AK.foreachindex(pred) do idx
        p = @inbounds pred[idx]
        t = @inbounds target[idx]
        @inbounds inter[idx] = p * t
        @inbounds union_arr[idx] = p + t
    end

    # GPU-compatible reduction
    inter_sum = AK.reduce(+, inter; init=zero(T))
    union_sum = AK.reduce(+, union_arr; init=zero(T))

    # Compute Dice score: 2 * intersection / union
    eps = T(1e-7)  # Avoid division by zero
    dice = T(2) * inter_sum / (union_sum + eps)

    result = similar(pred, 1)
    fill!(result, dice)
    return result
end

# Backward pass for dice_score
# d(dice)/d(pred) = d(2*inter/union)/d(pred[i])
#   = 2*d(inter)/d(pred[i])/union - 2*inter*d(union)/d(pred[i])/union^2
#   = 2*t/union - 2*inter/union^2
#   = 2*(t*union - inter)/union^2
function _∇dice_score!(d_pred, d_target, d_output_arr, pred, target, inter_sum::T, union_sum::T) where T
    eps = T(1e-7)
    union_safe = union_sum + eps
    two = T(2)

    # Extract upstream gradient
    d_output = _extract_scalar(d_output_arr)

    AK.foreachindex(d_pred) do idx
        p = @inbounds pred[idx]
        t = @inbounds target[idx]

        d_pred_i = two * (t * union_safe - inter_sum) / (union_safe * union_safe)
        d_target_i = two * (p * union_safe - inter_sum) / (union_safe * union_safe)

        @inbounds d_pred[idx] += d_output * d_pred_i
        @inbounds d_target[idx] += d_output * d_target_i
    end
    return nothing
end

# Mooncake rrule!! for dice_score
@is_primitive MinimalCtx Tuple{typeof(dice_score), AbstractArray{<:Any,4}, AbstractArray{<:Any,4}}
@is_primitive MinimalCtx Tuple{typeof(dice_score), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}

function Mooncake.rrule!!(
    ::CoDual{typeof(dice_score)},
    pred::CoDual{P, FP},
    target::CoDual{G, FG}
) where {P<:AbstractArray, FP, G<:AbstractArray, FG}
    pred_primal = pred.x
    target_primal = target.x
    pred_fdata = pred.dx
    target_fdata = target.dx
    T = eltype(pred_primal)

    # Compute inter and union for both forward and backward
    inter = similar(pred_primal)
    union_arr = similar(pred_primal)

    AK.foreachindex(pred_primal) do idx
        p = @inbounds pred_primal[idx]
        t = @inbounds target_primal[idx]
        @inbounds inter[idx] = p * t
        @inbounds union_arr[idx] = p + t
    end

    inter_sum = AK.reduce(+, inter; init=zero(T))
    union_sum = AK.reduce(+, union_arr; init=zero(T))

    eps = T(1e-7)
    dice = T(2) * inter_sum / (union_sum + eps)

    output = similar(pred_primal, 1)
    fill!(output, dice)
    output_fdata = similar(output)
    fill!(output_fdata, zero(T))

    function dice_score_pullback(_rdata)
        _∇dice_score!(pred_fdata, target_fdata, output_fdata, pred_primal, target_primal, inter_sum, union_sum)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), dice_score_pullback
end

"""
    dice_loss(pred, target)

Compute the Dice loss between `pred` and `target`.

    dice_loss = 1 - dice_score

# Arguments
- `pred`: Predicted array of shape (X, Y, [Z], C, N)
- `target`: Target array of same shape

# Returns
- Scalar Dice loss value (between 0 and 1), as 1-element array
"""
function dice_loss(pred::AbstractArray{T,4}, target::AbstractArray{T,4}) where T
    score = dice_score(pred, target)
    result = similar(score)
    # Use GPU-compatible operation
    AK.foreachindex(result) do idx
        @inbounds result[idx] = one(T) - score[idx]
    end
    return result
end

function dice_loss(pred::AbstractArray{T,5}, target::AbstractArray{T,5}) where T
    score = dice_score(pred, target)
    result = similar(score)
    AK.foreachindex(result) do idx
        @inbounds result[idx] = one(T) - score[idx]
    end
    return result
end

# Mooncake rrule!! for dice_loss
@is_primitive MinimalCtx Tuple{typeof(dice_loss), AbstractArray{<:Any,4}, AbstractArray{<:Any,4}}
@is_primitive MinimalCtx Tuple{typeof(dice_loss), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}

function Mooncake.rrule!!(
    ::CoDual{typeof(dice_loss)},
    pred::CoDual{P, FP},
    target::CoDual{G, FG}
) where {P<:AbstractArray, FP, G<:AbstractArray, FG}
    pred_primal = pred.x
    target_primal = target.x
    pred_fdata = pred.dx
    target_fdata = target.dx
    T = eltype(pred_primal)

    # Compute inter and union
    inter = similar(pred_primal)
    union_arr = similar(pred_primal)

    AK.foreachindex(pred_primal) do idx
        p = @inbounds pred_primal[idx]
        t = @inbounds target_primal[idx]
        @inbounds inter[idx] = p * t
        @inbounds union_arr[idx] = p + t
    end

    inter_sum = AK.reduce(+, inter; init=zero(T))
    union_sum = AK.reduce(+, union_arr; init=zero(T))

    eps = T(1e-7)
    dice = T(2) * inter_sum / (union_sum + eps)

    output = similar(pred_primal, 1)
    fill!(output, one(T) - dice)
    output_fdata = similar(output)
    fill!(output_fdata, zero(T))

    function dice_loss_pullback(_rdata)
        # Gradient of 1-dice is -1 * gradient of dice
        # We need to negate the upstream gradient
        neg_fdata = similar(output_fdata)
        AK.foreachindex(neg_fdata) do idx
            @inbounds neg_fdata[idx] = -output_fdata[idx]
        end
        _∇dice_score!(pred_fdata, target_fdata, neg_fdata, pred_primal, target_primal, inter_sum, union_sum)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), dice_loss_pullback
end


# ============================================================================
# NCC Loss - Normalized Cross Correlation (Local)
# ============================================================================

"""
    ncc_loss(pred, target; kernel_size=7, eps_num=1e-5, eps_denom=1e-5)

Compute the local Normalized Cross Correlation (NCC) loss between `pred` and `target`.

This computes NCC in local windows (defined by kernel_size) and returns the negative
mean NCC as a loss (minimizing this maximizes correlation).

# Arguments
- `pred`: Predicted array of shape (X, Y, Z, C, N) - 3D only for now
- `target`: Target array of same shape

# Keyword Arguments
- `kernel_size`: Size of local window for computing NCC (default: 7)
- `eps_num`: Epsilon for numerator numerical stability (default: 1e-5)
- `eps_denom`: Epsilon for denominator numerical stability (default: 1e-5)

# Returns
- Scalar NCC loss value (negative mean local NCC), as 1-element array

# Note
Currently only supports 3D inputs. The NCC is computed as:
    CC = (cross^2 + eps_num) / (var_pred * var_target + eps_denom)
where cross, var_pred, var_target are computed in local windows.
"""
function ncc_loss(
    pred::AbstractArray{T,5},
    target::AbstractArray{T,5};
    kernel_size::Int=7,
    eps_num::T=T(1e-5),
    eps_denom::T=T(1e-5)
) where T
    @assert size(pred) == size(target) "Size mismatch: pred $(size(pred)) vs target $(size(target))"
    return _ncc_loss_3d(pred, target, kernel_size, eps_num, eps_denom)
end

# Helper: compute local sum using AK.foreachindex
# This is a simple box filter implementation
function _local_sum_3d!(output::AbstractArray{T,5}, input::AbstractArray{T,5}, kernel_size::Int) where T
    X, Y, Z, C, N = size(input)
    half_k = kernel_size ÷ 2

    fill!(output, zero(T))

    AK.foreachindex(output) do idx
        # Convert linear index to (i, j, k, c, n)
        i, j, k, c, n = _linear_to_cartesian_5d_metrics(idx, X, Y, Z, C)

        # Sum over kernel window
        sum_val = zero(T)
        for kk in -half_k:half_k
            for jj in -half_k:half_k
                for ii in -half_k:half_k
                    # Clamp to bounds (border padding)
                    i_in = clamp(i + ii, 1, X)
                    j_in = clamp(j + jj, 1, Y)
                    k_in = clamp(k + kk, 1, Z)
                    sum_val += @inbounds input[i_in, j_in, k_in, c, n]
                end
            end
        end
        @inbounds output[idx] = sum_val
    end
    return nothing
end

@inline function _linear_to_cartesian_5d_metrics(idx::Int, X::Int, Y::Int, Z::Int, C::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    idx_0 = idx_0 ÷ Z
    c = idx_0 % C + 1
    n = idx_0 ÷ C + 1
    return i, j, k, c, n
end

function _ncc_loss_3d(
    pred::AbstractArray{T,5},
    target::AbstractArray{T,5},
    kernel_size::Int,
    eps_num::T,
    eps_denom::T
) where T
    X, Y, Z, C, N = size(pred)
    kernel_vol = T(kernel_size^3)

    # Allocate intermediate arrays
    t_sum = similar(target)
    p_sum = similar(pred)
    t2_sum = similar(target)
    p2_sum = similar(pred)
    tp_sum = similar(pred)

    # t_sum = local_sum(target)
    _local_sum_3d!(t_sum, target, kernel_size)

    # p_sum = local_sum(pred)
    _local_sum_3d!(p_sum, pred, kernel_size)

    # t2_sum = local_sum(target^2)
    t2 = similar(target)
    AK.foreachindex(t2) do idx
        t = @inbounds target[idx]
        @inbounds t2[idx] = t * t
    end
    _local_sum_3d!(t2_sum, t2, kernel_size)

    # p2_sum = local_sum(pred^2)
    p2 = similar(pred)
    AK.foreachindex(p2) do idx
        p = @inbounds pred[idx]
        @inbounds p2[idx] = p * p
    end
    _local_sum_3d!(p2_sum, p2, kernel_size)

    # tp_sum = local_sum(target * pred)
    tp = similar(pred)
    AK.foreachindex(tp) do idx
        @inbounds tp[idx] = target[idx] * pred[idx]
    end
    _local_sum_3d!(tp_sum, tp, kernel_size)

    # Compute NCC per element
    cc_arr = similar(pred)
    total_count = T(length(pred))

    AK.foreachindex(cc_arr) do idx
        t_s = @inbounds t_sum[idx]
        p_s = @inbounds p_sum[idx]
        t2_s = @inbounds t2_sum[idx]
        p2_s = @inbounds p2_sum[idx]
        tp_s = @inbounds tp_sum[idx]

        # cross = tp_sum - t_sum * p_sum / kernel_vol
        cross = tp_s - t_s * p_s / kernel_vol

        # t_var = max(0, t2_sum - t_sum^2 / kernel_vol)
        t_var = max(zero(T), t2_s - t_s * t_s / kernel_vol)

        # p_var = max(0, p2_sum - p_sum^2 / kernel_vol)
        p_var = max(zero(T), p2_s - p_s * p_s / kernel_vol)

        # cc = (cross^2 + eps_num) / (t_var * p_var + eps_denom)
        cc = (cross * cross + eps_num) / (t_var * p_var + eps_denom)

        @inbounds cc_arr[idx] = cc
    end

    # GPU-compatible sum
    ncc_sum = AK.reduce(+, cc_arr; init=zero(T))

    # Return negative mean (loss to minimize)
    result = similar(pred, 1)
    fill!(result, -ncc_sum / total_count)
    return result
end

# Backward pass for NCC loss
function _∇ncc_loss_3d!(
    d_pred::AbstractArray{T,5},
    d_target::AbstractArray{T,5},
    d_output_arr,
    pred::AbstractArray{T,5},
    target::AbstractArray{T,5},
    kernel_size::Int,
    eps_num::T,
    eps_denom::T
) where T
    X, Y, Z, C, N = size(pred)
    half_k = kernel_size ÷ 2
    kernel_vol = T(kernel_size^3)
    total_count = T(length(pred))

    # Extract upstream gradient
    d_output = _extract_scalar(d_output_arr)

    # Pre-compute local sums (needed for backward)
    t_sum = similar(target)
    p_sum = similar(pred)
    t2_sum = similar(target)
    p2_sum = similar(pred)
    tp_sum = similar(pred)

    _local_sum_3d!(t_sum, target, kernel_size)
    _local_sum_3d!(p_sum, pred, kernel_size)

    t2 = similar(target)
    AK.foreachindex(t2) do idx
        t = @inbounds target[idx]
        @inbounds t2[idx] = t * t
    end
    _local_sum_3d!(t2_sum, t2, kernel_size)

    p2 = similar(pred)
    AK.foreachindex(p2) do idx
        p = @inbounds pred[idx]
        @inbounds p2[idx] = p * p
    end
    _local_sum_3d!(p2_sum, p2, kernel_size)

    tp = similar(pred)
    AK.foreachindex(tp) do idx
        @inbounds tp[idx] = target[idx] * pred[idx]
    end
    _local_sum_3d!(tp_sum, tp, kernel_size)

    # Scale factor from mean and negative
    scale = -d_output / total_count

    # Compute gradients
    AK.foreachindex(d_pred) do idx
        i, j, k, c, n = _linear_to_cartesian_5d_metrics(idx, X, Y, Z, C)

        p_val = @inbounds pred[idx]
        t_val = @inbounds target[idx]

        d_pred_acc = zero(T)
        d_target_acc = zero(T)

        # This input affects all output positions within half_k
        for kk in -half_k:half_k
            for jj in -half_k:half_k
                for ii in -half_k:half_k
                    i_out = i + ii
                    j_out = j + jj
                    k_out = k + kk

                    # Skip if output position is out of bounds
                    if i_out < 1 || i_out > X || j_out < 1 || j_out > Y || k_out < 1 || k_out > Z
                        continue
                    end

                    # Get local sums at output position
                    out_idx = i_out + (j_out - 1) * X + (k_out - 1) * X * Y + (c - 1) * X * Y * Z + (n - 1) * X * Y * Z * C

                    t_s = @inbounds t_sum[out_idx]
                    p_s = @inbounds p_sum[out_idx]
                    t2_s = @inbounds t2_sum[out_idx]
                    p2_s = @inbounds p2_sum[out_idx]
                    tp_s = @inbounds tp_sum[out_idx]

                    cross = tp_s - t_s * p_s / kernel_vol
                    t_var = max(zero(T), t2_s - t_s * t_s / kernel_vol)
                    p_var = max(zero(T), p2_s - p_s * p_s / kernel_vol)
                    denom = t_var * p_var + eps_denom

                    # d(cross)/d(pred[idx]) = t_val - t_s / kernel_vol
                    d_cross_d_pred = t_val - t_s / kernel_vol

                    # d(p_var)/d(pred[idx]) = 2*pred[idx] - 2*p_s/kernel_vol (if p_var > 0)
                    d_pvar_d_pred = p_var > zero(T) ? T(2) * p_val - T(2) * p_s / kernel_vol : zero(T)

                    num = cross * cross + eps_num
                    d_num_d_pred = T(2) * cross * d_cross_d_pred
                    d_denom_d_pred = t_var * d_pvar_d_pred

                    d_cc_d_pred = d_num_d_pred / denom - num * d_denom_d_pred / (denom * denom)

                    d_pred_acc += scale * d_cc_d_pred

                    # Similarly for target
                    d_cross_d_target = p_val - p_s / kernel_vol
                    d_tvar_d_target = t_var > zero(T) ? T(2) * t_val - T(2) * t_s / kernel_vol : zero(T)
                    d_num_d_target = T(2) * cross * d_cross_d_target
                    d_denom_d_target = p_var * d_tvar_d_target

                    d_cc_d_target = d_num_d_target / denom - num * d_denom_d_target / (denom * denom)

                    d_target_acc += scale * d_cc_d_target
                end
            end
        end

        @inbounds d_pred[idx] += d_pred_acc
        @inbounds d_target[idx] += d_target_acc
    end

    return nothing
end

# Mooncake rrule!! for ncc_loss
@is_primitive MinimalCtx Tuple{typeof(ncc_loss), AbstractArray{<:Any,5}, AbstractArray{<:Any,5}}

function Mooncake.rrule!!(
    ::CoDual{typeof(ncc_loss)},
    pred::CoDual{P, FP},
    target::CoDual{G, FG};
    kernel_size::Int=7,
    eps_num=nothing,
    eps_denom=nothing
) where {P<:AbstractArray{<:Any,5}, FP, G<:AbstractArray{<:Any,5}, FG}
    pred_primal = pred.x
    target_primal = target.x
    pred_fdata = pred.dx
    target_fdata = target.dx
    T = eltype(pred_primal)

    # Default epsilon values
    eps_num_val = eps_num === nothing ? T(1e-5) : T(eps_num)
    eps_denom_val = eps_denom === nothing ? T(1e-5) : T(eps_denom)

    # Forward pass
    output = _ncc_loss_3d(pred_primal, target_primal, kernel_size, eps_num_val, eps_denom_val)

    output_fdata = similar(output)
    fill!(output_fdata, zero(T))

    function ncc_loss_pullback(_rdata)
        _∇ncc_loss_3d!(pred_fdata, target_fdata, output_fdata, pred_primal, target_primal,
                       kernel_size, eps_num_val, eps_denom_val)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), ncc_loss_pullback
end
