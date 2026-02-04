# GPU-accelerated Mutual Information loss with Mooncake AD support
# Uses differentiable histogram estimation
# Dependencies imported by parent module: AK, Mooncake, CoDual, NoFData, NoRData, @is_primitive, MinimalCtx, Atomix

# ============================================================================
# Mutual Information Loss
# ============================================================================

# When to use MI vs MSE/NCC:
# - MSE: Same modality, similar intensities (e.g., follow-up MRI)
# - NCC: Same modality, different contrast/brightness (e.g., T1 MRI with different scanners)
# - MI: Different modalities or contrast agents (e.g., CT non-contrast vs contrast-enhanced)
#
# For our cardiac CT use case (3mm non-contrast vs 0.5mm contrast):
# - Blood: 40 HU → 300 HU (contrast changes intensity dramatically)
# - MSE would be MAXIMIZED at correct alignment → wrong direction!
# - MI measures statistical DEPENDENCE: if blood is always (40 → 300), MI learns this

"""
    mi_loss(moving, static; num_bins=64, sigma=1.0, intensity_range=nothing)

Compute the negative Mutual Information loss between `moving` and `static` images.

Uses differentiable soft histogram estimation for gradient-based optimization.
Returns negative MI so that minimizing the loss maximizes mutual information
(better alignment).

# Arguments
- `moving`: Moving image array of shape (X, Y, [Z], C, N)
- `static`: Static/reference image array of same shape

# Keyword Arguments
- `num_bins::Int=64`: Number of histogram bins (default 64)
- `sigma::Real=1.0`: Smoothing parameter for soft histogram binning
- `intensity_range::Union{Nothing, Tuple{Real,Real}}=nothing`: If provided, (min, max)
  intensity values for binning. If nothing, automatically computed from data.

# Returns
- Negative MI loss as 1-element array (minimize to maximize alignment)

# Theory
Mutual Information between images X (moving) and Y (static):

    MI(X, Y) = H(X) + H(Y) - H(X, Y)

Where H is Shannon entropy:
- H(X) = -Σ p(x) log p(x)        (marginal entropy of moving)
- H(Y) = -Σ p(y) log p(y)        (marginal entropy of static)
- H(X,Y) = -Σ p(x,y) log p(x,y)  (joint entropy)

Higher MI means better alignment because aligned tissues form tight clusters
in the joint histogram (low joint entropy).

# GPU Acceleration
All operations use AcceleratedKernels.jl and work on GPU arrays (MtlArray, CuArray).

# Example
```julia
moving = MtlArray(rand(Float32, 64, 64, 64, 1, 1) .* 1000 .- 500)  # CT-like HU
static = MtlArray(rand(Float32, 64, 64, 64, 1, 1) .* 1000 .- 500)
loss = mi_loss(moving, static)  # Returns negative MI
```
"""
function mi_loss(
    moving::AbstractArray{T},
    static::AbstractArray{T};
    num_bins::Int=64,
    sigma::Real=T(1),
    intensity_range::Union{Nothing, Tuple{<:Real, <:Real}}=nothing
) where T
    @assert size(moving) == size(static) "Size mismatch: moving $(size(moving)) vs static $(size(static))"
    @assert num_bins >= 4 "num_bins must be at least 4, got $num_bins"
    return _mi_loss_impl(moving, static, num_bins, T(sigma), intensity_range)
end

"""
    nmi_loss(moving, static; num_bins=64, sigma=1.0, intensity_range=nothing)

Compute the negative Normalized Mutual Information loss.

NMI is more robust than MI as it normalizes by the sum of marginal entropies:

    NMI(X, Y) = 2 * MI(X, Y) / (H(X) + H(Y))

This makes NMI range from 0 (independent) to 1 (identical), and is less
sensitive to overlap changes between images.

# Arguments
Same as `mi_loss`.

# Returns
- Negative NMI loss as 1-element array (minimize to maximize alignment)

# Example
```julia
loss = nmi_loss(moving, static)  # Returns negative NMI in range [-1, 0]
```
"""
function nmi_loss(
    moving::AbstractArray{T},
    static::AbstractArray{T};
    num_bins::Int=64,
    sigma::Real=T(1),
    intensity_range::Union{Nothing, Tuple{<:Real, <:Real}}=nothing
) where T
    @assert size(moving) == size(static) "Size mismatch: moving $(size(moving)) vs static $(size(static))"
    @assert num_bins >= 4 "num_bins must be at least 4, got $num_bins"
    return _nmi_loss_impl(moving, static, num_bins, T(sigma), intensity_range)
end

# ============================================================================
# Implementation
# ============================================================================

# Compute min/max from array using GPU-compatible reduction
function _get_minmax(arr::AbstractArray{T}) where T
    min_val = AK.reduce(min, arr; init=typemax(T))
    max_val = AK.reduce(max, arr; init=typemin(T))
    return min_val, max_val
end

# ============================================================================
# Soft Histogram Computation (GPU-friendly)
# ============================================================================

# Instead of nested loops with Parzen windows, we use a simpler approach:
# 1. Compute soft bin assignments using a triangular kernel (B-spline order 1)
# 2. Each pixel contributes to at most 2 adjacent bins
# This avoids dynamic loops in the GPU kernel

# Linear kernel (triangular): max(0, 1 - |x|)
@inline function _linear_kernel(x::T) where T
    abs_x = abs(x)
    return abs_x < one(T) ? one(T) - abs_x : zero(T)
end

# Compute soft bin index and weight for a value
# Returns (bin_lo, bin_hi, weight_lo, weight_hi)
# GPU-friendly: avoids floor(Int, ...) which causes memory allocation
@inline function _soft_bin(val::T, min_val::T, bin_width::T, num_bins::Int) where T
    # Map to [0, num_bins] range
    normalized = (val - min_val) / bin_width
    # Clamp to valid range
    normalized = clamp(normalized, zero(T), T(num_bins - 1) + T(0.999f0))

    # GPU-friendly floor: use trunc on positive values
    normalized_floor = trunc(normalized)

    # Lower and upper bin indices (1-indexed)
    # Use unsafe_trunc to avoid the Int conversion issue on GPU
    bin_lo_f = normalized_floor + one(T)
    bin_hi_f = min(bin_lo_f + one(T), T(num_bins))

    # Convert to Int using unsafe_trunc (GPU safe)
    bin_lo = unsafe_trunc(Int, bin_lo_f)
    bin_hi = unsafe_trunc(Int, bin_hi_f)

    # Fractional part determines weight distribution
    frac = normalized - normalized_floor

    weight_lo = one(T) - frac
    weight_hi = frac

    # Clamp bins to valid range
    bin_lo = clamp(bin_lo, 1, num_bins)
    bin_hi = clamp(bin_hi, 1, num_bins)

    return bin_lo, bin_hi, weight_lo, weight_hi
end

# Compute marginal histograms (1D)
function _compute_marginal_histogram!(
    hist::AbstractArray{T,1},
    img::AbstractArray{T},
    min_val::T,
    bin_width::T,
    num_bins::Int
) where T
    fill!(hist, zero(T))

    AK.foreachindex(img) do idx
        val = @inbounds img[idx]
        bin_lo, bin_hi, weight_lo, weight_hi = _soft_bin(val, min_val, bin_width, num_bins)

        Atomix.@atomic hist[bin_lo] += weight_lo
        if bin_hi != bin_lo
            Atomix.@atomic hist[bin_hi] += weight_hi
        end
    end

    return nothing
end

# Compute joint histogram (2D)
function _compute_joint_histogram!(
    joint_hist::AbstractArray{T,2},
    moving::AbstractArray{T},
    static::AbstractArray{T},
    min_val::T,
    bin_width::T,
    num_bins::Int
) where T
    fill!(joint_hist, zero(T))

    AK.foreachindex(moving) do idx
        m_val = @inbounds moving[idx]
        s_val = @inbounds static[idx]

        m_bin_lo, m_bin_hi, m_weight_lo, m_weight_hi = _soft_bin(m_val, min_val, bin_width, num_bins)
        s_bin_lo, s_bin_hi, s_weight_lo, s_weight_hi = _soft_bin(s_val, min_val, bin_width, num_bins)

        # Contribute to up to 4 joint histogram bins
        Atomix.@atomic joint_hist[m_bin_lo, s_bin_lo] += m_weight_lo * s_weight_lo

        if m_bin_hi != m_bin_lo
            Atomix.@atomic joint_hist[m_bin_hi, s_bin_lo] += m_weight_hi * s_weight_lo
        end

        if s_bin_hi != s_bin_lo
            Atomix.@atomic joint_hist[m_bin_lo, s_bin_hi] += m_weight_lo * s_weight_hi
        end

        if m_bin_hi != m_bin_lo && s_bin_hi != s_bin_lo
            Atomix.@atomic joint_hist[m_bin_hi, s_bin_hi] += m_weight_hi * s_weight_hi
        end
    end

    return nothing
end

# ============================================================================
# Gaussian smoothing of histograms (optional post-processing)
# ============================================================================

# Apply 1D Gaussian smoothing to histogram using separable convolution
function _smooth_histogram_1d!(out::AbstractArray{T,1}, hist::AbstractArray{T,1}, sigma::T) where T
    if sigma < T(0.1)
        # No smoothing needed
        AK.foreachindex(out) do i
            @inbounds out[i] = hist[i]
        end
        return nothing
    end

    n = length(hist)
    kernel_radius = min(ceil(Int, T(3) * sigma), n - 1)

    AK.foreachindex(out) do i
        sum_val = zero(T)
        sum_weight = zero(T)

        # Unroll the kernel loop with fixed bounds
        for k in -kernel_radius:kernel_radius
            j = i + k
            if j >= 1 && j <= n
                dist = T(k)
                weight = exp(-dist * dist / (T(2) * sigma * sigma))
                sum_val += @inbounds hist[j] * weight
                sum_weight += weight
            end
        end

        @inbounds out[i] = sum_val / max(sum_weight, T(1e-10))
    end

    return nothing
end

# Apply 2D Gaussian smoothing to joint histogram
function _smooth_histogram_2d!(out::AbstractArray{T,2}, hist::AbstractArray{T,2}, sigma::T) where T
    if sigma < T(0.1)
        AK.foreachindex(out) do idx
            @inbounds out[idx] = hist[idx]
        end
        return nothing
    end

    m, n = size(hist)
    kernel_radius = min(ceil(Int, T(3) * sigma), min(m, n) - 1)

    AK.foreachindex(out) do idx
        # Convert linear index to 2D
        idx_0 = idx - 1
        i = idx_0 % m + 1
        j = idx_0 ÷ m + 1

        sum_val = zero(T)
        sum_weight = zero(T)

        for kj in -kernel_radius:kernel_radius
            jj = j + kj
            if jj < 1 || jj > n
                continue
            end
            for ki in -kernel_radius:kernel_radius
                ii = i + ki
                if ii < 1 || ii > m
                    continue
                end

                dist_sq = T(ki * ki + kj * kj)
                weight = exp(-dist_sq / (T(2) * sigma * sigma))
                sum_val += @inbounds hist[ii, jj] * weight
                sum_weight += weight
            end
        end

        @inbounds out[idx] = sum_val / max(sum_weight, T(1e-10))
    end

    return nothing
end

# ============================================================================
# Entropy Computation
# ============================================================================

# Compute entropy from histogram: H = -Σ p log p
function _compute_entropy(hist::AbstractArray{T}, total::T, eps::T) where T
    # Compute entropy element-wise
    entropy_arr = similar(hist)

    AK.foreachindex(entropy_arr) do idx
        count = @inbounds hist[idx]
        p = count / total
        # Handle p=0 case: 0*log(0) = 0
        if p > eps
            @inbounds entropy_arr[idx] = -p * log(p)
        else
            @inbounds entropy_arr[idx] = zero(T)
        end
    end

    return AK.reduce(+, entropy_arr; init=zero(T))
end

# ============================================================================
# MI Loss Implementation
# ============================================================================

function _mi_loss_impl(
    moving::AbstractArray{T},
    static::AbstractArray{T},
    num_bins::Int,
    sigma::T,
    intensity_range::Union{Nothing, Tuple{<:Real, <:Real}}
) where T
    # Determine intensity range
    if intensity_range === nothing
        m_min, m_max = _get_minmax(moving)
        s_min, s_max = _get_minmax(static)
        min_val = min(m_min, s_min)
        max_val = max(m_max, s_max)
    else
        min_val = T(intensity_range[1])
        max_val = T(intensity_range[2])
    end

    # Avoid division by zero
    range_val = max_val - min_val
    eps = T(1e-10)
    if range_val < eps
        range_val = one(T)
    end
    bin_width = range_val / T(num_bins - 1)

    # Allocate histograms
    joint_hist = similar(moving, num_bins, num_bins)
    moving_hist = similar(moving, num_bins)
    static_hist = similar(moving, num_bins)

    # Compute histograms
    _compute_marginal_histogram!(moving_hist, moving, min_val, bin_width, num_bins)
    _compute_marginal_histogram!(static_hist, static, min_val, bin_width, num_bins)
    _compute_joint_histogram!(joint_hist, moving, static, min_val, bin_width, num_bins)

    # Optional smoothing (only if sigma > 0.5)
    if sigma > T(0.5)
        moving_hist_smooth = similar(moving_hist)
        static_hist_smooth = similar(static_hist)
        joint_hist_smooth = similar(joint_hist)

        _smooth_histogram_1d!(moving_hist_smooth, moving_hist, sigma)
        _smooth_histogram_1d!(static_hist_smooth, static_hist, sigma)
        _smooth_histogram_2d!(joint_hist_smooth, joint_hist, sigma)

        moving_hist = moving_hist_smooth
        static_hist = static_hist_smooth
        joint_hist = joint_hist_smooth
    end

    # Total counts
    joint_total = AK.reduce(+, joint_hist; init=zero(T))
    moving_total = AK.reduce(+, moving_hist; init=zero(T))
    static_total = AK.reduce(+, static_hist; init=zero(T))

    # Compute entropies
    H_moving = _compute_entropy(moving_hist, moving_total, eps)
    H_static = _compute_entropy(static_hist, static_total, eps)
    H_joint = _compute_entropy(joint_hist, joint_total, eps)

    # MI = H(X) + H(Y) - H(X,Y)
    mi = H_moving + H_static - H_joint

    # Return negative MI (we want to minimize loss, maximize MI)
    result = similar(moving, 1)
    fill!(result, -mi)
    return result
end

function _nmi_loss_impl(
    moving::AbstractArray{T},
    static::AbstractArray{T},
    num_bins::Int,
    sigma::T,
    intensity_range::Union{Nothing, Tuple{<:Real, <:Real}}
) where T
    # Determine intensity range
    if intensity_range === nothing
        m_min, m_max = _get_minmax(moving)
        s_min, s_max = _get_minmax(static)
        min_val = min(m_min, s_min)
        max_val = max(m_max, s_max)
    else
        min_val = T(intensity_range[1])
        max_val = T(intensity_range[2])
    end

    # Avoid division by zero
    range_val = max_val - min_val
    eps = T(1e-10)
    if range_val < eps
        range_val = one(T)
    end
    bin_width = range_val / T(num_bins - 1)

    # Allocate histograms
    joint_hist = similar(moving, num_bins, num_bins)
    moving_hist = similar(moving, num_bins)
    static_hist = similar(moving, num_bins)

    # Compute histograms
    _compute_marginal_histogram!(moving_hist, moving, min_val, bin_width, num_bins)
    _compute_marginal_histogram!(static_hist, static, min_val, bin_width, num_bins)
    _compute_joint_histogram!(joint_hist, moving, static, min_val, bin_width, num_bins)

    # Optional smoothing
    if sigma > T(0.5)
        moving_hist_smooth = similar(moving_hist)
        static_hist_smooth = similar(static_hist)
        joint_hist_smooth = similar(joint_hist)

        _smooth_histogram_1d!(moving_hist_smooth, moving_hist, sigma)
        _smooth_histogram_1d!(static_hist_smooth, static_hist, sigma)
        _smooth_histogram_2d!(joint_hist_smooth, joint_hist, sigma)

        moving_hist = moving_hist_smooth
        static_hist = static_hist_smooth
        joint_hist = joint_hist_smooth
    end

    # Total counts
    joint_total = AK.reduce(+, joint_hist; init=zero(T))
    moving_total = AK.reduce(+, moving_hist; init=zero(T))
    static_total = AK.reduce(+, static_hist; init=zero(T))

    # Compute entropies
    H_moving = _compute_entropy(moving_hist, moving_total, eps)
    H_static = _compute_entropy(static_hist, static_total, eps)
    H_joint = _compute_entropy(joint_hist, joint_total, eps)

    # NMI = 2 * MI / (H(X) + H(Y))
    mi = H_moving + H_static - H_joint
    marginal_sum = H_moving + H_static

    # Avoid division by zero
    if marginal_sum < eps
        nmi = zero(T)
    else
        nmi = T(2) * mi / marginal_sum
    end

    # Return negative NMI
    result = similar(moving, 1)
    fill!(result, -nmi)
    return result
end

# ============================================================================
# Backward Pass - Gradient Computation
# ============================================================================

# The gradient of MI w.r.t. pixel values requires differentiating through:
# 1. Soft bin assignment
# 2. Histogram accumulation
# 3. Entropy computation

function _∇mi_loss!(
    d_moving::AbstractArray{T},
    d_static::AbstractArray{T},
    d_output_arr,
    moving::AbstractArray{T},
    static::AbstractArray{T},
    num_bins::Int,
    sigma::T,
    min_val::T,
    bin_width::T
) where T
    # Extract upstream gradient
    d_output = _extract_scalar(d_output_arr)

    eps = T(1e-10)

    # Recompute histograms
    # Note: We avoid reassigning variables to prevent Core.Box issues with GPU closures
    # For now, we skip smoothing in the backward pass gradient computation
    # (The soft binning already provides smoothness)

    moving_hist_final = similar(moving, num_bins)
    static_hist_final = similar(moving, num_bins)
    joint_hist_final = similar(moving, num_bins, num_bins)

    _compute_marginal_histogram!(moving_hist_final, moving, min_val, bin_width, num_bins)
    _compute_marginal_histogram!(static_hist_final, static, min_val, bin_width, num_bins)
    _compute_joint_histogram!(joint_hist_final, moving, static, min_val, bin_width, num_bins)

    # Get totals
    joint_total = AK.reduce(+, joint_hist_final; init=zero(T))
    moving_total = AK.reduce(+, moving_hist_final; init=zero(T))
    static_total = AK.reduce(+, static_hist_final; init=zero(T))

    # Precompute d(H)/d(hist_i) = -(log(p_i) + 1) / total for each histogram
    # These are the derivatives of entropy w.r.t. histogram bin counts
    d_H_d_moving_hist = similar(moving_hist_final)
    AK.foreachindex(d_H_d_moving_hist) do idx
        p = @inbounds moving_hist_final[idx] / moving_total
        if p > eps
            @inbounds d_H_d_moving_hist[idx] = -(log(p) + one(T)) / moving_total
        else
            @inbounds d_H_d_moving_hist[idx] = zero(T)
        end
    end

    d_H_d_static_hist = similar(static_hist_final)
    AK.foreachindex(d_H_d_static_hist) do idx
        p = @inbounds static_hist_final[idx] / static_total
        if p > eps
            @inbounds d_H_d_static_hist[idx] = -(log(p) + one(T)) / static_total
        else
            @inbounds d_H_d_static_hist[idx] = zero(T)
        end
    end

    d_H_d_joint_hist = similar(joint_hist_final)
    AK.foreachindex(d_H_d_joint_hist) do idx
        p = @inbounds joint_hist_final[idx] / joint_total
        if p > eps
            @inbounds d_H_d_joint_hist[idx] = -(log(p) + one(T)) / joint_total
        else
            @inbounds d_H_d_joint_hist[idx] = zero(T)
        end
    end

    # MI = H_moving + H_static - H_joint
    # d(MI)/d(pixel) = d(H_moving)/d(pixel) + d(H_static)/d(pixel) - d(H_joint)/d(pixel)
    # Loss = -MI, so d(Loss)/d(pixel) = -d(MI)/d(pixel)

    # Scale by upstream gradient
    scale = -d_output  # Negative because loss = -MI

    # Inverse bin width for gradient
    inv_bin_width = one(T) / bin_width

    # Compute gradient for each pixel
    AK.foreachindex(d_moving) do idx
        m_val = @inbounds moving[idx]
        s_val = @inbounds static[idx]

        # Soft bin assignment
        m_bin_lo, m_bin_hi, m_weight_lo, m_weight_hi = _soft_bin(m_val, min_val, bin_width, num_bins)
        s_bin_lo, s_bin_hi, s_weight_lo, s_weight_hi = _soft_bin(s_val, min_val, bin_width, num_bins)

        # Gradient of soft bin weights w.r.t. value
        # weight_lo = 1 - frac, weight_hi = frac
        # d(weight_lo)/d(val) = -1/bin_width, d(weight_hi)/d(val) = 1/bin_width

        d_m = zero(T)
        d_s = zero(T)

        # Moving marginal gradient
        # d(moving_hist[bin_lo])/d(m_val) = d(weight_lo)/d(m_val) = -inv_bin_width
        # d(moving_hist[bin_hi])/d(m_val) = d(weight_hi)/d(m_val) = inv_bin_width
        d_m += @inbounds d_H_d_moving_hist[m_bin_lo] * (-inv_bin_width)
        if m_bin_hi != m_bin_lo
            d_m += @inbounds d_H_d_moving_hist[m_bin_hi] * inv_bin_width
        end

        # Static marginal gradient
        d_s += @inbounds d_H_d_static_hist[s_bin_lo] * (-inv_bin_width)
        if s_bin_hi != s_bin_lo
            d_s += @inbounds d_H_d_static_hist[s_bin_hi] * inv_bin_width
        end

        # Joint histogram gradient
        # joint_hist[m_bin_lo, s_bin_lo] += m_weight_lo * s_weight_lo
        # d/d(m_val) = d(m_weight_lo)/d(m_val) * s_weight_lo = -inv_bin_width * s_weight_lo

        d_joint_m = zero(T)
        d_joint_s = zero(T)

        # d/d(m_val) for joint
        d_joint_m += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_lo] * (-inv_bin_width) * s_weight_lo
        if m_bin_hi != m_bin_lo
            d_joint_m += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_lo] * inv_bin_width * s_weight_lo
        end
        if s_bin_hi != s_bin_lo
            d_joint_m += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_hi] * (-inv_bin_width) * s_weight_hi
        end
        if m_bin_hi != m_bin_lo && s_bin_hi != s_bin_lo
            d_joint_m += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_hi] * inv_bin_width * s_weight_hi
        end

        # d/d(s_val) for joint
        d_joint_s += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_lo] * m_weight_lo * (-inv_bin_width)
        if s_bin_hi != s_bin_lo
            d_joint_s += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_hi] * m_weight_lo * inv_bin_width
        end
        if m_bin_hi != m_bin_lo
            d_joint_s += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_lo] * m_weight_hi * (-inv_bin_width)
        end
        if m_bin_hi != m_bin_lo && s_bin_hi != s_bin_lo
            d_joint_s += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_hi] * m_weight_hi * inv_bin_width
        end

        # Combine: d(MI)/d(moving) = d(H_moving)/d(moving) - d(H_joint)/d(moving)
        d_m -= d_joint_m
        d_s -= d_joint_s

        # Apply scale and accumulate
        @inbounds d_moving[idx] += scale * d_m
        @inbounds d_static[idx] += scale * d_s
    end

    return nothing
end

# Similar backward pass for NMI
function _∇nmi_loss!(
    d_moving::AbstractArray{T},
    d_static::AbstractArray{T},
    d_output_arr,
    moving::AbstractArray{T},
    static::AbstractArray{T},
    num_bins::Int,
    sigma::T,
    min_val::T,
    bin_width::T
) where T
    # Extract upstream gradient
    d_output = _extract_scalar(d_output_arr)

    eps = T(1e-10)

    # Recompute histograms (avoid variable reassignment for GPU compatibility)
    moving_hist_final = similar(moving, num_bins)
    static_hist_final = similar(moving, num_bins)
    joint_hist_final = similar(moving, num_bins, num_bins)

    _compute_marginal_histogram!(moving_hist_final, moving, min_val, bin_width, num_bins)
    _compute_marginal_histogram!(static_hist_final, static, min_val, bin_width, num_bins)
    _compute_joint_histogram!(joint_hist_final, moving, static, min_val, bin_width, num_bins)

    joint_total = AK.reduce(+, joint_hist_final; init=zero(T))
    moving_total = AK.reduce(+, moving_hist_final; init=zero(T))
    static_total = AK.reduce(+, static_hist_final; init=zero(T))

    H_moving = _compute_entropy(moving_hist_final, moving_total, eps)
    H_static = _compute_entropy(static_hist_final, static_total, eps)
    H_joint = _compute_entropy(joint_hist_final, joint_total, eps)

    mi = H_moving + H_static - H_joint
    marginal_sum = H_moving + H_static

    # NMI = 2 * MI / marginal_sum
    # d(NMI)/d(H_moving) = 2/marginal_sum - 2*MI/marginal_sum^2
    # d(NMI)/d(H_static) = same
    # d(NMI)/d(H_joint) = -2/marginal_sum

    if marginal_sum < eps
        # Early return - no gradient
        return nothing
    end

    inv_marg = one(T) / marginal_sum
    d_nmi_d_Hm = T(2) * inv_marg - T(2) * mi * inv_marg * inv_marg
    d_nmi_d_Hs = d_nmi_d_Hm
    d_nmi_d_Hj = -T(2) * inv_marg

    d_H_d_moving_hist = similar(moving_hist_final)
    AK.foreachindex(d_H_d_moving_hist) do idx
        p = @inbounds moving_hist_final[idx] / moving_total
        if p > eps
            @inbounds d_H_d_moving_hist[idx] = d_nmi_d_Hm * (-(log(p) + one(T)) / moving_total)
        else
            @inbounds d_H_d_moving_hist[idx] = zero(T)
        end
    end

    d_H_d_static_hist = similar(static_hist_final)
    AK.foreachindex(d_H_d_static_hist) do idx
        p = @inbounds static_hist_final[idx] / static_total
        if p > eps
            @inbounds d_H_d_static_hist[idx] = d_nmi_d_Hs * (-(log(p) + one(T)) / static_total)
        else
            @inbounds d_H_d_static_hist[idx] = zero(T)
        end
    end

    d_H_d_joint_hist = similar(joint_hist_final)
    AK.foreachindex(d_H_d_joint_hist) do idx
        p = @inbounds joint_hist_final[idx] / joint_total
        if p > eps
            @inbounds d_H_d_joint_hist[idx] = d_nmi_d_Hj * (-(log(p) + one(T)) / joint_total)
        else
            @inbounds d_H_d_joint_hist[idx] = zero(T)
        end
    end

    scale = -d_output  # Negative because loss = -NMI
    inv_bin_width = one(T) / bin_width

    AK.foreachindex(d_moving) do idx
        m_val = @inbounds moving[idx]
        s_val = @inbounds static[idx]

        m_bin_lo, m_bin_hi, m_weight_lo, m_weight_hi = _soft_bin(m_val, min_val, bin_width, num_bins)
        s_bin_lo, s_bin_hi, s_weight_lo, s_weight_hi = _soft_bin(s_val, min_val, bin_width, num_bins)

        d_m = zero(T)
        d_s = zero(T)

        # Moving marginal
        d_m += @inbounds d_H_d_moving_hist[m_bin_lo] * (-inv_bin_width)
        if m_bin_hi != m_bin_lo
            d_m += @inbounds d_H_d_moving_hist[m_bin_hi] * inv_bin_width
        end

        # Static marginal
        d_s += @inbounds d_H_d_static_hist[s_bin_lo] * (-inv_bin_width)
        if s_bin_hi != s_bin_lo
            d_s += @inbounds d_H_d_static_hist[s_bin_hi] * inv_bin_width
        end

        # Joint
        d_joint_m = zero(T)
        d_joint_s = zero(T)

        d_joint_m += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_lo] * (-inv_bin_width) * s_weight_lo
        if m_bin_hi != m_bin_lo
            d_joint_m += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_lo] * inv_bin_width * s_weight_lo
        end
        if s_bin_hi != s_bin_lo
            d_joint_m += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_hi] * (-inv_bin_width) * s_weight_hi
        end
        if m_bin_hi != m_bin_lo && s_bin_hi != s_bin_lo
            d_joint_m += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_hi] * inv_bin_width * s_weight_hi
        end

        d_joint_s += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_lo] * m_weight_lo * (-inv_bin_width)
        if s_bin_hi != s_bin_lo
            d_joint_s += @inbounds d_H_d_joint_hist[m_bin_lo, s_bin_hi] * m_weight_lo * inv_bin_width
        end
        if m_bin_hi != m_bin_lo
            d_joint_s += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_lo] * m_weight_hi * (-inv_bin_width)
        end
        if m_bin_hi != m_bin_lo && s_bin_hi != s_bin_lo
            d_joint_s += @inbounds d_H_d_joint_hist[m_bin_hi, s_bin_hi] * m_weight_hi * inv_bin_width
        end

        d_m += d_joint_m
        d_s += d_joint_s

        @inbounds d_moving[idx] += scale * d_m
        @inbounds d_static[idx] += scale * d_s
    end

    return nothing
end

# Note: _extract_scalar is defined in metrics.jl and reused here

# ============================================================================
# Mooncake rrule!! definitions
# ============================================================================

@is_primitive MinimalCtx Tuple{typeof(mi_loss), AbstractArray, AbstractArray}

function Mooncake.rrule!!(
    ::CoDual{typeof(mi_loss)},
    moving::CoDual{M, FM},
    static::CoDual{S, FS};
    num_bins::Int=64,
    sigma::Real=1.0,
    intensity_range::Union{Nothing, Tuple{<:Real, <:Real}}=nothing
) where {M<:AbstractArray, FM, S<:AbstractArray, FS}
    moving_primal = moving.x
    static_primal = static.x
    moving_fdata = moving.dx
    static_fdata = static.dx
    T = eltype(moving_primal)

    sigma_T = T(sigma)

    # Determine intensity range
    if intensity_range === nothing
        m_min, m_max = _get_minmax(moving_primal)
        s_min, s_max = _get_minmax(static_primal)
        min_val = min(m_min, s_min)
        max_val = max(m_max, s_max)
    else
        min_val = T(intensity_range[1])
        max_val = T(intensity_range[2])
    end

    range_val = max_val - min_val
    eps = T(1e-10)
    if range_val < eps
        range_val = one(T)
    end
    bin_width = range_val / T(num_bins - 1)

    # Forward pass
    output = _mi_loss_impl(moving_primal, static_primal, num_bins, sigma_T, intensity_range)
    output_fdata = similar(output)
    fill!(output_fdata, zero(T))

    function mi_loss_pullback(_rdata)
        _∇mi_loss!(moving_fdata, static_fdata, output_fdata,
                   moving_primal, static_primal, num_bins, sigma_T, min_val, bin_width)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), mi_loss_pullback
end

@is_primitive MinimalCtx Tuple{typeof(nmi_loss), AbstractArray, AbstractArray}

function Mooncake.rrule!!(
    ::CoDual{typeof(nmi_loss)},
    moving::CoDual{M, FM},
    static::CoDual{S, FS};
    num_bins::Int=64,
    sigma::Real=1.0,
    intensity_range::Union{Nothing, Tuple{<:Real, <:Real}}=nothing
) where {M<:AbstractArray, FM, S<:AbstractArray, FS}
    moving_primal = moving.x
    static_primal = static.x
    moving_fdata = moving.dx
    static_fdata = static.dx
    T = eltype(moving_primal)

    sigma_T = T(sigma)

    if intensity_range === nothing
        m_min, m_max = _get_minmax(moving_primal)
        s_min, s_max = _get_minmax(static_primal)
        min_val = min(m_min, s_min)
        max_val = max(m_max, s_max)
    else
        min_val = T(intensity_range[1])
        max_val = T(intensity_range[2])
    end

    range_val = max_val - min_val
    eps = T(1e-10)
    if range_val < eps
        range_val = one(T)
    end
    bin_width = range_val / T(num_bins - 1)

    output = _nmi_loss_impl(moving_primal, static_primal, num_bins, sigma_T, intensity_range)
    output_fdata = similar(output)
    fill!(output_fdata, zero(T))

    function nmi_loss_pullback(_rdata)
        _∇nmi_loss!(moving_fdata, static_fdata, output_fdata,
                    moving_primal, static_primal, num_bins, sigma_T, min_val, bin_width)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), nmi_loss_pullback
end
