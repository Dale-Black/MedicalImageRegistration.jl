# Loss functions and metrics for registration

"""
    dice_score(x1::AbstractArray, x2::AbstractArray)

Compute the Dice coefficient (Sørensen–Dice index) between two arrays.

The Dice coefficient measures overlap between two segmentations:
    Dice = 2|A∩B| / (|A| + |B|)

For soft segmentations (values in [0,1]):
- Intersection: sum(x1 .* x2)
- Union approximation: sum(x1) + sum(x2)

# Arguments
- `x1`: First array, shape `(X, Y, [Z,] C, N)` for 2D/3D
- `x2`: Second array, same shape as x1

# Returns
Scalar dice score averaged over batch, value in [0, 1] where 1 is perfect overlap.

# Example
```julia
x1 = rand(Float32, 32, 32, 1, 2)  # 2D batch of 2
x2 = rand(Float32, 32, 32, 1, 2)
score = dice_score(x1, x2)
```
"""
function dice_score(x1::AbstractArray{T, N}, x2::AbstractArray{T, N}) where {T, N}
    # Determine spatial dimensions to sum over
    # 2D: (X, Y, C, N) -> sum over dims 1:2
    # 3D: (X, Y, Z, C, N) -> sum over dims 1:3
    spatial_dims = N == 4 ? (1, 2) : (1, 2, 3)

    # Compute intersection and union
    inter = sum(x1 .* x2; dims=spatial_dims)
    union_sum = sum(x1 .+ x2; dims=spatial_dims)

    # Dice coefficient: 2 * intersection / (sum_x1 + sum_x2)
    dice = T(2) .* inter ./ union_sum

    # Average over batch (and channel) dimensions
    return mean(dice)
end

"""
    dice_loss(x1::AbstractArray, x2::AbstractArray)

Compute the Dice loss between two arrays.

Dice loss = 1 - dice_score, suitable for minimization during training.
Returns 0 when arrays are identical, 1 when completely non-overlapping.

# Arguments
- `x1`: First array, shape `(X, Y, [Z,] C, N)` for 2D/3D
- `x2`: Second array, same shape as x1

# Returns
Scalar dice loss value in [0, 1].

# Example
```julia
x1 = rand(Float32, 32, 32, 1, 2)
x2 = rand(Float32, 32, 32, 1, 2)
loss = dice_loss(x1, x2)
```
"""
function dice_loss(x1::AbstractArray{T, N}, x2::AbstractArray{T, N}) where {T, N}
    return one(T) - dice_score(x1, x2)
end

"""
    NCC(; kernel_size=7, epsilon_numerator=1e-5, epsilon_denominator=1e-5)

Normalized Cross-Correlation loss function for image registration.

Computes the local windowed cross-correlation between two images using a box kernel.
Returns negative NCC (for minimization during training).

# Arguments
- `kernel_size`: Size of the local window for computing correlation (default: 7)
- `epsilon_numerator`: Small constant added to numerator for numerical stability (default: 1e-5)
- `epsilon_denominator`: Small constant added to denominator for numerical stability (default: 1e-5)

# Example
```julia
ncc = NCC(kernel_size=9)
pred = rand(Float32, 32, 32, 32, 1, 2)  # 3D batch of 2
targ = rand(Float32, 32, 32, 32, 1, 2)
loss = ncc(pred, targ)
```
"""
struct NCC{T}
    kernel_size::Int
    eps_nr::T
    eps_dr::T
end

function NCC(; kernel_size::Int=7, epsilon_numerator::T=Float32(1e-5), epsilon_denominator::T=Float32(1e-5)) where T
    return NCC{T}(kernel_size, epsilon_numerator, epsilon_denominator)
end

"""
    (ncc::NCC)(pred::AbstractArray, targ::AbstractArray)

Compute NCC loss between prediction and target arrays.

Uses local windowed cross-correlation computed via convolution with a box kernel.

# Arguments
- `pred`: Predicted image, shape `(X, Y, [Z,] C, N)`
- `targ`: Target image, same shape as pred

# Returns
Negative mean NCC (scalar). More negative = better correlation.
Perfect correlation would approach -1.

# Implementation Details
- Creates a box kernel (all ones) of size `kernel_size`
- Computes local sums via convolution
- Uses variance formula: Var(X) = E[X²] - E[X]²
- Applies ReLU to variances for numerical stability
"""
function (ncc::NCC{T})(pred::AbstractArray{T, N}, targ::AbstractArray{T, N}) where {T, N}
    # Determine if 2D or 3D
    is_3d = N == 5
    ks = ncc.kernel_size
    pad = ks ÷ 2

    # Get dimensions
    C = size(targ, N - 1)  # Channel dimension
    batch_size = size(targ, N)  # Batch dimension

    # Create box kernel: all ones
    # 2D: (ks, ks, 1, 1) - single channel in, single channel out
    # 3D: (ks, ks, ks, 1, 1)
    if is_3d
        kernel = ones(T, ks, ks, ks, 1, 1)
    else
        kernel = ones(T, ks, ks, 1, 1)
    end
    n_elements = T(prod(size(kernel)[1:end-2]))  # Number of elements in spatial kernel

    # Compute local sums via convolution
    # Process each channel separately and sum
    if is_3d
        padding = (pad, pad, pad)
    else
        padding = (pad, pad)
    end

    # Compute local statistics for all channels
    t_sum = NNlib.conv(targ, kernel; pad=padding)
    p_sum = NNlib.conv(pred, kernel; pad=padding)
    t2_sum = NNlib.conv(targ .^ 2, kernel; pad=padding)
    p2_sum = NNlib.conv(pred .^ 2, kernel; pad=padding)
    tp_sum = NNlib.conv(targ .* pred, kernel; pad=padding)

    # Cross-covariance: E[TP] - E[T]E[P]
    cross = tp_sum .- t_sum .* p_sum ./ n_elements

    # Variances: E[X²] - E[X]² (use max with 0 for numerical stability)
    t_var = max.(t2_sum .- t_sum .^ 2 ./ n_elements, zero(T))
    p_var = max.(p2_sum .- p_sum .^ 2 ./ n_elements, zero(T))

    # NCC: cross² / (var_t * var_p)
    cc = (cross .^ 2 .+ ncc.eps_nr) ./ (t_var .* p_var .+ ncc.eps_dr)

    # Return negative mean (for minimization)
    return -mean(cc)
end
