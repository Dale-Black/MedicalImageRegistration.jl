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
