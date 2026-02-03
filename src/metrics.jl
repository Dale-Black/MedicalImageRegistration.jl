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

"""
    LinearElasticity(; mu=2.0, lam=1.0, refresh_id_grid=false)

Linear elasticity regularizer for displacement fields.

Penalizes non-smooth deformation fields using second-order spatial derivatives,
following the torchreg implementation. Useful for regularizing SyN registration.

# Arguments
- `mu`: Shear modulus - resistance to shearing deformation (default: 2.0)
- `lam`: First Lamé parameter - resistance to compression (default: 1.0)
- `refresh_id_grid`: If true, recreate identity grid on each call (default: false)

# Physical Interpretation
- Higher `mu` penalizes shearing more strongly
- Higher `lam` penalizes volume changes more strongly
- The regularizer encourages smooth, physically plausible deformations

# Example
```julia
reg = LinearElasticity(mu=2.0, lam=1.0)
u = randn(Float32, 32, 32, 32, 3, 1)  # Displacement field
penalty = reg(u)
```
"""
mutable struct LinearElasticity{T}
    mu::T
    lam::T
    refresh_id_grid::Bool
    id_grid::Union{Nothing, Array}  # Cached identity grid
end

function LinearElasticity(; mu::T=2.0f0, lam::T=1.0f0, refresh_id_grid::Bool=false) where T
    return LinearElasticity{T}(mu, lam, refresh_id_grid, nothing)
end

"""
    (reg::LinearElasticity)(u::AbstractArray{T, 5}) where T

Compute linear elasticity penalty for a displacement field.

# Arguments
- `u`: Displacement field of shape `(X, Y, Z, 3, N)` in Julia convention.
       The 4th dimension contains (u_x, u_y, u_z) displacement components.

# Returns
Scalar penalty value (mean squared Frobenius norm of stress tensor).

# Implementation Notes
Following torchreg, this computes **second-order spatial derivatives** of the
displacement field using a specific approach:

1. First compute gradients of u: grad[deriv_dir, spatial..., component]
2. For each derivative direction, compute gradients again to get second derivatives
3. Combine into stress tensor and compute mean Frobenius norm

The result matches torchreg's LinearElasticity regularizer.
"""
function (reg::LinearElasticity{T})(u::AbstractArray{T, 5}) where T
    X, Y, Z, C, N = size(u)
    @assert C == 3 "Expected 3 displacement components, got $C"

    # Create or retrieve identity grid
    if reg.id_grid === nothing || reg.refresh_id_grid
        reg.id_grid = create_identity_grid((X, Y, Z), T)
    end
    id_grid = reg.id_grid

    # First-order gradients using torchreg-style jacobi_gradient
    # This computes gradients in a specific way that we need to match
    gradients = _jacobi_gradient_torchreg_style(u, id_grid)
    # Shape: (3, X, Y, Z, 3) where first 3 is deriv direction (z,y,x order like torchreg)
    # and last 3 is displacement component

    # Second-order derivatives following torchreg exactly:
    # u_xz, u_xy, u_xx = jacobi_gradient(gradients[None, 2], id_grid)  # x-derivatives
    # u_yz, u_yy, u_yx = jacobi_gradient(gradients[None, 1], id_grid)  # y-derivatives
    # u_zz, u_zy, u_zx = jacobi_gradient(gradients[None, 0], id_grid)  # z-derivatives

    # In Julia, gradients has shape (3, X, Y, Z, 3)
    # We need to extract each derivative direction and compute second derivatives

    # Extract x-derivatives (index 3 in Julia = index 2 in torchreg's 0-indexing)
    # gradients[3, :, :, :, :] has shape (X, Y, Z, 3)
    grad_x = reshape(gradients[3, :, :, :, :], 1, X, Y, Z, 3)  # (1, X, Y, Z, 3)
    grad_y = reshape(gradients[2, :, :, :, :], 1, X, Y, Z, 3)  # (1, X, Y, Z, 3)
    grad_z = reshape(gradients[1, :, :, :, :], 1, X, Y, Z, 3)  # (1, X, Y, Z, 3)

    # Permute to (X, Y, Z, 3, 1) for our jacobi_gradient
    grad_x_perm = permutedims(grad_x, (2, 3, 4, 5, 1))  # (X, Y, Z, 3, 1)
    grad_y_perm = permutedims(grad_y, (2, 3, 4, 5, 1))  # (X, Y, Z, 3, 1)
    grad_z_perm = permutedims(grad_z, (2, 3, 4, 5, 1))  # (X, Y, Z, 3, 1)

    # Compute second derivatives
    grad2_x = _jacobi_gradient_torchreg_style(grad_x_perm, id_grid)  # (3, X, Y, Z, 3)
    grad2_y = _jacobi_gradient_torchreg_style(grad_y_perm, id_grid)  # (3, X, Y, Z, 3)
    grad2_z = _jacobi_gradient_torchreg_style(grad_z_perm, id_grid)  # (3, X, Y, Z, 3)

    # Extract second derivatives
    # grad2_x[d, ...] = ∂/∂x_d of (∂u/∂x) for all components
    # torchreg indexing: 0=z, 1=y, 2=x, so Julia: 1=z, 2=y, 3=x
    u_xz = grad2_x[1, :, :, :, :]  # ∂²/∂z∂x for all components
    u_xy = grad2_x[2, :, :, :, :]  # ∂²/∂y∂x for all components
    u_xx = grad2_x[3, :, :, :, :]  # ∂²/∂x∂x for all components

    u_yz = grad2_y[1, :, :, :, :]
    u_yy = grad2_y[2, :, :, :, :]
    u_yx = grad2_y[3, :, :, :, :]

    u_zz = grad2_z[1, :, :, :, :]
    u_zy = grad2_z[2, :, :, :, :]
    u_zx = grad2_z[3, :, :, :, :]

    # Symmetric shear components
    e_xy = T(0.5) .* (u_xy .+ u_yx)
    e_xz = T(0.5) .* (u_xz .+ u_zx)
    e_yz = T(0.5) .* (u_yz .+ u_zy)

    # Trace term
    trace = u_xx .+ u_yy .+ u_zz

    # Cauchy stress tensor components
    sigma_xx = T(2) .* reg.mu .* u_xx .+ reg.lam .* trace
    sigma_yy = T(2) .* reg.mu .* u_yy .+ reg.lam .* trace
    sigma_zz = T(2) .* reg.mu .* u_zz .+ reg.lam .* trace
    sigma_xy = T(2) .* reg.mu .* e_xy
    sigma_xz = T(2) .* reg.mu .* e_xz
    sigma_yz = T(2) .* reg.mu .* e_yz

    # Frobenius norm squared of stress tensor
    frobenius_sq = sigma_xx .^ 2 .+ sigma_yy .^ 2 .+ sigma_zz .^ 2 .+
                   sigma_xy .^ 2 .+ sigma_xz .^ 2 .+ sigma_yz .^ 2

    return mean(frobenius_sq)
end

"""
    _jacobi_gradient_torchreg_style(u, id_grid)

Compute jacobi gradient in torchreg's style.

Input u: (X, Y, Z, 3, N) in Julia convention
Output: (3, X, Y, Z, 3) where first 3 is deriv direction (z, y, x order) and last 3 is component.
        Note: batch dimension is collapsed since torchreg doesn't preserve it in gradient output.
"""
function _jacobi_gradient_torchreg_style(u::AbstractArray{T, 5}, id_grid) where T
    X, Y, Z, C, N = size(u)
    @assert C == 3 "Expected 3 components"
    @assert N == 1 "torchreg-style gradient only supports batch size 1"

    # Scale displacement from normalized [-1,1] to voxel coordinates
    # Following torchreg: x = 0.5 * (u + id_grid) * (shape - 1)
    scale_z = T(Z - 1)
    scale_y = T(Y - 1)
    scale_x = T(X - 1)

    # Output: (3_deriv_dirs, X, Y, Z, 3_components)
    # Derivative direction order: (z, y, x) to match torchreg's (0, 1, 2)
    gradients = zeros(T, 3, X, Y, Z, 3)

    @inbounds for c in 1:3  # Component
        # Scale factor for this component
        scale = c == 1 ? scale_x : (c == 2 ? scale_y : scale_z)

        for k in 1:Z, j in 1:Y, i in 1:X
            # Current scaled value: 0.5 * (u + grid) * (size - 1)
            # id_grid[c, i, j, k] gives normalized coord for component c
            x_curr = T(0.5) * (u[i, j, k, c, 1] + id_grid[c, i, j, k]) * scale

            # z-derivative (deriv index 1 = z direction, stored in output[1, ...])
            if k == 1
                x_next = T(0.5) * (u[i, j, 2, c, 1] + id_grid[c, i, j, 2]) * scale
                gradients[1, i, j, k, c] = x_next - x_curr
            elseif k == Z
                x_prev = T(0.5) * (u[i, j, Z-1, c, 1] + id_grid[c, i, j, Z-1]) * scale
                gradients[1, i, j, k, c] = x_curr - x_prev
            else
                x_prev = T(0.5) * (u[i, j, k-1, c, 1] + id_grid[c, i, j, k-1]) * scale
                x_next = T(0.5) * (u[i, j, k+1, c, 1] + id_grid[c, i, j, k+1]) * scale
                gradients[1, i, j, k, c] = T(0.5) * (x_next - x_prev)
            end

            # y-derivative (deriv index 2 = y direction)
            if j == 1
                x_next = T(0.5) * (u[i, 2, k, c, 1] + id_grid[c, i, 2, k]) * scale
                gradients[2, i, j, k, c] = x_next - x_curr
            elseif j == Y
                x_prev = T(0.5) * (u[i, Y-1, k, c, 1] + id_grid[c, i, Y-1, k]) * scale
                gradients[2, i, j, k, c] = x_curr - x_prev
            else
                x_prev = T(0.5) * (u[i, j-1, k, c, 1] + id_grid[c, i, j-1, k]) * scale
                x_next = T(0.5) * (u[i, j+1, k, c, 1] + id_grid[c, i, j+1, k]) * scale
                gradients[2, i, j, k, c] = T(0.5) * (x_next - x_prev)
            end

            # x-derivative (deriv index 3 = x direction)
            if i == 1
                x_next = T(0.5) * (u[2, j, k, c, 1] + id_grid[c, 2, j, k]) * scale
                gradients[3, i, j, k, c] = x_next - x_curr
            elseif i == X
                x_prev = T(0.5) * (u[X-1, j, k, c, 1] + id_grid[c, X-1, j, k]) * scale
                gradients[3, i, j, k, c] = x_curr - x_prev
            else
                x_prev = T(0.5) * (u[i-1, j, k, c, 1] + id_grid[c, i-1, j, k]) * scale
                x_next = T(0.5) * (u[i+1, j, k, c, 1] + id_grid[c, i+1, j, k]) * scale
                gradients[3, i, j, k, c] = T(0.5) * (x_next - x_prev)
            end
        end
    end

    return gradients
end
