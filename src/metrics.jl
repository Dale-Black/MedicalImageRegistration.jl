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
displacement field. The naming convention matches torchreg where:
- u_xx, u_yy, u_zz are the second derivatives (∂²u_i/∂x_i²)
- u_xy, u_xz, etc. are mixed second derivatives

The stress tensor is then computed and its Frobenius norm is returned.
"""
function (reg::LinearElasticity{T})(u::AbstractArray{T, 5}) where T
    X, Y, Z, C, N = size(u)
    @assert C == 3 "Expected 3 displacement components, got $C"

    # Create or retrieve identity grid
    if reg.id_grid === nothing || reg.refresh_id_grid
        reg.id_grid = create_identity_grid((X, Y, Z), T)
    end
    id_grid = reg.id_grid

    # First-order gradients: (3, 3, X, Y, Z, N)
    # gradients[component, deriv_direction, spatial..., batch]
    gradients = jacobi_gradient(u, id_grid)

    # Following torchreg: compute second-order derivatives
    # For each displacement component, take gradient of its gradient
    #
    # torchreg does: jacobi_gradient(gradients[None, 2], id_grid) for x-component
    # where gradients[None, 2] extracts the x-derivative (index 2 in torchreg = x)
    #
    # In our convention:
    # gradients[c, d, ...] = ∂u_c / ∂x_d where c,d ∈ {1=x, 2=y, 3=z}
    #
    # We need to take the gradient of each first derivative to get second derivatives.
    # Following torchreg indexing (0=z, 1=y, 2=x in their convention):

    # Extract first derivatives for x-component (u_x)
    # ∂u_x/∂x, ∂u_x/∂y, ∂u_x/∂z -> these are gradients[1, 1:3, ...]
    # Then take gradient of each to get second derivatives

    # Create arrays for second derivatives
    # For torchreg parity, we match their variable naming:
    # u_xz, u_xy, u_xx = second derivatives of u_x w.r.t. z, y, x
    # u_yz, u_yy, u_yx = second derivatives of u_y w.r.t. z, y, x
    # u_zz, u_zy, u_zx = second derivatives of u_z w.r.t. z, y, x

    # Package first derivatives as displacement-like fields for jacobi_gradient
    # Take x-component first derivatives: shape (X, Y, Z, 3, N) where 3 = (∂u_x/∂x, ∂u_x/∂y, ∂u_x/∂z)
    grad_ux = permutedims(gradients[1, :, :, :, :, :], (2, 3, 4, 1, 5))  # (X, Y, Z, 3, N)
    grad_uy = permutedims(gradients[2, :, :, :, :, :], (2, 3, 4, 1, 5))  # (X, Y, Z, 3, N)
    grad_uz = permutedims(gradients[3, :, :, :, :, :], (2, 3, 4, 1, 5))  # (X, Y, Z, 3, N)

    # Compute second derivatives
    # grad2_ux[c, d, ...] = ∂/∂x_d (∂u_x/∂x_c) = ∂²u_x/(∂x_c ∂x_d)
    grad2_ux = jacobi_gradient(grad_ux, id_grid)  # (3, 3, X, Y, Z, N)
    grad2_uy = jacobi_gradient(grad_uy, id_grid)  # (3, 3, X, Y, Z, N)
    grad2_uz = jacobi_gradient(grad_uz, id_grid)  # (3, 3, X, Y, Z, N)

    # Extract second derivatives following torchreg naming convention
    # In torchreg: u_xz, u_xy, u_xx = jacobi_gradient(gradients[None, 2], ...)
    # Their index 2 = x, 1 = y, 0 = z (reverse order)
    #
    # Our grad2_ux[c, d, ...] = ∂²u_x/(∂x_c ∂x_d)
    # For torchreg's u_xx (= ∂²u_x/∂x²), we need grad2_ux[1, 1, ...] (derivative of ∂u_x/∂x w.r.t. x)
    # But jacobi_gradient of grad_ux where grad_ux[:,:,:,1,:] = ∂u_x/∂x, etc.

    # Actually, let me reconsider. grad_ux has shape (X, Y, Z, 3, N) where:
    # grad_ux[:,:,:,1,:] = ∂u_x/∂x
    # grad_ux[:,:,:,2,:] = ∂u_x/∂y
    # grad_ux[:,:,:,3,:] = ∂u_x/∂z
    #
    # Then jacobi_gradient(grad_ux) computes gradients of these as if they were displacement components.
    # Output grad2_ux[c, d, ...] = ∂(grad_ux[:,:,:,c,:])/∂x_d
    # = ∂(∂u_x/∂x_c)/∂x_d = ∂²u_x/(∂x_c ∂x_d)

    # So for u_xx = ∂²u_x/∂x² = grad2_ux[1, 1, ...]
    u_xx = grad2_ux[1, 1, :, :, :, :]
    u_xy = grad2_ux[1, 2, :, :, :, :]  # or grad2_ux[2, 1, ...] by symmetry
    u_xz = grad2_ux[1, 3, :, :, :, :]

    u_yx = grad2_uy[2, 1, :, :, :, :]
    u_yy = grad2_uy[2, 2, :, :, :, :]
    u_yz = grad2_uy[2, 3, :, :, :, :]

    u_zx = grad2_uz[3, 1, :, :, :, :]
    u_zy = grad2_uz[3, 2, :, :, :, :]
    u_zz = grad2_uz[3, 3, :, :, :, :]

    # Symmetric shear components (average of mixed partials)
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
