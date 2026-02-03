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
Matches torchreg's LinearElasticity exactly by:
1. Converting Julia array to torchreg format (permuting axes)
2. Computing jacobi_gradient using central differences with convolution
3. Computing second-order derivatives
4. Combining into stress tensor and returning mean Frobenius norm

Note: Only supports batch size N=1 for torchreg compatibility.
"""
function (reg::LinearElasticity{T})(u::AbstractArray{T, 5}) where T
    X, Y, Z, C, N = size(u)
    @assert C == 3 "Expected 3 displacement components, got $C"
    @assert N == 1 "LinearElasticity currently only supports batch size 1"

    # Convert from Julia (X, Y, Z, 3, N) to torchreg (N, Z, Y, X, 3) format
    u_torch = permutedims(u, (5, 3, 2, 1, 4))  # (N, Z, Y, X, 3) = (1, Z, Y, X, 3)

    # Create identity grid in torchreg format (1, Z, Y, X, 3)
    if reg.id_grid === nothing || reg.refresh_id_grid
        reg.id_grid = _create_id_grid_torchreg(Z, Y, X, T)
    end
    id_grid = reg.id_grid

    # First-order gradients
    # Output: (3, Z, Y, X, 3) where first 3 = component, last 3 = deriv direction
    gradients = _jacobi_gradient_torchreg(u_torch, id_grid)

    # Second-order derivatives following torchreg:
    # u_xz, u_xy, u_xx = jacobi_gradient(gradients[None, 2], id_grid)
    # u_yz, u_yy, u_yx = jacobi_gradient(gradients[None, 1], id_grid)
    # u_zz, u_zy, u_zx = jacobi_gradient(gradients[None, 0], id_grid)
    #
    # gradients[None, c] extracts component c and adds batch dim: (1, Z, Y, X, 3)

    # Extract each component and compute second derivatives
    # Component indexing: 0=u_x, 1=u_y, 2=u_z (Julia 1-indexed: 1,2,3)
    grad_comp3 = reshape(gradients[3, :, :, :, :], 1, Z, Y, X, 3)  # u_z derivatives
    grad_comp2 = reshape(gradients[2, :, :, :, :], 1, Z, Y, X, 3)  # u_y derivatives
    grad_comp1 = reshape(gradients[1, :, :, :, :], 1, Z, Y, X, 3)  # u_x derivatives

    grad2_comp3 = _jacobi_gradient_torchreg(grad_comp3, id_grid)  # (3, Z, Y, X, 3)
    grad2_comp2 = _jacobi_gradient_torchreg(grad_comp2, id_grid)
    grad2_comp1 = _jacobi_gradient_torchreg(grad_comp1, id_grid)

    # torchreg unpacking: result[0]=z-deriv, result[1]=y-deriv, result[2]=x-deriv (0-indexed)
    # Julia: result[1]=first component (which was z), etc.
    # But actually the unpacking u_xz, u_xy, u_xx = jacobi_gradient(...) unpacks along first dim
    # So output[0] -> u_xz (confusing naming, but matches torchreg structure)

    # From grad2_comp3 (gradients of u_z):
    u_xz = grad2_comp3[1, :, :, :, :]  # (Z, Y, X, 3)
    u_xy = grad2_comp3[2, :, :, :, :]
    u_xx = grad2_comp3[3, :, :, :, :]

    # From grad2_comp2 (gradients of u_y):
    u_yz = grad2_comp2[1, :, :, :, :]
    u_yy = grad2_comp2[2, :, :, :, :]
    u_yx = grad2_comp2[3, :, :, :, :]

    # From grad2_comp1 (gradients of u_x):
    u_zz = grad2_comp1[1, :, :, :, :]
    u_zy = grad2_comp1[2, :, :, :, :]
    u_zx = grad2_comp1[3, :, :, :, :]

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
Create identity grid in torchreg format: (1, Z, Y, X, 3)
Coordinates are normalized to [-1, 1].
Last dimension contains (x, y, z) coordinates.
"""
function _create_id_grid_torchreg(Z::Int, Y::Int, X::Int, ::Type{T}) where T
    grid = zeros(T, 1, Z, Y, X, 3)

    xs = collect(T, range(T(-1), T(1), length=X))
    ys = collect(T, range(T(-1), T(1), length=Y))
    zs = collect(T, range(T(-1), T(1), length=Z))

    @inbounds for k in 1:Z, j in 1:Y, i in 1:X
        grid[1, k, j, i, 1] = xs[i]  # x-coordinate
        grid[1, k, j, i, 2] = ys[j]  # y-coordinate
        grid[1, k, j, i, 3] = zs[k]  # z-coordinate
    end

    return grid
end

"""
Compute jacobi gradient matching torchreg's implementation.

Input: u with shape (N, Z, Y, X, 3) where N=1
Output: (3, Z, Y, X, 3) where first 3 = component treated as batch, last 3 = deriv direction

Uses central differences with [-0.5, 0, 0.5] kernel and boundary replication.
"""
function _jacobi_gradient_torchreg(u::AbstractArray{T, 5}, id_grid::AbstractArray{T, 5}) where T
    N, Z, Y, X, C = size(u)
    @assert N == 1 "Only batch size 1 supported"
    @assert C == 3 "Expected 3 components"

    # Scale: x = 0.5 * (u + id_grid) * (shape - 1)
    # shape = (Z, Y, X) in torchreg convention
    scale_factors = T.([Z - 1, Y - 1, X - 1])

    # Scaled coordinates
    scaled = zeros(T, N, Z, Y, X, C)
    @inbounds for c in 1:3
        scale = scale_factors[c == 1 ? 3 : (c == 2 ? 2 : 1)]  # x->X, y->Y, z->Z
        for i in 1:X, j in 1:Y, k in 1:Z
            scaled[1, k, j, i, c] = T(0.5) * (u[1, k, j, i, c] + id_grid[1, k, j, i, c]) * scale
        end
    end

    # Output: (3_components, Z, Y, X, 3_deriv_dirs)
    # After torchreg's permutation scheme, the output has:
    # - First dim = original component (treated as batch during conv)
    # - Last dim = derivative direction (conv output channels)
    gradients = zeros(T, 3, Z, Y, X, 3)

    # Central difference: derivative in each direction
    # torchreg kernel setup:
    # w[2,0,:,1,1] = window -> D derivative (Z direction) -> output channel 2
    # w[1,0,1,:,1] = window -> H derivative (Y direction) -> output channel 1
    # w[0,0,1,1,:] = window -> W derivative (X direction) -> output channel 0
    #
    # So output channels: 0=X-deriv, 1=Y-deriv, 2=Z-deriv

    @inbounds for c in 1:3  # Component (becomes batch in torchreg)
        for i in 1:X, j in 1:Y, k in 1:Z
            # X derivative (output channel 0, stored at deriv index 1 in Julia)
            if i == 1
                dx = scaled[1, k, j, 2, c] - scaled[1, k, j, 1, c]
            elseif i == X
                dx = scaled[1, k, j, X, c] - scaled[1, k, j, X-1, c]
            else
                dx = T(0.5) * (scaled[1, k, j, i+1, c] - scaled[1, k, j, i-1, c])
            end

            # Y derivative (output channel 1, stored at deriv index 2)
            if j == 1
                dy = scaled[1, k, 2, i, c] - scaled[1, k, 1, i, c]
            elseif j == Y
                dy = scaled[1, k, Y, i, c] - scaled[1, k, Y-1, i, c]
            else
                dy = T(0.5) * (scaled[1, k, j+1, i, c] - scaled[1, k, j-1, i, c])
            end

            # Z derivative (output channel 2, stored at deriv index 3)
            if k == 1
                dz = scaled[1, 2, j, i, c] - scaled[1, 1, j, i, c]
            elseif k == Z
                dz = scaled[1, Z, j, i, c] - scaled[1, Z-1, j, i, c]
            else
                dz = T(0.5) * (scaled[1, k+1, j, i, c] - scaled[1, k-1, j, i, c])
            end

            # Store in output (component, Z, Y, X, deriv_dir)
            # deriv_dir: 1=X, 2=Y, 3=Z (matching torchreg channel order 0,1,2)
            gradients[c, k, j, i, 1] = dx
            gradients[c, k, j, i, 2] = dy
            gradients[c, k, j, i, 3] = dz
        end
    end

    return gradients
end
