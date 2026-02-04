# GPU-accelerated compose_affine with Mooncake AD support
# Builds affine transformation matrix from translation, rotation, zoom, and shear components
# Dependencies imported by parent module: AK, Mooncake, CoDual, NoFData, NoRData, @is_primitive, MinimalCtx

# ============================================================================
# Julia Array Convention: column-major, dimensions are (D, N) or (D, D, N)
# PyTorch Convention: row-major, dimensions are (N, D) or (N, D, D)
#
# For 2D:
#   translation: (2, N) - [tx; ty] per batch
#   rotation:    (2, 2, N) - 2x2 rotation matrix per batch
#   zoom:        (2, N) - [sx; sy] scale factors per batch
#   shear:       (2, N) - [s_xy; 0] shear parameter (only s_xy used for 2D)
#   output:      (2, 3, N) - 2x3 affine matrix per batch [R*S | t]
#
# For 3D:
#   translation: (3, N) - [tx; ty; tz] per batch
#   rotation:    (3, 3, N) - 3x3 rotation matrix per batch
#   zoom:        (3, N) - [sx; sy; sz] scale factors per batch
#   shear:       (3, N) - [s_xy; s_xz; s_yz] shear parameters
#   output:      (3, 4, N) - 3x4 affine matrix per batch [R*S | t]
# ============================================================================

"""
    compose_affine(translation, rotation, zoom, shear)

Compose an affine transformation matrix from individual components.

The affine matrix is computed as: `[R @ S | t]` where:
- R is the rotation matrix
- S is the scale + shear matrix (upper triangular)
- t is the translation vector

# Arguments
- `translation`: Translation vector (D, N)
- `rotation`: Rotation matrix (D, D, N)
- `zoom`: Scale factors (D, N)
- `shear`: Shear parameters (D, N)
  - For 2D: only shear[1] (s_xy) is used
  - For 3D: shear[1]=s_xy, shear[2]=s_xz, shear[3]=s_yz

# Returns
- Affine matrix (D, D+1, N)

# Example
```julia
# 2D identity affine
translation = zeros(Float32, 2, 1)
rotation = cat(I(2), dims=3) |> x -> Float32.(x)
zoom = ones(Float32, 2, 1)
shear = zeros(Float32, 2, 1)

theta = compose_affine(translation, rotation, zoom, shear)
# theta[:,:,1] is [[1 0 0]; [0 1 0]]
```
"""
function compose_affine(
    translation::AbstractArray{T,2},
    rotation::AbstractArray{T,3},
    zoom::AbstractArray{T,2},
    shear::AbstractArray{T,2}
) where T
    D = size(translation, 1)
    N = size(translation, 2)

    @assert size(rotation) == (D, D, N) "rotation must be (D, D, N)"
    @assert size(zoom) == (D, N) "zoom must be (D, N)"
    @assert size(shear) == (D, N) "shear must be (D, N)"

    if D == 2
        return _compose_affine_2d(translation, rotation, zoom, shear)
    elseif D == 3
        return _compose_affine_3d(translation, rotation, zoom, shear)
    else
        error("Only 2D and 3D affines are supported")
    end
end

# ============================================================================
# 2D Compose Affine
# ============================================================================

function _compose_affine_2d(
    translation::AbstractArray{T,2},
    rotation::AbstractArray{T,3},
    zoom::AbstractArray{T,2},
    shear::AbstractArray{T,2}
) where T
    N = size(translation, 2)

    # Output: (2, 3, N)
    theta = similar(translation, 2, 3, N)

    AK.foreachindex(theta) do idx
        # Linear index to (row, col, n)
        row, col, n = _linear_to_cartesian_3d_compose(idx, 2, 3)

        if col == 3
            # Translation column
            @inbounds theta[idx] = translation[row, n]
        else
            # Compute (R @ S)[row, col] where S is scale + shear matrix
            # S = [sx  sxy]
            #     [0   sy ]
            # For col=1: S[:,1] = [sx; 0]
            # For col=2: S[:,2] = [sxy; sy]

            sx = @inbounds zoom[1, n]
            sy = @inbounds zoom[2, n]
            sxy = @inbounds shear[1, n]

            # S element
            if col == 1
                s1 = sx
                s2 = zero(T)
            else  # col == 2
                s1 = sxy
                s2 = sy
            end

            # R @ S: (R @ S)[row, col] = R[row, 1] * S[1, col] + R[row, 2] * S[2, col]
            r1 = @inbounds rotation[row, 1, n]
            r2 = @inbounds rotation[row, 2, n]

            @inbounds theta[idx] = r1 * s1 + r2 * s2
        end
    end

    return theta
end

# ============================================================================
# 3D Compose Affine
# ============================================================================

function _compose_affine_3d(
    translation::AbstractArray{T,2},
    rotation::AbstractArray{T,3},
    zoom::AbstractArray{T,2},
    shear::AbstractArray{T,2}
) where T
    N = size(translation, 2)

    # Output: (3, 4, N)
    theta = similar(translation, 3, 4, N)

    AK.foreachindex(theta) do idx
        row, col, n = _linear_to_cartesian_3d_compose(idx, 3, 4)

        if col == 4
            # Translation column
            @inbounds theta[idx] = translation[row, n]
        else
            # Compute (R @ S)[row, col] where S is scale + shear matrix
            # S = [sx  sxy sxz]
            #     [0   sy  syz]
            #     [0   0   sz ]
            # S is upper triangular with zoom on diagonal and shear off-diagonal

            sx = @inbounds zoom[1, n]
            sy = @inbounds zoom[2, n]
            sz = @inbounds zoom[3, n]
            sxy = @inbounds shear[1, n]
            sxz = @inbounds shear[2, n]
            syz = @inbounds shear[3, n]

            # Get column of S
            if col == 1
                s1, s2, s3 = sx, zero(T), zero(T)
            elseif col == 2
                s1, s2, s3 = sxy, sy, zero(T)
            else  # col == 3
                s1, s2, s3 = sxz, syz, sz
            end

            # R @ S: (R @ S)[row, col] = sum_k R[row, k] * S[k, col]
            r1 = @inbounds rotation[row, 1, n]
            r2 = @inbounds rotation[row, 2, n]
            r3 = @inbounds rotation[row, 3, n]

            @inbounds theta[idx] = r1 * s1 + r2 * s2 + r3 * s3
        end
    end

    return theta
end

# ============================================================================
# Index Conversion Helper
# ============================================================================

@inline function _linear_to_cartesian_3d_compose(idx::Int, R::Int, C::Int)
    # Shape: (R, C, N)
    idx_0 = idx - 1
    row = idx_0 % R + 1
    idx_0 = idx_0 ÷ R
    col = idx_0 % C + 1
    n = idx_0 ÷ C + 1
    return row, col, n
end

# ============================================================================
# Backward Pass (Gradients)
# ============================================================================

function _∇compose_affine_2d!(
    d_translation::AbstractArray{T,2},
    d_rotation::AbstractArray{T,3},
    d_zoom::AbstractArray{T,2},
    d_shear::AbstractArray{T,2},
    d_theta::AbstractArray{T,3},
    translation::AbstractArray{T,2},
    rotation::AbstractArray{T,3},
    zoom::AbstractArray{T,2},
    shear::AbstractArray{T,2}
) where T
    N = size(translation, 2)

    AK.foreachindex(d_theta) do idx
        row, col, n = _linear_to_cartesian_3d_compose(idx, 2, 3)

        d_out = @inbounds d_theta[idx]
        if d_out == zero(T)
            return nothing
        end

        if col == 3
            # Gradient for translation
            Atomix.@atomic d_translation[row, n] += d_out
        else
            sx = @inbounds zoom[1, n]
            sy = @inbounds zoom[2, n]
            sxy = @inbounds shear[1, n]

            r1 = @inbounds rotation[row, 1, n]
            r2 = @inbounds rotation[row, 2, n]

            if col == 1
                # theta[row, 1] = r1 * sx + r2 * 0 = r1 * sx
                # d_sx += d_out * r1
                # d_r1 += d_out * sx
                Atomix.@atomic d_zoom[1, n] += d_out * r1
                Atomix.@atomic d_rotation[row, 1, n] += d_out * sx
            else  # col == 2
                # theta[row, 2] = r1 * sxy + r2 * sy
                # d_sxy += d_out * r1
                # d_sy += d_out * r2
                # d_r1 += d_out * sxy
                # d_r2 += d_out * sy
                Atomix.@atomic d_shear[1, n] += d_out * r1
                Atomix.@atomic d_zoom[2, n] += d_out * r2
                Atomix.@atomic d_rotation[row, 1, n] += d_out * sxy
                Atomix.@atomic d_rotation[row, 2, n] += d_out * sy
            end
        end

        return nothing
    end

    return nothing
end

function _∇compose_affine_3d!(
    d_translation::AbstractArray{T,2},
    d_rotation::AbstractArray{T,3},
    d_zoom::AbstractArray{T,2},
    d_shear::AbstractArray{T,2},
    d_theta::AbstractArray{T,3},
    translation::AbstractArray{T,2},
    rotation::AbstractArray{T,3},
    zoom::AbstractArray{T,2},
    shear::AbstractArray{T,2}
) where T
    N = size(translation, 2)

    AK.foreachindex(d_theta) do idx
        row, col, n = _linear_to_cartesian_3d_compose(idx, 3, 4)

        d_out = @inbounds d_theta[idx]
        if d_out == zero(T)
            return nothing
        end

        if col == 4
            Atomix.@atomic d_translation[row, n] += d_out
        else
            sx = @inbounds zoom[1, n]
            sy = @inbounds zoom[2, n]
            sz = @inbounds zoom[3, n]
            sxy = @inbounds shear[1, n]
            sxz = @inbounds shear[2, n]
            syz = @inbounds shear[3, n]

            r1 = @inbounds rotation[row, 1, n]
            r2 = @inbounds rotation[row, 2, n]
            r3 = @inbounds rotation[row, 3, n]

            if col == 1
                # theta[row, 1] = r1 * sx + r2 * 0 + r3 * 0
                Atomix.@atomic d_zoom[1, n] += d_out * r1
                Atomix.@atomic d_rotation[row, 1, n] += d_out * sx
            elseif col == 2
                # theta[row, 2] = r1 * sxy + r2 * sy + r3 * 0
                Atomix.@atomic d_shear[1, n] += d_out * r1
                Atomix.@atomic d_zoom[2, n] += d_out * r2
                Atomix.@atomic d_rotation[row, 1, n] += d_out * sxy
                Atomix.@atomic d_rotation[row, 2, n] += d_out * sy
            else  # col == 3
                # theta[row, 3] = r1 * sxz + r2 * syz + r3 * sz
                Atomix.@atomic d_shear[2, n] += d_out * r1
                Atomix.@atomic d_shear[3, n] += d_out * r2
                Atomix.@atomic d_zoom[3, n] += d_out * r3
                Atomix.@atomic d_rotation[row, 1, n] += d_out * sxz
                Atomix.@atomic d_rotation[row, 2, n] += d_out * syz
                Atomix.@atomic d_rotation[row, 3, n] += d_out * sz
            end
        end

        return nothing
    end

    return nothing
end

# ============================================================================
# Mooncake rrule!! Definitions
# ============================================================================

@is_primitive MinimalCtx Tuple{typeof(compose_affine), AbstractArray{<:Any,2}, AbstractArray{<:Any,3}, AbstractArray{<:Any,2}, AbstractArray{<:Any,2}}

function Mooncake.rrule!!(
    ::CoDual{typeof(compose_affine)},
    translation::CoDual{A1, F1},
    rotation::CoDual{A2, F2},
    zoom::CoDual{A3, F3},
    shear::CoDual{A4, F4}
) where {A1<:AbstractArray{<:Any,2}, F1, A2<:AbstractArray{<:Any,3}, F2, A3<:AbstractArray{<:Any,2}, F3, A4<:AbstractArray{<:Any,2}, F4}
    translation_primal = translation.x
    rotation_primal = rotation.x
    zoom_primal = zoom.x
    shear_primal = shear.x

    translation_fdata = translation.dx
    rotation_fdata = rotation.dx
    zoom_fdata = zoom.dx
    shear_fdata = shear.dx

    D = size(translation_primal, 1)

    output = compose_affine(translation_primal, rotation_primal, zoom_primal, shear_primal)
    output_fdata = similar(output)
    fill!(output_fdata, zero(eltype(output)))

    function compose_affine_pullback(_rdata)
        if D == 2
            _∇compose_affine_2d!(
                translation_fdata, rotation_fdata, zoom_fdata, shear_fdata,
                output_fdata, translation_primal, rotation_primal, zoom_primal, shear_primal
            )
        else
            _∇compose_affine_3d!(
                translation_fdata, rotation_fdata, zoom_fdata, shear_fdata,
                output_fdata, translation_primal, rotation_primal, zoom_primal, shear_primal
            )
        end
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(output, output_fdata), compose_affine_pullback
end
