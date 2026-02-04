# High-level clinical registration API for MedicalImageRegistration.jl
# GPU-first architecture with AK.foreachindex + Mooncake rrule!!
#
# Provides a simple interface for clinical CT registration workflow:
# 1. PREPROCESS: Align centers of mass, crop to overlap, resample
# 2. REGISTER: Run registration at low resolution with MI loss
# 3. UPSAMPLE: Upsample transform to high resolution
# 4. APPLY: Transform original image with nearest-neighbor for HU preservation
#
# The preprocessing step is critical for clinical CT with FOV mismatch:
# - CCTA (tight FOV) vs Non-contrast (wide FOV)
# - Without COM alignment, gradient descent cannot find solution
#
# Use case: Cardiac CT registration (3mm non-contrast vs 0.5mm contrast)

# ============================================================================
# ClinicalRegistrationResult
# ============================================================================

"""
    ClinicalRegistrationResult{T, A<:AbstractArray{T}}

Result of a clinical registration operation.

# Fields
- `moved_image::PhysicalImage{T}`: The registered moving image in static space
- `transform::A`: The computed displacement field (high-resolution)
- `metrics::Dict{Symbol, T}`: Registration metrics (mi_before, mi_after, etc.)
- `metadata::Dict{Symbol, Any}`: Processing metadata (resolutions, parameters)

# Example
```julia
result = register_clinical(moving, static)

# Access the moved image
moved = result.moved_image

# Check registration quality
println("MI improved: \$(result.metrics[:mi_before]) → \$(result.metrics[:mi_after])")

# Apply transform to another image (e.g., segmentation mask)
mask_moved = transform_clinical(result, mask)
```
"""
struct ClinicalRegistrationResult{T, N, A<:AbstractArray{T}}
    moved_image::PhysicalImage{T, N, A}
    transform::A  # Displacement field (X, Y, [Z], D, N)
    inverse_transform::Union{Nothing, A}  # Optional inverse
    metrics::Dict{Symbol, T}
    metadata::Dict{Symbol, Any}
end

# ============================================================================
# Main Registration Function
# ============================================================================

"""
    register_clinical(moving::PhysicalImage, static::PhysicalImage; kwargs...) -> ClinicalRegistrationResult

Register a moving image to a static image using a clinical workflow optimized for
medical CT imaging with potentially different resolutions, FOVs, and contrast agents.

# Arguments
- `moving::PhysicalImage`: The image to be transformed (e.g., 0.5mm contrast CT)
- `static::PhysicalImage`: The reference image (e.g., 3mm non-contrast CT)

# Keyword Arguments
## Preprocessing
- `preprocess::Bool=true`: Run preprocessing pipeline (COM alignment, overlap crop, resample)
- `center_of_mass_init::Bool=true`: Align centers of mass before registration
- `crop_to_overlap::Bool=true`: Crop both images to overlapping FOV region
- `window_hu::Bool=true`: Apply HU windowing for preprocessing
- `min_hu::Real=-200`: Minimum HU for windowing
- `max_hu::Real=1000`: Maximum HU for windowing
- `com_threshold::Real=-200`: Threshold for COM computation (excludes air/lung)

## Registration
- `registration_resolution::Real=2.0`: Isotropic resolution (mm) for registration
- `loss_fn::Function=mi_loss`: Loss function (mi_loss for multi-modal, mse_loss for same-modality)
- `preserve_hu::Bool=true`: Use nearest-neighbor for final output to preserve exact HU values
- `registration_type::Symbol=:affine`: Registration type (:affine or :syn)
- `affine_scales::Tuple=(4, 2, 1)`: Multi-resolution scales for affine
- `affine_iterations::Tuple=(100, 50, 25)`: Iterations per scale for affine
- `syn_scales::Tuple=(4, 2, 1)`: Multi-resolution scales for SyN
- `syn_iterations::Tuple=(30, 30, 10)`: Iterations per scale for SyN
- `learning_rate::Real=0.01`: Optimizer learning rate
- `compute_inverse::Bool=false`: Also compute inverse transform
- `verbose::Bool=true`: Print progress information

# Returns
- `ClinicalRegistrationResult` containing moved image, transform, and metrics

# Workflow
1. **PREPROCESS** (if `preprocess=true`):
   - Compute and align centers of mass (handles FOV mismatch)
   - Detect overlapping region between images
   - Crop both images to overlap
   - Resample to registration resolution
   - Apply HU windowing
2. **REGISTER**: Run affine or SyN registration with specified loss function
3. **UPSAMPLE**: Upsample displacement field to original moving image resolution
4. **APPLY**: Transform ORIGINAL moving image (not preprocessed) to static space
   using nearest-neighbor (if `preserve_hu=true`) for HU preservation

# Why Preprocessing?
For clinical CT with FOV mismatch:
- CCTA: 180mm FOV, tight on heart
- Non-contrast: 350mm FOV, includes whole chest
- Without COM alignment, optimization starts far from solution
- Gradient descent cannot escape local minimum

# Why Mutual Information (MI)?
For clinical CT with contrast mismatch:
- Blood: 40 HU (non-contrast) vs 300 HU (contrast)
- MSE/NCC would PENALIZE correct alignment!
- MI measures statistical dependence - learns that 40 HU ↔ 300 HU

# Example
```julia
using MedicalImageRegistration
using Metal

# Load cardiac CT scans with different parameters
# Non-contrast: 3mm slice thickness, large FOV (350mm)
non_contrast = PhysicalImage(Array{Float32}(volume1);
    spacing=(0.7, 0.7, 3.0), origin=(-175.0, -175.0, 0.0))

# Contrast CCTA: 0.5mm slice thickness, tight FOV (180mm)
contrast = PhysicalImage(Array{Float32}(volume2);
    spacing=(0.5, 0.5, 0.5), origin=(-90.0, -90.0, 50.0))

# Register contrast (moving) to non-contrast (static)
result = register_clinical(
    contrast, non_contrast;
    preprocess=true,              # Critical for FOV mismatch!
    center_of_mass_init=true,     # Align hearts first
    registration_resolution=2.0,  # 2mm isotropic for speed
    loss_fn=mi_loss,              # Handles contrast difference
    preserve_hu=true,             # Exact HU values in output
    registration_type=:syn,       # Diffeomorphic for local deformation
    verbose=true
)

# Result has exact HU values from original contrast CT
@assert Set(unique(result.moved_image.data)) ⊆ Set(unique(contrast.data))
```
"""
function register_clinical(
    moving::PhysicalImage{T, 5, A},
    static::PhysicalImage{T, 5, A};
    # Preprocessing options
    preprocess::Bool=true,
    center_of_mass_init::Bool=true,
    crop_to_overlap::Bool=true,
    window_hu::Bool=true,
    min_hu::Real=T(-200),
    max_hu::Real=T(1000),
    com_threshold::Real=T(-200),
    # Registration options
    registration_resolution::Real=T(2),
    loss_fn::Function=mi_loss,
    preserve_hu::Bool=true,
    registration_type::Symbol=:affine,
    affine_scales::Tuple{Vararg{Int}}=(4, 2, 1),
    affine_iterations::Tuple{Vararg{Int}}=(100, 50, 25),
    syn_scales::Tuple{Vararg{Int}}=(4, 2, 1),
    syn_iterations::Tuple{Vararg{Int}}=(30, 30, 10),
    learning_rate::Real=T(0.01),
    compute_inverse::Bool=false,
    verbose::Bool=true
) where {T, A<:AbstractArray{T,5}}
    reg_res = T(registration_resolution)
    lr = T(learning_rate)

    # Store metadata
    metadata = Dict{Symbol, Any}(
        :moving_spacing => spatial_spacing(moving),
        :static_spacing => spatial_spacing(static),
        :moving_size => spatial_size(moving),
        :static_size => spatial_size(static),
        :registration_resolution => reg_res,
        :registration_type => registration_type,
        :loss_fn => string(loss_fn),
        :preserve_hu => preserve_hu,
        :preprocess => preprocess,
        :center_of_mass_init => center_of_mass_init
    )

    if verbose
        println("═" ^ 60)
        println("Clinical Registration Workflow")
        println("═" ^ 60)
        println("Moving image: $(spatial_size(moving)), spacing=$(spatial_spacing(moving)) mm")
        println("Static image: $(spatial_size(static)), spacing=$(spatial_spacing(static)) mm")
        println("Registration resolution: $(reg_res) mm isotropic")
        println("Loss function: $(string(loss_fn))")
        println("Registration type: $registration_type")
        println("Preserve HU: $preserve_hu")
        println("Preprocessing: $preprocess")
        println("─" ^ 60)
    end

    metrics = Dict{Symbol, T}()

    # ======================================================================
    # Step 1: PREPROCESSING
    # ======================================================================
    preprocess_info = nothing
    moving_for_registration = moving
    static_for_registration = static

    if preprocess
        if verbose
            println("Step 1: Preprocessing (COM alignment, overlap detection, resampling)")
            println("─" ^ 60)
        end

        moving_prep, static_prep, preprocess_info = preprocess_for_registration(
            moving, static;
            registration_resolution=reg_res,
            align_com=center_of_mass_init,
            do_crop_to_overlap=crop_to_overlap,
            window_hu=window_hu,
            min_hu=T(min_hu),
            max_hu=T(max_hu),
            com_threshold=T(com_threshold)
        )

        if verbose
            println("  COM moving: $(round.(preprocess_info.com_moving, digits=1)) mm")
            println("  COM static: $(round.(preprocess_info.com_static, digits=1)) mm")
            println("  Translation applied: $(round.(preprocess_info.translation, digits=1)) mm")
            if !isnothing(preprocess_info.overlap_region)
                overlap_extent = preprocess_info.overlap_region.max .- preprocess_info.overlap_region.min
                println("  Overlap region: $(round.(overlap_extent, digits=1)) mm")
            end
            println("  Preprocessed size: $(size(parent(moving_prep)))")
        end

        moving_for_registration = moving_prep
        static_for_registration = static_prep

        # After preprocessing, sizes might still differ by 1-2 voxels due to rounding
        # Ensure matching sizes for registration
        moving_prep_size = spatial_size(moving_for_registration)
        static_prep_size = spatial_size(static_for_registration)
        if moving_prep_size != static_prep_size
            if verbose
                println("  Size adjustment: moving $(moving_prep_size) → static $(static_prep_size)")
            end
            moving_for_registration = _resample_physical_to_size(moving_for_registration, static_prep_size)
        end

        # Store preprocessing info in metadata
        metadata[:preprocess_translation] = preprocess_info.translation
        metadata[:preprocess_com_moving] = preprocess_info.com_moving
        metadata[:preprocess_com_static] = preprocess_info.com_static
        if !isnothing(preprocess_info.overlap_region)
            metadata[:preprocess_overlap_region] = preprocess_info.overlap_region
        end
    else
        if verbose
            println("Step 1: Preprocessing SKIPPED (preprocess=false)")
            println("─" ^ 60)
        end

        # Without preprocessing, just resample to registration resolution
        common_spacing = (reg_res, reg_res, reg_res)
        moving_for_registration = resample(moving, common_spacing; interpolation=:bilinear)
        static_for_registration = resample(static, common_spacing; interpolation=:bilinear)

        # Match sizes if different
        moving_size_lowres = spatial_size(moving_for_registration)
        static_size_lowres = spatial_size(static_for_registration)
        if moving_size_lowres != static_size_lowres
            if verbose
                println("  Resampling moving $(moving_size_lowres) to match static $(static_size_lowres)")
            end
            moving_for_registration = _resample_physical_to_size(moving_for_registration, static_size_lowres)
        end
    end

    # Compute initial MI (before registration)
    mi_before_arr = mi_loss(moving_for_registration.data, static_for_registration.data)
    metrics[:mi_before] = -AK.reduce(+, mi_before_arr; init=zero(T))

    if verbose
        println("  Initial MI: $(round(metrics[:mi_before], digits=4))")
    end

    # ======================================================================
    # Step 2: REGISTRATION at low resolution
    # ======================================================================
    if verbose
        println("─" ^ 60)
        println("Step 2: Registration at $(reg_res) mm resolution")
        println("─" ^ 60)
    end

    transform_lowres, inverse_lowres = _register_at_resolution(
        moving_for_registration, static_for_registration;
        registration_type=registration_type,
        loss_fn=loss_fn,
        affine_scales=affine_scales,
        affine_iterations=affine_iterations,
        syn_scales=syn_scales,
        syn_iterations=syn_iterations,
        learning_rate=lr,
        compute_inverse=compute_inverse,
        verbose=verbose
    )

    # ======================================================================
    # Step 3: UPSAMPLE transform to original moving resolution
    # ======================================================================
    if verbose
        println("─" ^ 60)
        println("Step 3: Upsampling transform to original moving resolution")
        println("─" ^ 60)
    end

    target_size = spatial_size(moving)
    transform_highres = resample_displacement(transform_lowres, target_size)

    inverse_highres = if compute_inverse && inverse_lowres !== nothing
        resample_displacement(inverse_lowres, spatial_size(static))
    else
        nothing
    end

    if verbose
        println("  Transform upsampled: $(size(transform_lowres)[1:3]) → $(target_size)")
    end

    # ======================================================================
    # Step 4: APPLY transform to ORIGINAL moving image (not preprocessed!)
    # ======================================================================
    if verbose
        println("─" ^ 60)
        println("Step 4: Applying transform to ORIGINAL moving image")
        println("─" ^ 60)
    end

    # If preprocessing was applied, we need to compose transforms:
    # The registration transform was learned on preprocessed images.
    # To apply to original, we need: original → (preprocess translation) → (registration transform)
    #
    # However, the displacement field is already in the correct coordinate frame
    # because we upsampled it to the original moving image size.
    #
    # For COM alignment, the translation was applied via origin shift, not data resampling.
    # So the displacement field coordinates are still correct for the original image.

    interpolation = preserve_hu ? :nearest : :bilinear
    moved_data = spatial_transform(moving.data, transform_highres; interpolation=interpolation)

    # Create result PhysicalImage with static's spacing (moved is now in static space)
    moved_image = PhysicalImage(moved_data; spacing=spatial_spacing(static), origin=static.origin)

    if verbose && preserve_hu
        println("  Using nearest-neighbor interpolation for HU preservation")
        # Verify HU preservation
        original_values = Set(vec(Array(moving.data)))
        moved_values = Set(vec(Array(moved_data)))
        if moved_values ⊆ original_values
            println("  ✓ HU values preserved (output values ⊆ input values)")
        else
            println("  ⚠ Warning: Some interpolated values detected")
        end
    end

    # ======================================================================
    # Step 5: Compute metrics after registration
    # ======================================================================
    # Resample moved to low res for fair comparison
    common_spacing = (reg_res, reg_res, reg_res)
    moved_lowres = resample(moved_image, common_spacing; interpolation=:bilinear)

    # Ensure sizes match for MI computation
    static_size_lowres = spatial_size(static_for_registration)
    moved_lowres_size = spatial_size(moved_lowres)
    if moved_lowres_size != static_size_lowres
        moved_lowres = _resample_physical_to_size(moved_lowres, static_size_lowres)
    end

    mi_after_arr = mi_loss(moved_lowres.data, static_for_registration.data)
    metrics[:mi_after] = -AK.reduce(+, mi_after_arr; init=zero(T))
    metrics[:mi_improvement] = metrics[:mi_after] - metrics[:mi_before]

    if verbose
        println("─" ^ 60)
        println("Registration complete!")
        println("MI: $(round(metrics[:mi_before], digits=4)) → $(round(metrics[:mi_after], digits=4))")
        println("MI improvement: $(round(metrics[:mi_improvement], digits=4))")
        println("═" ^ 60)
    end

    return ClinicalRegistrationResult{T, 5, A}(
        moved_image,
        transform_highres,
        inverse_highres,
        metrics,
        metadata
    )
end

# 2D version
function register_clinical(
    moving::PhysicalImage{T, 4, A},
    static::PhysicalImage{T, 4, A};
    # Preprocessing options
    preprocess::Bool=true,
    center_of_mass_init::Bool=true,
    crop_to_overlap::Bool=true,
    window_hu::Bool=true,
    min_hu::Real=T(-200),
    max_hu::Real=T(1000),
    com_threshold::Real=T(-200),
    # Registration options
    registration_resolution::Real=T(2),
    loss_fn::Function=mi_loss,
    preserve_hu::Bool=true,
    registration_type::Symbol=:affine,
    affine_scales::Tuple{Vararg{Int}}=(4, 2, 1),
    affine_iterations::Tuple{Vararg{Int}}=(100, 50, 25),
    learning_rate::Real=T(0.01),
    compute_inverse::Bool=false,
    verbose::Bool=true
) where {T, A<:AbstractArray{T,4}}
    reg_res = T(registration_resolution)
    lr = T(learning_rate)

    metadata = Dict{Symbol, Any}(
        :moving_spacing => spatial_spacing(moving),
        :static_spacing => spatial_spacing(static),
        :moving_size => spatial_size(moving),
        :static_size => spatial_size(static),
        :registration_resolution => reg_res,
        :registration_type => registration_type,
        :loss_fn => string(loss_fn),
        :preserve_hu => preserve_hu,
        :preprocess => preprocess,
        :center_of_mass_init => center_of_mass_init
    )

    if verbose
        println("═" ^ 60)
        println("Clinical Registration Workflow (2D)")
        println("═" ^ 60)
        println("Moving image: $(spatial_size(moving)), spacing=$(spatial_spacing(moving)) mm")
        println("Static image: $(spatial_size(static)), spacing=$(spatial_spacing(static)) mm")
        println("Registration resolution: $(reg_res) mm isotropic")
        println("Preprocessing: $preprocess")
        println("─" ^ 60)
    end

    metrics = Dict{Symbol, T}()

    # Step 1: Preprocessing
    preprocess_info = nothing
    moving_for_registration = moving
    static_for_registration = static

    if preprocess
        if verbose
            println("Step 1: Preprocessing (COM alignment, overlap detection, resampling)")
            println("─" ^ 60)
        end

        moving_prep, static_prep, preprocess_info = preprocess_for_registration(
            moving, static;
            registration_resolution=reg_res,
            align_com=center_of_mass_init,
            do_crop_to_overlap=crop_to_overlap,
            window_hu=window_hu,
            min_hu=T(min_hu),
            max_hu=T(max_hu),
            com_threshold=T(com_threshold)
        )

        if verbose
            println("  COM moving: $(round.(preprocess_info.com_moving[1:2], digits=1)) mm")
            println("  COM static: $(round.(preprocess_info.com_static[1:2], digits=1)) mm")
            println("  Translation applied: $(round.(preprocess_info.translation[1:2], digits=1)) mm")
            println("  Preprocessed size: $(size(parent(moving_prep)))")
        end

        moving_for_registration = moving_prep
        static_for_registration = static_prep

        # After preprocessing, sizes might still differ by 1-2 voxels due to rounding
        moving_prep_size = spatial_size(moving_for_registration)
        static_prep_size = spatial_size(static_for_registration)
        if moving_prep_size != static_prep_size
            if verbose
                println("  Size adjustment: moving $(moving_prep_size) → static $(static_prep_size)")
            end
            moving_for_registration = _resample_physical_to_size(moving_for_registration, static_prep_size)
        end

        metadata[:preprocess_translation] = preprocess_info.translation
        metadata[:preprocess_com_moving] = preprocess_info.com_moving
        metadata[:preprocess_com_static] = preprocess_info.com_static
    else
        if verbose
            println("Step 1: Preprocessing SKIPPED (preprocess=false)")
            println("─" ^ 60)
        end

        common_spacing = (reg_res, reg_res)
        moving_for_registration = resample(moving, common_spacing; interpolation=:bilinear)
        static_for_registration = resample(static, common_spacing; interpolation=:bilinear)

        moving_size_lowres = spatial_size(moving_for_registration)
        static_size_lowres = spatial_size(static_for_registration)
        if moving_size_lowres != static_size_lowres
            if verbose
                println("  Resampling moving $(moving_size_lowres) to match static $(static_size_lowres)")
            end
            moving_for_registration = _resample_physical_to_size(moving_for_registration, static_size_lowres)
        end
    end

    mi_before_arr = mi_loss(moving_for_registration.data, static_for_registration.data)
    metrics[:mi_before] = -AK.reduce(+, mi_before_arr; init=zero(T))

    if verbose
        println("  Initial MI: $(round(metrics[:mi_before], digits=4))")
    end

    # Step 2: Register
    if verbose
        println("─" ^ 60)
        println("Step 2: Registration at $(reg_res) mm resolution")
        println("─" ^ 60)
    end

    transform_lowres, inverse_lowres = _register_at_resolution_2d(
        moving_for_registration, static_for_registration;
        registration_type=registration_type,
        loss_fn=loss_fn,
        affine_scales=affine_scales,
        affine_iterations=affine_iterations,
        learning_rate=lr,
        compute_inverse=compute_inverse,
        verbose=verbose
    )

    # Step 3: Upsample
    if verbose
        println("─" ^ 60)
        println("Step 3: Upsampling transform to original moving resolution")
        println("─" ^ 60)
    end

    target_size = spatial_size(moving)
    transform_highres = resample_displacement(transform_lowres, target_size)

    inverse_highres = if compute_inverse && inverse_lowres !== nothing
        resample_displacement(inverse_lowres, spatial_size(static))
    else
        nothing
    end

    if verbose
        println("  Transform upsampled: $(size(transform_lowres)[1:2]) → $(target_size)")
    end

    # Step 4: Apply to original
    if verbose
        println("─" ^ 60)
        println("Step 4: Applying transform to ORIGINAL moving image")
        println("─" ^ 60)
    end

    interpolation = preserve_hu ? :nearest : :bilinear
    moved_data = _apply_displacement_2d(moving.data, transform_highres; interpolation=interpolation)

    moved_image = PhysicalImage(moved_data; spacing=spatial_spacing(static), origin=(static.origin[1], static.origin[2]))

    if verbose && preserve_hu
        println("  Using nearest-neighbor interpolation for HU preservation")
        original_values = Set(vec(Array(moving.data)))
        moved_values = Set(vec(Array(moved_data)))
        if moved_values ⊆ original_values
            println("  ✓ HU values preserved (output values ⊆ input values)")
        else
            println("  ⚠ Warning: Some interpolated values detected")
        end
    end

    # Metrics after
    common_spacing = (reg_res, reg_res)
    moved_lowres = resample(moved_image, common_spacing; interpolation=:bilinear)

    static_size_lowres = spatial_size(static_for_registration)
    moved_lowres_size = spatial_size(moved_lowres)
    if moved_lowres_size != static_size_lowres
        moved_lowres = _resample_physical_to_size(moved_lowres, static_size_lowres)
    end

    mi_after_arr = mi_loss(moved_lowres.data, static_for_registration.data)
    metrics[:mi_after] = -AK.reduce(+, mi_after_arr; init=zero(T))
    metrics[:mi_improvement] = metrics[:mi_after] - metrics[:mi_before]

    if verbose
        println("─" ^ 60)
        println("Registration complete!")
        println("MI: $(round(metrics[:mi_before], digits=4)) → $(round(metrics[:mi_after], digits=4))")
        println("MI improvement: $(round(metrics[:mi_improvement], digits=4))")
        println("═" ^ 60)
    end

    return ClinicalRegistrationResult{T, 4, A}(
        moved_image,
        transform_highres,
        inverse_highres,
        metrics,
        metadata
    )
end

# ============================================================================
# Internal Registration at Specific Resolution
# ============================================================================

function _register_at_resolution(
    moving::PhysicalImage{T, 5},
    static::PhysicalImage{T, 5};
    registration_type::Symbol,
    loss_fn::Function,
    affine_scales::Tuple,
    affine_iterations::Tuple,
    syn_scales::Tuple,
    syn_iterations::Tuple,
    learning_rate::T,
    compute_inverse::Bool,
    verbose::Bool
) where T
    if registration_type === :affine
        return _register_affine_3d(
            moving.data, static.data;
            scales=affine_scales,
            iterations=affine_iterations,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            verbose=verbose
        )
    elseif registration_type === :syn
        return _register_syn_3d(
            moving.data, static.data;
            scales=syn_scales,
            iterations=syn_iterations,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            compute_inverse=compute_inverse,
            verbose=verbose
        )
    else
        error("Unknown registration_type: $registration_type. Use :affine or :syn.")
    end
end

function _register_at_resolution_2d(
    moving::PhysicalImage{T, 4},
    static::PhysicalImage{T, 4};
    registration_type::Symbol,
    loss_fn::Function,
    affine_scales::Tuple,
    affine_iterations::Tuple,
    learning_rate::T,
    compute_inverse::Bool,
    verbose::Bool
) where T
    if registration_type === :affine
        return _register_affine_2d(
            moving.data, static.data;
            scales=affine_scales,
            iterations=affine_iterations,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            verbose=verbose
        )
    else
        error("SyN registration is only implemented for 3D. Use :affine for 2D.")
    end
end

# ============================================================================
# Affine Registration Wrapper
# ============================================================================

function _register_affine_3d(
    moving::AbstractArray{T,5},
    static::AbstractArray{T,5};
    scales::Tuple,
    iterations::Tuple,
    loss_fn::Function,
    learning_rate::T,
    verbose::Bool
) where T
    N = size(moving, 5)

    # Create AffineRegistration
    reg = AffineRegistration{T}(
        is_3d=true,
        batch_size=N,
        scales=scales,
        iterations=iterations,
        learning_rate=learning_rate,
        with_translation=true,
        with_rotation=true,
        with_zoom=true,
        with_shear=false,
        array_type=typeof(moving).name.wrapper
    )

    # Register
    moved = register(reg, moving, static; loss_fn=loss_fn, verbose=verbose, final_interpolation=:bilinear)

    # Convert affine to displacement field
    theta = get_affine(reg)
    X, Y, Z = size(static, 1), size(static, 2), size(static, 3)

    # Generate displacement field from affine
    grid = affine_grid(theta, (X, Y, Z); align_corners=true)
    id_grid = _create_identity_grid_clinical((X, Y, Z), grid)

    # Displacement = grid - identity
    disp = _compute_displacement_from_grid(grid, id_grid)

    return disp, nothing  # No inverse for affine (could compute if needed)
end

function _register_affine_2d(
    moving::AbstractArray{T,4},
    static::AbstractArray{T,4};
    scales::Tuple,
    iterations::Tuple,
    loss_fn::Function,
    learning_rate::T,
    verbose::Bool
) where T
    N = size(moving, 4)

    reg = AffineRegistration{T}(
        is_3d=false,
        batch_size=N,
        scales=scales,
        iterations=iterations,
        learning_rate=learning_rate,
        with_translation=true,
        with_rotation=true,
        with_zoom=true,
        with_shear=false,
        array_type=typeof(moving).name.wrapper
    )

    moved = register(reg, moving, static; loss_fn=loss_fn, verbose=verbose, final_interpolation=:bilinear)

    theta = get_affine(reg)
    X, Y = size(static, 1), size(static, 2)

    grid = affine_grid(theta, (X, Y); align_corners=true)
    id_grid = _create_identity_grid_clinical_2d((X, Y), grid)

    disp = _compute_displacement_from_grid_2d(grid, id_grid)

    return disp, nothing
end

# ============================================================================
# SyN Registration Wrapper
# ============================================================================

function _register_syn_3d(
    moving::AbstractArray{T,5},
    static::AbstractArray{T,5};
    scales::Tuple,
    iterations::Tuple,
    loss_fn::Function,
    learning_rate::T,
    compute_inverse::Bool,
    verbose::Bool
) where T
    reg = SyNRegistration{T}(
        scales=scales,
        iterations=iterations,
        learning_rate=learning_rate,
        verbose=verbose,
        array_type=typeof(moving).name.wrapper
    )

    moved_xy, moved_yx, flow_xy, flow_yx = register(
        reg, moving, static;
        loss_fn=loss_fn,
        final_interpolation=:bilinear
    )

    inverse = compute_inverse ? flow_yx : nothing

    return flow_xy, inverse
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    _resample_physical_to_size(img::PhysicalImage, target_size)

Resample a PhysicalImage to a specific target size while preserving spacing.
Used when images with different original extents need to be registered.
"""
function _resample_physical_to_size(
    img::PhysicalImage{T, 5},
    target_size::NTuple{3,Int};
    interpolation::Symbol=:bilinear
) where T
    X_out, Y_out, Z_out = target_size
    C, N = size(img.data, 4), size(img.data, 5)

    # Create identity affine
    theta = similar(img.data, 3, 4, N)
    _fill_identity_affine_clinical_3d!(theta)

    # Generate grid and sample
    grid = affine_grid(theta, target_size; align_corners=true)
    resampled_data = grid_sample(img.data, grid; padding_mode=:border, align_corners=true, interpolation=interpolation)

    return PhysicalImage(resampled_data; spacing=img.spacing, origin=img.origin)
end

function _resample_physical_to_size(
    img::PhysicalImage{T, 4},
    target_size::NTuple{2,Int};
    interpolation::Symbol=:bilinear
) where T
    X_out, Y_out = target_size
    C, N = size(img.data, 3), size(img.data, 4)

    # Create identity affine
    theta = similar(img.data, 2, 3, N)
    _fill_identity_affine_clinical_2d!(theta)

    # Generate grid and sample
    grid = affine_grid(theta, target_size; align_corners=true)
    resampled_data = grid_sample(img.data, grid; padding_mode=:border, align_corners=true, interpolation=interpolation)

    return PhysicalImage(resampled_data; spacing=spatial_spacing(img), origin=(img.origin[1], img.origin[2]))
end

function _fill_identity_affine_clinical_3d!(theta::AbstractArray{T,3}) where T
    N = size(theta, 3)
    theta_cpu = zeros(T, 3, 4, N)
    for n in 1:N
        theta_cpu[1, 1, n] = one(T)
        theta_cpu[2, 2, n] = one(T)
        theta_cpu[3, 3, n] = one(T)
    end
    copyto!(theta, theta_cpu)
    return nothing
end

function _fill_identity_affine_clinical_2d!(theta::AbstractArray{T,3}) where T
    N = size(theta, 3)
    theta_cpu = zeros(T, 2, 3, N)
    for n in 1:N
        theta_cpu[1, 1, n] = one(T)
        theta_cpu[2, 2, n] = one(T)
    end
    copyto!(theta, theta_cpu)
    return nothing
end

"""
    _create_identity_grid_clinical(shape, reference)

Create identity grid for computing displacement from affine grid.
"""
function _create_identity_grid_clinical(shape::NTuple{3,Int}, reference::AbstractArray{T}) where T
    X, Y, Z = shape
    N = size(reference, 5)

    grid = similar(reference, 3, X, Y, Z, N)

    AK.foreachindex(grid) do idx
        d, i, j, k, n = _linear_to_cartesian_5d_clinical(idx, X, Y, Z)

        x_norm = T(2) * (T(i) - one(T)) / T(max(X - 1, 1)) - one(T)
        y_norm = T(2) * (T(j) - one(T)) / T(max(Y - 1, 1)) - one(T)
        z_norm = T(2) * (T(k) - one(T)) / T(max(Z - 1, 1)) - one(T)

        if d == 1
            @inbounds grid[idx] = x_norm
        elseif d == 2
            @inbounds grid[idx] = y_norm
        else
            @inbounds grid[idx] = z_norm
        end
    end

    return grid
end

function _create_identity_grid_clinical_2d(shape::NTuple{2,Int}, reference::AbstractArray{T}) where T
    X, Y = shape
    N = size(reference, 4)

    grid = similar(reference, 2, X, Y, N)

    AK.foreachindex(grid) do idx
        d, i, j, n = _linear_to_cartesian_4d_clinical(idx, X, Y)

        x_norm = T(2) * (T(i) - one(T)) / T(max(X - 1, 1)) - one(T)
        y_norm = T(2) * (T(j) - one(T)) / T(max(Y - 1, 1)) - one(T)

        if d == 1
            @inbounds grid[idx] = x_norm
        else
            @inbounds grid[idx] = y_norm
        end
    end

    return grid
end

"""
    _compute_displacement_from_grid(grid, id_grid)

Compute displacement field from affine grid and identity grid.
Grid format: (D, X, Y, Z, N) → Displacement format: (X, Y, Z, D, N)
"""
function _compute_displacement_from_grid(
    grid::AbstractArray{T,5},
    id_grid::AbstractArray{T,5}
) where T
    _, X, Y, Z, N = size(grid)
    disp = similar(grid, X, Y, Z, 3, N)

    AK.foreachindex(disp) do idx
        i, j, k, d, n = _linear_to_cartesian_5d_disp_clinical(idx, X, Y, Z)
        @inbounds disp[idx] = grid[d, i, j, k, n] - id_grid[d, i, j, k, n]
    end

    return disp
end

function _compute_displacement_from_grid_2d(
    grid::AbstractArray{T,4},
    id_grid::AbstractArray{T,4}
) where T
    _, X, Y, N = size(grid)
    disp = similar(grid, X, Y, 2, N)

    AK.foreachindex(disp) do idx
        i, j, d, n = _linear_to_cartesian_4d_disp_clinical(idx, X, Y)
        @inbounds disp[idx] = grid[d, i, j, n] - id_grid[d, i, j, n]
    end

    return disp
end

"""
    _apply_displacement_2d(image, disp; interpolation)

Apply 2D displacement field to image using spatial transform approach.
"""
function _apply_displacement_2d(
    image::AbstractArray{T,4},
    disp::AbstractArray{T,4};
    interpolation::Symbol=:bilinear
) where T
    X, Y, C, N = size(image)
    X_d, Y_d, D, N_d = size(disp)

    @assert (X, Y) == (X_d, Y_d) "Spatial dimensions must match"
    @assert D == 2 "Displacement must have 2 channels"
    @assert N == N_d "Batch sizes must match"

    # Create identity grid + displacement
    grid = _create_displaced_grid_2d_clinical(disp)

    return grid_sample(image, grid; interpolation=interpolation, padding_mode=:border, align_corners=true)
end

function _create_displaced_grid_2d_clinical(disp::AbstractArray{T,4}) where T
    X, Y, D, N = size(disp)

    grid = similar(disp, 2, X, Y, N)

    norm_x = T(2) / T(max(X - 1, 1))
    norm_y = T(2) / T(max(Y - 1, 1))

    AK.foreachindex(grid) do idx
        d, i, j, n = _linear_to_cartesian_4d_clinical(idx, X, Y)

        if d == 1
            id_coord = (T(i) - one(T)) * norm_x - one(T)
            @inbounds grid[idx] = id_coord + disp[i, j, 1, n]
        else
            id_coord = (T(j) - one(T)) * norm_y - one(T)
            @inbounds grid[idx] = id_coord + disp[i, j, 2, n]
        end
    end

    return grid
end

# Index conversion helpers
@inline function _linear_to_cartesian_5d_clinical(idx::Int, X::Int, Y::Int, Z::Int)
    idx_0 = idx - 1
    d = idx_0 % 3 + 1
    idx_0 = idx_0 ÷ 3
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    n = idx_0 ÷ Z + 1
    return d, i, j, k, n
end

@inline function _linear_to_cartesian_4d_clinical(idx::Int, X::Int, Y::Int)
    idx_0 = idx - 1
    d = idx_0 % 2 + 1
    idx_0 = idx_0 ÷ 2
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    n = idx_0 ÷ Y + 1
    return d, i, j, n
end

@inline function _linear_to_cartesian_5d_disp_clinical(idx::Int, X::Int, Y::Int, Z::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    k = idx_0 % Z + 1
    idx_0 = idx_0 ÷ Z
    d = idx_0 % 3 + 1
    n = idx_0 ÷ 3 + 1
    return i, j, k, d, n
end

@inline function _linear_to_cartesian_4d_disp_clinical(idx::Int, X::Int, Y::Int)
    idx_0 = idx - 1
    i = idx_0 % X + 1
    idx_0 = idx_0 ÷ X
    j = idx_0 % Y + 1
    idx_0 = idx_0 ÷ Y
    d = idx_0 % 2 + 1
    n = idx_0 ÷ 2 + 1
    return i, j, d, n
end

# ============================================================================
# Transform Clinical (Apply to New Images)
# ============================================================================

"""
    transform_clinical(result::ClinicalRegistrationResult, image::PhysicalImage; interpolation=:nearest)

Apply a learned clinical registration transform to another image.

This is useful for transforming segmentation masks, other imaging modalities,
or the same image at different time points.

# Arguments
- `result`: A `ClinicalRegistrationResult` from `register_clinical`
- `image`: Image to transform (must have same spatial dimensions as original moving)
- `interpolation`: Interpolation mode (:nearest default for masks/labels, :bilinear for images)

# Returns
- Transformed PhysicalImage

# Example
```julia
# Register contrast CT to non-contrast CT
result = register_clinical(contrast, non_contrast)

# Apply same transform to segmentation mask
mask_transformed = transform_clinical(result, mask; interpolation=:nearest)

# Apply to another modality
pet_transformed = transform_clinical(result, pet; interpolation=:bilinear)
```
"""
function transform_clinical(
    result::ClinicalRegistrationResult{T, 5, A},
    image::PhysicalImage{T, 5};
    interpolation::Symbol=:nearest
) where {T, A}
    # Verify dimensions match original moving image
    expected_size = result.metadata[:moving_size]
    actual_size = spatial_size(image)

    if expected_size != actual_size
        error("Image size $actual_size does not match registration moving size $expected_size")
    end

    # Apply transform
    moved_data = spatial_transform(image.data, result.transform; interpolation=interpolation)

    # Use static spacing (since we're moving to static space)
    static_spacing = result.metadata[:static_spacing]

    return PhysicalImage(moved_data; spacing=static_spacing)
end

function transform_clinical(
    result::ClinicalRegistrationResult{T, 4, A},
    image::PhysicalImage{T, 4};
    interpolation::Symbol=:nearest
) where {T, A}
    expected_size = result.metadata[:moving_size]
    actual_size = spatial_size(image)

    if expected_size != actual_size
        error("Image size $actual_size does not match registration moving size $expected_size")
    end

    moved_data = _apply_displacement_2d(image.data, result.transform; interpolation=interpolation)
    static_spacing = result.metadata[:static_spacing]

    return PhysicalImage(moved_data; spacing=static_spacing)
end

"""
    transform_clinical_inverse(result::ClinicalRegistrationResult, image::PhysicalImage; interpolation=:nearest)

Apply the inverse transform (static → moving direction).

Only available if `compute_inverse=true` was set during registration.
"""
function transform_clinical_inverse(
    result::ClinicalRegistrationResult{T, 5, A},
    image::PhysicalImage{T, 5};
    interpolation::Symbol=:nearest
) where {T, A}
    if result.inverse_transform === nothing
        error("Inverse transform not computed. Use compute_inverse=true in register_clinical.")
    end

    expected_size = result.metadata[:static_size]
    actual_size = spatial_size(image)

    if expected_size != actual_size
        error("Image size $actual_size does not match registration static size $expected_size")
    end

    moved_data = spatial_transform(image.data, result.inverse_transform; interpolation=interpolation)
    moving_spacing = result.metadata[:moving_spacing]

    return PhysicalImage(moved_data; spacing=moving_spacing)
end
