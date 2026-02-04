# MedicalImageRegistration.jl HU Preservation Demo
#
# This demo compares standard bilinear interpolation with HU-preserving
# nearest-neighbor interpolation using the Shepp-Logan phantom.
#
# It demonstrates:
# 1. Standard registration (bilinear): Creates interpolated values
# 2. HU-preserving registration (hybrid mode): Preserves exact input values
#
# For CT images where Hounsfield Unit values are clinically important,
# the hybrid mode ensures no new intensity values are introduced.
#
# GPU Acceleration: Automatically uses Metal GPU if available.

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))
Pkg.add(["TestImages", "Images", "FileIO", "Colors", "Metal", "UnicodePlots"])

using MedicalImageRegistration
using TestImages
using Images
using FileIO
using Colors
using UnicodePlots

# Try to load Metal for GPU acceleration
const USE_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

if USE_GPU
    println("Metal GPU detected - running on GPU")
    using Metal
else
    println("Metal GPU not available - running on CPU")
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
Convert array to GPU if Metal is available.
"""
function to_device(arr)
    if USE_GPU
        return MtlArray(arr)
    else
        return arr
    end
end

"""
Convert array from GPU back to CPU.
"""
function to_cpu(arr)
    if arr isa AbstractArray && !(arr isa Array)
        return Array(arr)
    else
        return arr
    end
end

"""
Convert a grayscale image to the array format expected by MedicalImageRegistration.
"""
function image_to_array(img)
    img_float = Float32.(Gray.(img))
    return permutedims(reshape(img_float, size(img_float)..., 1, 1), (2, 1, 3, 4))
end

"""
Convert registration array back to displayable image.
"""
function array_to_image(arr)
    arr_cpu = to_cpu(arr)
    img_data = permutedims(arr_cpu[:, :, 1, 1], (2, 1))
    return Gray.(clamp.(img_data, 0f0, 1f0))
end

"""
Create a transformed version of an image using affine_transform.
"""
function apply_synthetic_transform(img_arr; translation=(0f0, 0f0), rotation_deg=0f0, zoom=(1f0, 1f0))
    T = Float32

    trans_cpu = T[translation[1]; translation[2];;]

    θ = T(rotation_deg * π / 180)
    c, s = cos(θ), sin(θ)
    rot_cpu = reshape(T[c -s; s c], 2, 2, 1)

    z_cpu = T[zoom[1]; zoom[2];;]
    shear_cpu = zeros(T, 2, 1)

    trans = to_device(trans_cpu)
    rot = to_device(rot_cpu)
    z = to_device(z_cpu)
    shear = to_device(shear_cpu)

    affine = compose_affine(trans, rot, z, shear)
    return affine_transform(img_arr, affine; padding_mode=:border)
end

"""
Blend two images with alpha for visualization.
"""
function blend_images(img1, img2, alpha=0.5f0)
    return Gray.(alpha .* Float32.(img1) .+ (1f0 - alpha) .* Float32.(img2))
end

"""
Create a checkerboard overlay to visualize alignment.
"""
function checkerboard_overlay(img1, img2; block_size=16)
    h, w = size(img1)
    result = similar(img1)
    for i in 1:h, j in 1:w
        block_i = div(i - 1, block_size)
        block_j = div(j - 1, block_size)
        if (block_i + block_j) % 2 == 0
            result[i, j] = img1[i, j]
        else
            result[i, j] = img2[i, j]
        end
    end
    return result
end

"""
Print histogram statistics for an array.
"""
function print_intensity_stats(name, arr; indent="")
    arr_cpu = vec(to_cpu(arr))
    unique_vals = length(unique(arr_cpu))
    println("$(indent)$(name):")
    println("$(indent)  Unique values: $(unique_vals)")
    println("$(indent)  Min: $(minimum(arr_cpu))")
    println("$(indent)  Max: $(maximum(arr_cpu))")
    println("$(indent)  Mean: $(round(sum(arr_cpu)/length(arr_cpu), digits=4))")
end

"""
Check if output values are a subset of input values.
"""
function check_hu_preservation(input_arr, output_arr)
    input_vals = Set(vec(to_cpu(input_arr)))
    output_vals = Set(filter(!iszero, vec(to_cpu(output_arr))))
    is_preserved = issubset(output_vals, input_vals)
    new_vals = setdiff(output_vals, input_vals)
    return is_preserved, new_vals
end

# ============================================================================
# Main Demo
# ============================================================================

function run_hu_preservation_demo()
    println("=" ^ 70)
    println("MedicalImageRegistration.jl - HU Preservation Demo")
    println("Comparing Standard vs HU-Preserving Registration")
    println("=" ^ 70)

    # -------------------------------------------------------------------------
    # 1. Load Shepp-Logan Phantom
    # -------------------------------------------------------------------------
    println("\n1. Loading Shepp-Logan phantom...")

    # Generate 2D Shepp-Logan phantom (256x256)
    # Use TestImages.shepp_logan(N) which generates the phantom directly
    phantom_img = TestImages.shepp_logan(256)
    println("   Using 2D Shepp-Logan phantom (256x256)")
    println("   Image size: $(size(phantom_img))")

    # Convert to array format
    static_arr_cpu = image_to_array(phantom_img)
    static_arr = to_device(static_arr_cpu)
    device_name = USE_GPU ? "GPU (MtlArray)" : "CPU (Array)"
    println("   Array shape: $(size(static_arr)) on $(device_name)")

    # -------------------------------------------------------------------------
    # 2. Analyze Original Intensity Values
    # -------------------------------------------------------------------------
    println("\n2. Analyzing original intensity distribution...")
    print_intensity_stats("Original phantom", static_arr, indent="   ")

    original_unique = sort(unique(vec(to_cpu(static_arr))))
    println("\n   First 10 unique values: $(original_unique[1:min(10, end)])")

    # -------------------------------------------------------------------------
    # 3. Create Misaligned Version
    # -------------------------------------------------------------------------
    println("\n3. Creating misaligned image...")

    translation = (8f0, -6f0)
    rotation_deg = 4f0
    zoom = (0.97f0, 1.03f0)

    norm_trans = (translation[1] / 128f0, translation[2] / 128f0)

    moving_arr = apply_synthetic_transform(
        static_arr;
        translation=norm_trans,
        rotation_deg=rotation_deg,
        zoom=zoom
    )
    println("   Applied: translation=$(translation), rotation=$(rotation_deg)°, zoom=$(zoom)")
    print_intensity_stats("Misaligned (moving) image", moving_arr, indent="   ")

    # Store the moving values for later comparison
    moving_vals = Set(vec(to_cpu(moving_arr)))

    # Convert to images for visualization
    static_img = array_to_image(static_arr)
    moving_img = array_to_image(moving_arr)

    # Save images
    println("\n4. Saving input images...")
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    save(joinpath(output_dir, "hu_static.png"), static_img)
    save(joinpath(output_dir, "hu_moving_before.png"), moving_img)
    save(joinpath(output_dir, "hu_overlay_before.png"), checkerboard_overlay(static_img, moving_img))
    println("   Saved to: $(output_dir)")

    # -------------------------------------------------------------------------
    # 5. Run Standard Registration (Bilinear)
    # -------------------------------------------------------------------------
    println("\n5. Running STANDARD registration (bilinear interpolation)...")

    array_type = USE_GPU ? MtlArray : Array
    reg_standard = AffineRegistration{Float32}(
        is_3d=false,
        scales=(4, 2, 1),
        iterations=(100, 50, 25),
        learning_rate=0.01f0,
        with_translation=true,
        with_rotation=true,
        with_zoom=true,
        with_shear=false,
        array_type=array_type
    )

    moved_standard = register(reg_standard, moving_arr, static_arr;
                              verbose=true, final_interpolation=:bilinear)

    println("\n   Standard registration complete.")
    print_intensity_stats("Standard (bilinear) result", moved_standard, indent="   ")

    # Check HU preservation
    is_preserved_std, new_vals_std = check_hu_preservation(moving_arr, moved_standard)
    println("\n   HU Preserved: $(is_preserved_std)")
    println("   New interpolated values created: $(length(new_vals_std))")

    # -------------------------------------------------------------------------
    # 6. Run HU-Preserving Registration (Hybrid Mode)
    # -------------------------------------------------------------------------
    println("\n6. Running HU-PRESERVING registration (hybrid mode)...")
    println("   (Bilinear during optimization, nearest-neighbor for final output)")

    reg_hu = AffineRegistration{Float32}(
        is_3d=false,
        scales=(4, 2, 1),
        iterations=(100, 50, 25),
        learning_rate=0.01f0,
        with_translation=true,
        with_rotation=true,
        with_zoom=true,
        with_shear=false,
        array_type=array_type
    )

    moved_hu = register(reg_hu, moving_arr, static_arr;
                        verbose=true, final_interpolation=:nearest)

    println("\n   HU-preserving registration complete.")
    print_intensity_stats("HU-preserving (nearest) result", moved_hu, indent="   ")

    # Check HU preservation
    is_preserved_hu, new_vals_hu = check_hu_preservation(moving_arr, moved_hu)
    println("\n   HU Preserved: $(is_preserved_hu)")
    println("   New values created: $(length(new_vals_hu))")

    # -------------------------------------------------------------------------
    # 7. Quantitative Comparison
    # -------------------------------------------------------------------------
    println("\n7. Quantitative Comparison:")
    println("   " * "-" ^ 60)

    standard_unique = length(unique(vec(to_cpu(moved_standard))))
    hu_unique = length(unique(vec(to_cpu(moved_hu))))
    moving_unique = length(unique(vec(to_cpu(moving_arr))))

    println("   Unique values in moving image: $(moving_unique)")
    println("   Unique values in standard result: $(standard_unique)")
    println("   Unique values in HU-preserving result: $(hu_unique)")
    println()

    # Calculate value drift for standard
    moving_cpu = vec(to_cpu(moving_arr))
    standard_cpu = vec(to_cpu(moved_standard))
    hu_cpu = vec(to_cpu(moved_hu))

    println("   Standard (bilinear):")
    println("     - Creates $(standard_unique - moving_unique) new intensity values")
    println("     - Mean value: $(round(sum(standard_cpu)/length(standard_cpu), digits=4))")

    println("   HU-Preserving (nearest):")
    println("     - Creates $(max(0, hu_unique - moving_unique)) new intensity values")
    println("     - Mean value: $(round(sum(hu_cpu)/length(hu_cpu), digits=4))")
    println("     - All output values exist in input: $(is_preserved_hu)")

    println("   " * "-" ^ 60)

    # -------------------------------------------------------------------------
    # 8. Save Results and Create GIFs
    # -------------------------------------------------------------------------
    println("\n8. Saving output images...")

    moved_standard_img = array_to_image(moved_standard)
    moved_hu_img = array_to_image(moved_hu)

    save(joinpath(output_dir, "hu_standard_result.png"), moved_standard_img)
    save(joinpath(output_dir, "hu_preserving_result.png"), moved_hu_img)
    save(joinpath(output_dir, "hu_standard_overlay.png"), checkerboard_overlay(static_img, moved_standard_img))
    save(joinpath(output_dir, "hu_preserving_overlay.png"), checkerboard_overlay(static_img, moved_hu_img))

    # Create standard registration GIF
    println("\n9. Creating animation GIFs...")

    frames_standard = []
    frames_hu = []

    # Frame sequence: static -> moving -> registered -> overlay
    for _ in 1:10
        push!(frames_standard, RGB.(static_img))
        push!(frames_hu, RGB.(static_img))
    end
    for _ in 1:10
        push!(frames_standard, RGB.(moving_img))
        push!(frames_hu, RGB.(moving_img))
    end
    for i in 1:10
        alpha = i / 10
        push!(frames_standard, RGB.(blend_images(moved_standard_img, moving_img, Float32(alpha))))
        push!(frames_hu, RGB.(blend_images(moved_hu_img, moving_img, Float32(alpha))))
    end
    for _ in 1:10
        push!(frames_standard, RGB.(moved_standard_img))
        push!(frames_hu, RGB.(moved_hu_img))
    end
    for _ in 1:10
        push!(frames_standard, RGB.(checkerboard_overlay(static_img, moved_standard_img)))
        push!(frames_hu, RGB.(checkerboard_overlay(static_img, moved_hu_img)))
    end

    save(joinpath(output_dir, "registration_standard.gif"), cat(frames_standard..., dims=3); fps=10)
    save(joinpath(output_dir, "registration_hu_preserving.gif"), cat(frames_hu..., dims=3); fps=10)
    println("   Saved: registration_standard.gif")
    println("   Saved: registration_hu_preserving.gif")

    # -------------------------------------------------------------------------
    # 10. Print Histogram Comparison (ASCII)
    # -------------------------------------------------------------------------
    println("\n10. Intensity Distribution Comparison:")

    println("\n   Moving Image (Input):")
    println(histogram(moving_cpu; nbins=20, vertical=true, height=10))

    println("\n   Standard Result (Bilinear - creates new values):")
    println(histogram(standard_cpu; nbins=20, vertical=true, height=10))

    println("\n   HU-Preserving Result (Nearest - exact input values):")
    println(histogram(hu_cpu; nbins=20, vertical=true, height=10))

    # -------------------------------------------------------------------------
    # 11. Summary
    # -------------------------------------------------------------------------
    println("\n" * "=" ^ 70)
    println("SUMMARY: Standard vs HU-Preserving Registration")
    println("=" ^ 70)
    println()
    println("Standard (bilinear) interpolation:")
    println("  - Smooth visual result")
    println("  - Creates $(standard_unique - moving_unique) NEW interpolated values")
    println("  - NOT suitable for quantitative CT analysis")
    println()
    println("HU-Preserving (hybrid) mode:")
    println("  - Uses bilinear during optimization (smooth gradients)")
    println("  - Uses nearest-neighbor for final output (exact values)")
    println("  - Creates $(max(0, hu_unique - moving_unique)) new values")
    println("  - Output values are EXACT subset of input: $(is_preserved_hu)")
    println("  - RECOMMENDED for CT dose calculation, quantitative analysis")
    println()
    println("Usage:")
    println("  # Standard registration")
    println("  moved = register(reg, moving, static)")
    println()
    println("  # HU-preserving registration")
    println("  moved = register(reg, moving, static; final_interpolation=:nearest)")
    println()
    println("Output files saved to: $(output_dir)")
    println("=" ^ 70)

    return moved_standard, moved_hu, reg_standard, reg_hu
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    run_hu_preservation_demo()
end
