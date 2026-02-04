# MedicalImageRegistration.jl Demo
#
# This demo shows how to use the package to register 2D images.
# It creates a misaligned image pair, registers them, and generates
# a GIF animation showing the registration process.
#
# GPU Acceleration: If Metal.jl is available (macOS with Apple Silicon),
# the demo automatically runs on GPU for faster registration.

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))
Pkg.add(["TestImages", "Images", "FileIO", "Colors", "Metal"])

using MedicalImageRegistration
using TestImages
using Images
using FileIO
using Colors

# Try to load Metal for GPU acceleration
const USE_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

if USE_GPU
    println("Metal GPU detected - running on GPU")
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
Convert array from GPU back to CPU for visualization.
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
TestImages returns Matrix{Gray{N0f8}}, we need Array{Float32, 4} with shape (X, Y, C, N).
"""
function image_to_array(img)
    # Convert to Float32
    img_float = Float32.(Gray.(img))
    # Reshape from (Y, X) to (X, Y, 1, 1) - note the transpose for Julia convention
    return permutedims(reshape(img_float, size(img_float)..., 1, 1), (2, 1, 3, 4))
end

"""
Convert registration array back to displayable image.
Array shape: (X, Y, C, N) -> Image shape: (Y, X)
Automatically handles GPU arrays by converting to CPU first.
"""
function array_to_image(arr)
    # Ensure we're on CPU for visualization
    arr_cpu = to_cpu(arr)
    # Extract 2D data and transpose back
    img_data = permutedims(arr_cpu[:, :, 1, 1], (2, 1))
    return Gray.(clamp.(img_data, 0f0, 1f0))
end

"""
Create a transformed version of an image using our affine_transform function.
GPU-compatible: parameters are created on CPU then transferred to device.
"""
function apply_synthetic_transform(img_arr; translation=(0f0, 0f0), rotation_deg=0f0, zoom=(1f0, 1f0))
    T = Float32
    ndim = 2
    batch_size = 1

    # Build affine parameters on CPU first
    trans_cpu = T[translation[1]; translation[2];;]  # (2, 1)

    # Build rotation matrix (2D rotation around center)
    θ = T(rotation_deg * π / 180)
    c, s = cos(θ), sin(θ)
    rot_cpu = reshape(T[c -s; s c], 2, 2, 1)  # (2, 2, 1)

    # Zoom
    z_cpu = T[zoom[1]; zoom[2];;]  # (2, 1)

    # Shear (none)
    shear_cpu = zeros(T, 2, 1)

    # Transfer to same device as input
    trans = to_device(trans_cpu)
    rot = to_device(rot_cpu)
    z = to_device(z_cpu)
    shear = to_device(shear_cpu)

    # Compose and apply
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

# ============================================================================
# Main Demo
# ============================================================================

function run_demo()
    println("=" ^ 60)
    println("MedicalImageRegistration.jl Demo")
    println("=" ^ 60)

    # Load test image (cameraman is a classic test image)
    println("\n1. Loading test image...")
    img = testimage("cameraman")
    println("   Image size: $(size(img))")

    # Convert to our array format (on GPU if available)
    static_arr_cpu = image_to_array(img)
    static_arr = to_device(static_arr_cpu)
    device_name = USE_GPU ? "GPU (MtlArray)" : "CPU (Array)"
    println("   Array shape: $(size(static_arr)) (X, Y, C, N) on $(device_name)")

    # Create misaligned version (this will be our "moving" image)
    println("\n2. Creating misaligned image...")
    # Apply translation and rotation
    translation = (10f0, -15f0)  # Shift in pixels (normalized to [-1, 1] internally)
    rotation_deg = 5f0           # 5 degree rotation
    zoom = (0.95f0, 1.05f0)      # Slight zoom difference

    # Normalize translation to [-1, 1] range (assuming 256x256 image)
    norm_trans = (translation[1] / 128f0, translation[2] / 128f0)

    moving_arr = apply_synthetic_transform(
        static_arr;
        translation=norm_trans,
        rotation_deg=rotation_deg,
        zoom=zoom
    )
    println("   Applied: translation=$(translation), rotation=$(rotation_deg)°, zoom=$(zoom)")

    # Convert back to images for visualization
    static_img = array_to_image(static_arr)
    moving_img = array_to_image(moving_arr)

    # Save before images
    println("\n3. Saving before images...")
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    save(joinpath(output_dir, "static.png"), static_img)
    save(joinpath(output_dir, "moving_before.png"), moving_img)
    save(joinpath(output_dir, "overlay_before.png"), checkerboard_overlay(static_img, moving_img))
    println("   Saved to: $(output_dir)")

    # Run registration
    println("\n4. Running affine registration...")
    array_type = USE_GPU ? MtlArray : Array
    reg = AffineRegistration{Float32}(
        is_3d=false,  # 2D registration
        scales=(4, 2, 1),
        iterations=(100, 50, 25),
        learning_rate=0.01f0,
        with_translation=true,
        with_rotation=true,
        with_zoom=true,
        with_shear=false,
        array_type=array_type
    )

    moved_arr = register(reg, moving_arr, static_arr; verbose=true)

    # Get learned parameters
    affine_matrix = to_cpu(get_affine(reg))
    println("\n5. Learned affine matrix:")
    display(affine_matrix[:, :, 1])

    # Convert result to image
    moved_img = array_to_image(moved_arr)

    # Save after images
    println("\n6. Saving after images...")
    save(joinpath(output_dir, "moving_after.png"), moved_img)
    save(joinpath(output_dir, "overlay_after.png"), checkerboard_overlay(static_img, moved_img))

    # Create animated GIF showing before/after
    println("\n7. Creating animation GIF...")
    frames = []

    # Frame 1-10: Static image
    for _ in 1:10
        push!(frames, RGB.(static_img))
    end

    # Frame 11-20: Moving image (before registration)
    for _ in 1:10
        push!(frames, RGB.(moving_img))
    end

    # Frame 21-30: Blend transition from moving to registered
    for i in 1:10
        alpha = i / 10
        blended = blend_images(moved_img, moving_img, Float32(alpha))
        push!(frames, RGB.(blended))
    end

    # Frame 31-40: Registered result
    for _ in 1:10
        push!(frames, RGB.(moved_img))
    end

    # Frame 41-50: Overlay comparison
    for _ in 1:10
        push!(frames, RGB.(checkerboard_overlay(static_img, moved_img)))
    end

    # Save GIF
    gif_path = joinpath(output_dir, "registration_demo.gif")
    save(gif_path, cat(frames..., dims=3); fps=10)
    println("   Saved animation: $(gif_path)")

    println("\n" * "=" ^ 60)
    println("Demo complete!")
    println("Output files in: $(output_dir)")
    println("=" ^ 60)

    return moved_arr, reg
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
