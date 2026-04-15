
import struct
import numpy as np
import cv2
import open3d as o3d
from typing import Tuple
import math
from GenerateData.utils.camera_utils import CameraSample, CameraIntrinsics
import argparse
from pathlib import Path
import imageio.v2 as imageio
from typing import List, Optional
from utils.io_utils import _load_texture_image, _find_texture_file
import open3d.visualization.rendering as rendering
from scene.colmap_loader import rotmat2qvec
from PIL import Image
import os
import sys
from GenerateData.utils.lighting import get_lighting_config

def _sample_points(mesh: o3d.geometry.TriangleMesh, thresh: Optional[float], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample exact mesh vertices using radius-based thinning for uniform distribution.
    
    Uses greedy Poisson disk sampling: iteratively selects vertices ensuring 
    no two selected points are closer than the specified radius threshold.
    If thresh is None, returns all mesh vertices without downsampling.
    
    Args:
        mesh: Triangle mesh to sample from
        thresh: Minimum distance between sampled points (radius threshold), or None for no downsampling
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (vertices, colors, normals)
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    
    # If no threshold specified, return all vertices
    if thresh is None:
        sampled_vertices = vertices
        sampled_colors = colors
        sampled_normals = normals
    else:
        import sklearn.neighbors as skln
        
        # Random shuffle vertices
        rng = np.random.default_rng(seed)
        shuffle_order = np.arange(len(vertices))
        rng.shuffle(shuffle_order)
        vertices_shuffled = vertices[shuffle_order]
        
        # Build KD-tree and find neighbors within radius
        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(vertices_shuffled)
        rnn_idxs = nn_engine.radius_neighbors(vertices_shuffled, radius=thresh, return_distance=False)
        
        # Greedy selection: keep point if not already masked, mask its neighbors
        mask = np.ones(len(vertices_shuffled), dtype=bool)
        for curr, neighbor_idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[neighbor_idxs] = False
                mask[curr] = True
        
        # Map back to original indices
        selected_shuffled = np.where(mask)[0]
        selected_indices = shuffle_order[selected_shuffled]
        
        sampled_vertices = vertices[selected_indices]
        sampled_colors = colors[selected_indices]
        sampled_normals = normals[selected_indices]
    
    # Convert to uint8 [0-255] range
    if sampled_colors.max() <= 1.0:
        sampled_colors = (sampled_colors * 255.0).astype(np.uint8)
    else:
        sampled_colors = sampled_colors.astype(np.uint8)
    
    return sampled_vertices, sampled_colors, sampled_normals


def _render_images(mesh: o3d.geometry.TriangleMesh, centers: np.ndarray, targets: np.ndarray, args: argparse.Namespace, output_dir: Path, texture_folder: str, texture_name: str) -> List[CameraSample]:
    """Render images using Open3D's offscreen renderer.

    Uses Open3D's EGL-based headless rendering for GPU-accelerated image generation.
    """
    print("🎨 Using Open3D for headless rendering...")
    use_decoupled_appearance = getattr(args, 'use_decoupled_appearance', False)
    light_id = getattr(args, 'light_id', None)
    return _render_images_open3d(mesh, centers, targets, args, output_dir, texture_folder, texture_name, use_decoupled_appearance, light_id)


def _render_images_open3d(mesh: o3d.geometry.TriangleMesh, centers: np.ndarray, targets: np.ndarray, args: argparse.Namespace, output_dir: Path, texture_folder: str, texture_name: str, use_decoupled_appearance: bool = False, light_id: int = None) -> List[CameraSample]:
    """Render images using Open3D renderer.

    The camera sampling (centers/targets) is preserved from the original
    implementation; rendering is done directly with Open3D mesh.
    
    Args:
        use_decoupled_appearance: If True, varies lighting across view groups to simulate
                                 appearance variations for training appearance networks.
        light_id: If specified (0-4) and use_decoupled_appearance=False, uses one of 5 fixed
                 lighting configurations for consistent but varied lighting setups.
    """
    '''

    def fov2focal(fov, pixels):
        return pixels / (2 * math.tan(fov / 2))

    def focal2fov(focal, pixels):
        return 2*math.atan(pixels/(2*focal))
    '''
    width = args.image_width
    height = args.image_height
    yfov = math.radians(args.vertical_fov)
    fy = (height / 2.0) / math.tan(yfov / 2.0)
    fx = fy * (width / height)
    cx = width / 2.0
    cy = height / 2.0

    samples: List[CameraSample] = []
    camera_intrinsics: List[CameraIntrinsics] = []

    # Load texture if it exists
    texture_folder_path = Path(__file__).parent / texture_folder
    texture_path = _find_texture_file(texture_folder_path, texture_name)
    has_texture = texture_path is not None
    
    # Create renderer once outside the loop
    renderer = rendering.OffscreenRenderer(width, height)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"

   
    
    # Apply texture if available
    if has_texture:
        print(f"Loading texture from {texture_path}")
        texture_image = _load_texture_image(texture_path)
        material.albedo_img = texture_image
        material.shader = "defaultLit"
    material.base_metallic = 0.0  # Non-metallic for better diffuse shading
    material.base_roughness = 0.95  # Very rough/matte for clear geometry shading
    material.base_reflectance = 0.0 # No specular reflections
        #print(f"Texture loaded: {texture_image.width}x{texture_image.height}")
    
    # Add geometry once
    renderer.scene.add_geometry("mesh", mesh, material)
    
    # Set camera intrinsics once
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    renderer.setup_camera(intrinsic, np.eye(4))
    
    # Debug mesh before rendering
    print(f"Mesh has {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Mesh has colors: {mesh.has_vertex_colors()}")
    if mesh.has_vertex_colors():
        colors_arr = np.asarray(mesh.vertex_colors)
        print(f"Color range: [{colors_arr.min():.4f}, {colors_arr.max():.4f}]")
    
    # Set dark background for clear foreground/background separation
    renderer.scene.set_background([0, 0, 0, 0])
    
    # Track light setup counter for unique naming
    light_setup_counter = [0]  # Use list to allow mutation in nested function
    
    # LIGHTING SETUP FUNCTION
    def setup_lighting_for_group(group_id: int):
        """Setup lighting configuration for a specific view group.
        
        Args:
            group_id: Lighting group ID
                     -1 = default/standard lighting
                     0-4 = fixed light ID presets (when not use_decoupled_appearance)
                     0-9 = appearance variation groups (when use_decoupled_appearance)
        """
        # Clear existing lights
        renderer.scene.scene.enable_sun_light(False)
        
        # Get lighting configuration from lighting module
        light_configs, indirect_intensity = get_lighting_config(group_id, use_decoupled_appearance)
        
        # Use unique light names to avoid "already been added" warnings
        # Open3D doesn't support removing lights, so we use unique names each time
        counter = light_setup_counter[0]
        light_setup_counter[0] += 1
        
        # Apply all lights from configuration
        for name_suffix, color, direction, intensity in light_configs:
            unique_name = f"{name_suffix}_{counter}"
            renderer.scene.scene.add_directional_light(
                unique_name, color, direction, intensity, False
            )
        
        # Enable indirect lighting
        try:
            renderer.scene.scene.enable_indirect_light(True)
            renderer.scene.scene.set_indirect_light_intensity(indirect_intensity)
            if use_decoupled_appearance:
                print(f"  Indirect lighting: {indirect_intensity}")
            else:
                print(f"Indirect lighting enabled with intensity {indirect_intensity}")
        except Exception as e:
            print(f"Warning: Could not enable indirect lighting: {e}")
    
    # Calculate number of views per group if using appearance variation
    num_views = len(centers)
    if use_decoupled_appearance:
        # Split views into 10 groups for appearance diversity
        num_groups = 10
        views_per_group = num_views // num_groups
        
        # Create random shuffled indices for view assignment
        # This ensures lighting variation is distributed randomly across viewpoints
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        view_indices = np.arange(num_views)
        rng.shuffle(view_indices)
        
        # Create mapping: shuffled_index -> lighting_group
        view_to_group = {}
        for i, view_idx in enumerate(view_indices):
            group_id = i // views_per_group
            group_id = min(group_id, num_groups - 1)  # Clamp to valid range
            view_to_group[view_idx] = group_id
        
        print(f"\n🎨 APPEARANCE VARIATION MODE: {num_groups} lighting groups, ~{views_per_group} views per group")
        print(f"   Views randomly shuffled with seed=42 for diverse appearance training")
    elif light_id is not None:
        # FIXED LIGHTING ID MODE: Use one of 5 preset lighting configurations
        # This allows generating datasets with different but consistent lighting
        num_groups = 1
        view_to_group = None
        if light_id < 0 or light_id > 5:
            print(f"⚠️  Warning: light_id={light_id} out of range [0-5], clamping to valid range")
            light_id = max(0, min(5, light_id))
        print(f"\n🎨 FIXED LIGHTING MODE: Using lighting preset {light_id} (0=default balanced)")
    else:
        # Single lighting setup for all views
        num_groups = 1
        view_to_group = None
        print("\n🎨 STANDARD MODE: Consistent lighting for all views")
    
    # Initialize with appropriate lighting setup
    if use_decoupled_appearance:
        setup_lighting_for_group(0)
        last_group = 0
    elif light_id is not None:
        # Map light_id to specific lighting configuration
        # light_id 0 = default balanced (same as standard mode)
        # light_id 1-4 = variations of the default
        setup_lighting_for_group(light_id if light_id > 0 else -1)  # -1 for default
        last_group = None
    else:
        setup_lighting_for_group(-1)  # Default lighting
        last_group = None

    for idx, (eye, target) in enumerate(zip(centers, targets), start=1):
        # Update lighting if using appearance variation
        if use_decoupled_appearance:
            # Get the lighting group for this view (using shuffled assignment)
            current_group = view_to_group[idx - 1]
            
            # Change lighting when entering a new group
            if current_group != last_group:
                # Clear and reset lighting for new group
                # Note: Open3D doesn't have remove_light, so we just set new lights
                # They will override the previous configuration
                setup_lighting_for_group(current_group)
                last_group = current_group
        
        # Set the camera pose in the renderer using the c2w matrix
        renderer.scene.camera.look_at(target, eye, [0,1,0])

        # Get c2w camera ematrix from the renderer
        c2w = np.linalg.inv(renderer.scene.camera.get_view_matrix())

       
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
    
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        #R_w2c = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        R_w2c = w2c[:3,:3]
        tvec = w2c[:3, 3]


        
        # Transform mesh vertices to camera coordinates and check minimum z value
        vertices_world = np.asarray(mesh.vertices)
        vertices_camera = (R_w2c @ vertices_world.T).T + tvec
        min_z = vertices_camera[:, 2].min()
        max_z = vertices_camera[:, 2].max()
        print(f"Camera {idx}: Z range = [{min_z:.4f}, {max_z:.4f}]")
        assert min_z >= 0.2, f"Camera {idx}: Minimum z value ({min_z:.4f}) is less than 0.2 - mesh too close to camera!"
        
        # Render image
        img = renderer.render_to_image()
        img_np = np.array(img, dtype=np.float32)
        
        # Foreground detection: depth buffer + dilation for edge/shadow pixels.
        # The depth buffer gives clean interior detection but misses anti-aliased
        # edge pixels and shadow fringe at silhouettes. Dilating the depth mask
        # by a few pixels captures those, then we intersect with a permissive
        # color threshold to avoid extending into true background.
        depth = np.asarray(renderer.render_to_depth_image(), dtype=np.float32)
        depth_fg = (depth < 1.0).astype(np.uint8)
        # Dilate depth mask by 2 pixels to cover edge/shadow fringe
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #dilated_fg = cv2.dilate(depth_fg, kernel, iterations=1)
        # Only keep dilated pixels that actually have some color (not pure black bg)
        #has_color = np.sum(img_np , axis=2)>60
        is_foreground = (depth_fg > 0)# & has_color
        
        # CRITICAL: Apply minimum brightness to foreground pixels to prevent completely black regions
        # This ensures all surface areas are visible for Gaussian splat training
        min_brightness_255 = 20.0  # Minimum brightness level in 0-255 range (8% of 255 = ~20)
        
        # For foreground pixels, ensure minimum brightness while preserving color ratios
        if is_foreground.any():
            # Get foreground pixels
            foreground_rgb = img_np[is_foreground]
            
            # Calculate luminance (in 0-255 range)
            luminance = 0.299 * foreground_rgb[:, 0] + 0.587 * foreground_rgb[:, 1] + 0.114 * foreground_rgb[:, 2]
            
            # Find pixels darker than minimum
            too_dark = luminance < min_brightness_255
            num_dark = too_dark.sum()
            
            if num_dark > 0:
                if idx == 1:  # Debug first image
                    print(f"  [BRIGHTNESS CLAMPING] Found {num_dark} dark pixels (below {min_brightness_255:.0f}/255)")
                    print(f"    Luminance range before: [{luminance[too_dark].min():.2f}, {luminance[too_dark].max():.2f}]")
                
                # Boost dark pixels to minimum brightness while preserving color ratios
                dark_pixels = foreground_rgb[too_dark]
                dark_lum = luminance[too_dark]
                
                # For pixels with some brightness, scale proportionally
                has_brightness = dark_lum > 0.5  # Small threshold in 0-255 range
                
                #if has_brightness.any():
                #    boost_factors = min_brightness_255 / dark_lum[has_brightness]
                #    dark_pixels[has_brightness] = np.clip(
                #        dark_pixels[has_brightness] * boost_factors[:, np.newaxis],
                #        0, 255
                #    )
                
                # For completely black pixels, set to uniform gray
                completely_black = ~has_brightness
                if completely_black.any():
                    dark_pixels[completely_black] = 0 / 3  # Distribute across RGB
                
                # Write back to image
                foreground_rgb[too_dark] = dark_pixels
                img_np[is_foreground] = foreground_rgb
                
                if idx == 1:  # Debug verification
                    new_lum = 0.299 * foreground_rgb[too_dark, 0] + 0.587 * foreground_rgb[too_dark, 1] + 0.114 * foreground_rgb[too_dark, 2]
                    print(f"    Luminance range after:  [{new_lum.min():.2f}, {new_lum.max():.2f}]")
        
        # Create ground truth alpha mask
        alpha_mask = is_foreground.astype(np.uint8) * 255
        
        # Convert to uint8 for image saving
        img_np_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Create RGBA image by adding alpha channel
        img_rgba = np.dstack([img_np_uint8, alpha_mask])
        
        # Debug first image
        if idx == 1:
            print(f"First image stats: shape={img_np.shape}, range=[{img_np.min():.4f}, {img_np.max():.4f}], mean={img_np.mean():.4f}")
            print(f"Alpha mask coverage: {is_foreground.mean()*100:.2f}% foreground")
            if img_np.mean() > 0.9 or img_np.mean() < 0.1:
                print("WARNING: Image appears blank/uniform - mesh may not be visible!")
        
        # Save as PNG with alpha channel to preserve the mask
        image_name = f"view_{idx:03d}.png"
        imageio.imwrite(str(output_dir / "images" / image_name), img_rgba)

        camera_intrinsics.append(
            CameraIntrinsics(
                camera_id=idx,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )

        qvec = rotmat2qvec(R_w2c)
        samples.append(CameraSample(image_id=idx, camera_id=idx, image_name=image_name, qvec=qvec, tvec=tvec))

    return samples, camera_intrinsics


def _generate_uv_coordinates(vertices: np.ndarray) -> np.ndarray:
    """Generate UV coordinates for vertices based on x,y positions.
    
    Maps x,y coordinates to [0,1] range for texture mapping.
    
    Args:
        vertices: (N, 3) array of vertex positions
    
    Returns:
        uv_coords: (N, 2) array of UV coordinates in [0, 1]
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Normalize x and y to [0, 1] range
    u = (x - x.min()) / (np.ptp(x) + 1e-8)
    v = (y - y.min()) / (np.ptp(y) + 1e-8)
    return np.column_stack([u, v])


def _sample_colors_from_texture(texture_path: Path, uv_coords: np.ndarray) -> np.ndarray:
    """Sample RGB colors from texture image at given UV coordinates.
    
    Args:
        texture_path: Path to texture image file
        uv_coords: (N, 2) array of UV coordinates in [0, 1] range
    
    Returns:
        colors: (N, 3) array of RGB colors in [0, 1] range
    """
    # Load texture image
    
    pil_image = Image.open(texture_path).convert('RGB')
    img_array = np.array(pil_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    height, width = img_array.shape[:2]
    
    # Convert UV coordinates to pixel coordinates
    # UV (0,0) is typically bottom-left, but image (0,0) is top-left
    # So we flip V coordinate
    u = np.clip(uv_coords[:, 0], 0, 1)
    v = np.clip(1.0 - uv_coords[:, 1], 0, 1)  # Flip V
    
    # Convert to pixel indices
    x_pixels = (u * (width - 1)).astype(np.int32)
    y_pixels = (v * (height - 1)).astype(np.int32)
    
    # Sample colors from texture
    colors = img_array[y_pixels, x_pixels, :]
    
    return colors
