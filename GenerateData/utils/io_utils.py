
from collections.abc import Sequence
import struct
from pathlib import Path
from tkinter import Image
from typing import Sequence, Optional
import numpy as np
import open3d as o3d
from PIL import Image

from scene.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
)


from GenerateData.utils.camera_utils import CameraSample, CameraIntrinsics




def _write_cameras_txt(path: Path, cameras: Sequence[CameraIntrinsics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam in cameras:
            f.write(
                f"{cam.camera_id} PINHOLE {cam.width} {cam.height} "
                f"{cam.fx:.6f} {cam.fy:.6f} {cam.cx:.6f} {cam.cy:.6f}\n"
            )

def _write_cameras_bin(path: Path, cameras: Sequence[CameraIntrinsics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_id = 1  # PINHOLE
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<iiQQ", cam.camera_id, model_id, cam.width, cam.height))
            params = [float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)]
            f.write(struct.pack("<" + "d" * len(params), *params))


def _write_images_txt(path: Path, samples: Sequence[CameraSample]) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for s in samples:
            f.write(
                f"{s.image_id} {s.qvec[0]:.8f} {s.qvec[1]:.8f} {s.qvec[2]:.8f} {s.qvec[3]:.8f} "
                f"{s.tvec[0]:.8f} {s.tvec[1]:.8f} {s.tvec[2]:.8f} {s.camera_id} {s.image_name}\n"
            )
            f.write("\n")


def _write_images_bin(path: Path, samples: Sequence[CameraSample]) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(samples)))
        for s in samples:
            f.write(
                struct.pack(
                    "<idddddddi",
                    s.image_id,
                    float(s.qvec[0]),
                    float(s.qvec[1]),
                    float(s.qvec[2]),
                    float(s.qvec[3]),
                    float(s.tvec[0]),
                    float(s.tvec[1]),
                    float(s.tvec[2]),
                    s.camera_id,
                )
            )
            f.write(s.image_name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num points2D


def _write_points3d_txt(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for idx, (v, c) in enumerate(zip(vertices, colors), start=1):
            f.write(f"{idx} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")


def _write_points3d_bin(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(vertices)))
        for idx, (v, c) in enumerate(zip(vertices, colors), start=1):
            f.write(struct.pack("<QdddBBBd", idx, v[0], v[1], v[2], int(c[0]), int(c[1]), int(c[2]), 1.0))
            f.write(struct.pack("<Q", 0))  # track length



def _write_points3d_ply(path: Path, vertices: np.ndarray, colors: np.ndarray, normals: Optional[np.ndarray] = None) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        "comment synthetic polynomial dataset",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if normals is not None:
            for v, n, c in zip(vertices, normals, colors):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            # Fallback to zero normals if not provided
            for v, c in zip(vertices, colors):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 0.0 0.0 0.0 {int(c[0])} {int(c[1])} {int(c[2])}\n")

def _find_texture_file(texture_folder: Path, texture_name: str) -> Optional[Path]:
    """Find texture file with given name, searching for common image extensions.
    
    Args:
        texture_folder: Path to folder containing textures
        texture_name: Texture filename with or without extension
    
    Returns:
        Path to texture file if found, None otherwise
    """
    # If texture_name already has an extension, try it first
    if '.' in texture_name:
        candidate = texture_folder / texture_name
        if candidate.exists():
            return candidate
    
    # Try common image extensions
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG']
    for ext in extensions:
        candidate = texture_folder / f"{texture_name}{ext}"
        if candidate.exists():
            return candidate
    
    return None


def _load_texture_image(texture_path: Path) -> o3d.geometry.Image:
    """Load texture image and convert to Open3D format.
    
    Args:
        texture_path: Path to texture image file
    
    Returns:
        o3d_image: Open3D Image object
    """
    # Load with PIL
    pil_image = Image.open(texture_path).convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(pil_image, dtype=np.uint8)
    
    # Convert to Open3D Image
    o3d_image = o3d.geometry.Image(img_array)
    
    return o3d_image


def _ensure_dirs(root: Path) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "sparse" / "0").mkdir(parents=True, exist_ok=True)


def _validate_dataset(dataset_dir: Path) -> None:
    sparse_dir = dataset_dir / "sparse" / "0"
    try:
        cams = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
        xyzs, rgbs, _ = read_points3D_binary(str(sparse_dir / "points3D.bin"))

        print(
            "Validation via colmap_loader:",
            f"{len(cams)} camera(s), {len(images)} image(s), {len(xyzs)} points"
        )
    except Exception as exc:
        print("Validation warning (colmap_loader parsing failed):", exc)


def _create_training_script(dataset_dir: Path, use_decoupled_appearance: bool = False) -> None:
    """Create a train_and_extract_mesh.sh script in the dataset directory.
    
    Args:
        dataset_dir: Path to the dataset directory where script will be created
        use_decoupled_appearance: If True, adds --use_decoupled_appearance flag to training command
    """
    script_path = dataset_dir / "train_and_extract_mesh.sh"
    
    # Build training command with optional appearance flag
    appearance_flag = " --use_decoupled_appearance" if use_decoupled_appearance else ""
    
    script_content = f"""#!/bin/bash

# Auto-generated training and mesh extraction script
# Dataset: {dataset_dir}

DATASET_DIR="{dataset_dir}"
OUTPUT_DIR="${{DATASET_DIR}}/output"

echo "=========================================="
echo "Training Gaussian Splatting Model"
echo "=========================================="
echo "Dataset: ${{DATASET_DIR}}"
echo "Output: ${{OUTPUT_DIR}}"
echo ""

python train.py -s "${{DATASET_DIR}}" -m "${{OUTPUT_DIR}}" --eval{appearance_flag}

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Training failed! Exiting..."
    exit 1
fi

echo ""
echo "=========================================="
echo "Extracting Mesh with Tetrahedra"
echo "=========================================="
echo ""

python mesh_extract_tetrahedra.py -s "${{DATASET_DIR}}" -m "${{OUTPUT_DIR}}" --eval

# Check if mesh extraction was successful
if [ $? -ne 0 ]; then
    echo "Mesh extraction failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results saved to: ${{OUTPUT_DIR}}"
echo "=========================================="
"""
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # Make script executable
    import stat
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    
    print(f"Created training script: {script_path}")

