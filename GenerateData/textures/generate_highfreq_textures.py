"""Generate high-frequency textures for Gaussian splatting training.

These textures provide strong local color gradients that help densification
by ensuring every 8x8 pixel block has measurable variation in the rendered images.

Usage:
    python GenerateData/textures/generate_highfreq_textures.py

Generates:
    - checkerboard.png: Classic checkerboard with colored squares
    - checkerboard_blue.png: Blue-toned checkerboard (visually similar to blue texture)
"""

import numpy as np
from PIL import Image
from pathlib import Path


def generate_checkerboard(size=512, cell_size=32, color_a=(40, 80, 180), color_b=(180, 200, 240)):
    """Generate a colored checkerboard pattern.
    
    Args:
        size: Image size in pixels (square)
        cell_size: Size of each checker cell in pixels
        color_a: RGB tuple for dark squares
        color_b: RGB tuple for light squares
    
    Returns:
        PIL Image
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((x // cell_size) + (y // cell_size)) % 2 == 0:
                img[y, x] = color_a
            else:
                img[y, x] = color_b
    return Image.fromarray(img)


def generate_multicolor_checker(size=512, cell_size=32):
    """Generate a multi-color checkerboard with 4 alternating colors.
    
    Provides even stronger local gradients than 2-color checkerboard.
    """
    # 4 colors that tile in a 2x2 pattern
    colors = [
        (40, 80, 180),    # dark blue
        (180, 200, 240),  # light blue
        (60, 140, 200),   # medium blue
        (120, 160, 220),  # blue-gray
    ]
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            cx = (x // cell_size) % 2
            cy = (y // cell_size) % 2
            idx = cy * 2 + cx
            img[y, x] = colors[idx]
    return Image.fromarray(img)


def generate_uv_test(size=512):
    """Generate a UV test pattern with gradients, grid lines, and color blocks.
    
    This provides the strongest possible local gradient signal.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Base: smooth gradient
    for y in range(size):
        for x in range(size):
            u = x / size
            v = y / size
            img[y, x] = (
                int(u * 200 + 30),
                int(v * 150 + 50),  
                int((1 - u) * 180 + 40),
            )
    
    # Overlay grid lines every 32 pixels
    grid_color = np.array([255, 255, 255], dtype=np.uint8)
    for i in range(0, size, 32):
        img[i, :] = grid_color
        img[:, i] = grid_color
    
    return Image.fromarray(img)


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    
    # Checkerboard in blue tones (matches existing blue texture aesthetic)
    checker_blue = generate_checkerboard(
        size=512, cell_size=32,
        color_a=(30, 60, 160),
        color_b=(150, 190, 240)
    )
    checker_blue.save(out_dir / "checkerboard_blue.png")
    print(f"Saved {out_dir / 'checkerboard_blue.png'}")
    
    # Multi-color checker
    multi = generate_multicolor_checker(size=512, cell_size=32)
    multi.save(out_dir / "checkerboard_multi.png")
    print(f"Saved {out_dir / 'checkerboard_multi.png'}")
    
    # Generic checkerboard (red/white)
    checker = generate_checkerboard(
        size=512, cell_size=32,
        color_a=(200, 50, 50),
        color_b=(240, 230, 220)
    )
    checker.save(out_dir / "checkerboard.png")
    print(f"Saved {out_dir / 'checkerboard.png'}")
    
    # UV test pattern
    uv = generate_uv_test(size=512)
    uv.save(out_dir / "uv_test.png")
    print(f"Saved {out_dir / 'uv_test.png'}")
    
    print("\nDone! Use with --texture_name checkerboard_blue (or checkerboard, checkerboard_multi, uv_test)")
