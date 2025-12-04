#!/usr/bin/env python3
"""
Download TNT (Tanks and Temples) preprocessed dataset from Hugging Face
Dataset: https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields
"""

import os
import argparse
from pathlib import Path

def print_instructions():
    """Print download instructions for TNT dataset from Hugging Face"""
    
    print("="*80)
    print("TNT Dataset Download Instructions from Hugging Face")
    print("="*80)
    print()
    
    print("Dataset URL: https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields")
    print()
    
    print("="*80)
    print("METHOD 1: Manual Download (Recommended for beginners)")
    print("="*80)
    print()
    print("1. Go to: https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main")
    print()
    print("2. Navigate to the TNT scenes you want:")
    print("   - tandt/train/Barn")
    print("   - tandt/train/Caterpillar")
    print("   - tandt/train/Courthouse")
    print("   - tandt/train/Ignatius")
    print("   - tandt/train/Meetingroom")
    print("   - tandt/train/Truck")
    print()
    print("3. For each scene, download:")
    print("   - The entire folder as a zip, OR")
    print("   - Individual files if you prefer")
    print()
    print("4. Extract to your data directory, e.g.:")
    print("   /your/data/path/TNT/Barn/")
    print("   /your/data/path/TNT/Caterpillar/")
    print("   etc.")
    print()
    
    print("="*80)
    print("METHOD 2: Using Git LFS (For all scenes)")
    print("="*80)
    print()
    print("# Install git-lfs if not already installed:")
    print("sudo apt-get install git-lfs  # Ubuntu/Debian")
    print("# or")
    print("brew install git-lfs          # macOS")
    print()
    print("# Initialize git-lfs")
    print("git lfs install")
    print()
    print("# Clone the entire dataset (WARNING: Large download ~50GB+)")
    print("git clone https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields")
    print()
    print("# The TNT data will be in:")
    print("gaussian-opacity-fields/tandt/train/")
    print()
    
    print("="*80)
    print("METHOD 3: Using Hugging Face CLI (Recommended for automation)")
    print("="*80)
    print()
    print("# Install huggingface-hub")
    print("pip install huggingface-hub[cli]")
    print()
    print("# Download specific scene (example: Barn)")
    print("huggingface-cli download ZehaoYu/gaussian-opacity-fields \\")
    print("    --repo-type dataset \\")
    print("    --include 'tandt/train/Barn/*' \\")
    print("    --local-dir ./tnt_data")
    print()
    print("# Or download all TNT scenes:")
    print("huggingface-cli download ZehaoYu/gaussian-opacity-fields \\")
    print("    --repo-type dataset \\")
    print("    --include 'tandt/train/*' \\")
    print("    --local-dir ./tnt_data")
    print()
    
    print("="*80)
    print("METHOD 4: Using Python Script (wget)")
    print("="*80)
    print()
    print("# Download individual scenes using wget")
    print("# Note: You need to find the direct URLs for each file")
    print()
    print("# Example for downloading Barn scene files:")
    print('BASE_URL="https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/resolve/main"')
    print('SCENE="Barn"')
    print('mkdir -p data/TNT/$SCENE')
    print()
    print('# Download sparse reconstruction')
    print('wget $BASE_URL/tandt/train/$SCENE/sparse/0/cameras.bin -P data/TNT/$SCENE/sparse/0/')
    print('wget $BASE_URL/tandt/train/$SCENE/sparse/0/images.bin -P data/TNT/$SCENE/sparse/0/')
    print('wget $BASE_URL/tandt/train/$SCENE/sparse/0/points3D.bin -P data/TNT/$SCENE/sparse/0/')
    print()
    print('# Download images (you need to list all image files)')
    print('# This is tedious - better use git-lfs or huggingface-cli')
    print()
    
    print("="*80)
    print("Expected Final Structure:")
    print("="*80)
    print()
    print("data/TNT/")
    print("├── Barn/")
    print("│   ├── sparse/0/")
    print("│   │   ├── cameras.bin")
    print("│   │   ├── images.bin")
    print("│   │   └── points3D.bin")
    print("│   └── images/")
    print("│       ├── 000000.jpg")
    print("│       ├── 000001.jpg")
    print("│       └── ...")
    print("├── Caterpillar/")
    print("├── Courthouse/")
    print("├── Ignatius/")
    print("├── Meetingroom/")
    print("└── Truck/")
    print()
    
    print("="*80)
    print("Quick Start Commands")
    print("="*80)
    print()
    print("# Install huggingface-hub (RECOMMENDED)")
    print("pip install huggingface-hub[cli]")
    print()
    print("# Download a single scene (e.g., Barn)")
    print("huggingface-cli download ZehaoYu/gaussian-opacity-fields \\")
    print("    --repo-type dataset \\")
    print("    --include 'tandt/train/Barn/*' \\")
    print("    --local-dir ./tnt_data")
    print()
    print("# Then train with:")
    print("python train.py -s ./tnt_data/tandt/train/Barn -m output/barn -r 2 --eval")
    print()
    
    print("="*80)
    print("Tips:")
    print("="*80)
    print("- Start with ONE scene to test (Barn is smallest)")
    print("- Each scene is several GB (images + sparse reconstruction)")
    print("- Use huggingface-cli for best experience")
    print("- Make sure you have enough disk space (~50GB for all scenes)")
    print()


def download_scene_hf(scene_name, output_dir):
    """
    Download a specific TNT scene using huggingface-hub
    
    Args:
        scene_name: Name of the scene (e.g., 'Barn', 'Truck')
        output_dir: Directory to save the downloaded data
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Downloading {scene_name} from Hugging Face...")
        print(f"Output directory: {output_dir}")
        
        # Download specific scene
        snapshot_download(
            repo_id="ZehaoYu/gaussian-opacity-fields",
            repo_type="dataset",
            allow_patterns=f"tandt/train/{scene_name}/*",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"\n✓ Successfully downloaded {scene_name}!")
        print(f"Location: {output_dir}/tandt/train/{scene_name}")
        
    except ImportError:
        print("ERROR: huggingface_hub not installed!")
        print("Install with: pip install huggingface-hub")
    except Exception as e:
        print(f"ERROR downloading: {e}")
        print("\nTry installing: pip install huggingface-hub")


def main():
    parser = argparse.ArgumentParser(
        description="Download TNT dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck", "all"],
        help="Scene to download (or 'all' for all scenes)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for downloaded data (default: <project_root>/data/tnt)"
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Show detailed download instructions"
    )
    
    args = parser.parse_args()
    
    if args.instructions or not args.scene:
        print_instructions()
        return
    
    # Set default output directory to project data folder
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, "data", "tnt")
        print(f"Using default output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download scene(s)
    scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"] if args.scene == "all" else [args.scene]
    
    for scene in scenes:
        download_scene_hf(scene, args.output_dir)
        print()


if __name__ == "__main__":
    main()
