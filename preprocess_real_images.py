"""
Preprocess Real Chess Images for CUT Training

This script:
1. Collects all real images from games 2, 4, 5, 6, 7
2. Copies original images (white viewpoint)
3. Creates 180° rotated versions (black viewpoint)
4. Organizes them into a training-ready structure

Output structure:
    dataset/
    └── real/
        ├── game2_frame_000200_white.jpg
        ├── game2_frame_000200_black.jpg  (180° rotated)
        └── ...

Usage:
    python preprocess_real_images.py
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm


def rotate_180(image):
    """Rotate image 180 degrees"""
    return cv2.rotate(image, cv2.ROTATE_180)


def preprocess_real_images(
    project_dir: str,
    output_dir: str = None,
    games: list = [2, 4, 5, 6, 7]
):
    """
    Preprocess real chess images for CUT training.

    Args:
        project_dir: Path to the project directory
        output_dir: Path for output (default: project_dir/dataset/real)
        games: List of game numbers to process
    """
    project_path = Path(project_dir)

    if output_dir is None:
        output_path = project_path / "dataset" / "real"
    else:
        output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preprocessing Real Chess Images")
    print("=" * 60)
    print(f"Project directory: {project_path}")
    print(f"Output directory: {output_path}")
    print(f"Games to process: {games}")
    print()

    # Collect all image paths
    all_images = []
    for game_num in games:
        game_folder = project_path / f"game{game_num}_per_frame" / "tagged_images"
        if game_folder.exists():
            images = list(game_folder.glob("*.jpg"))
            all_images.extend([(game_num, img) for img in images])
            print(f"Game {game_num}: Found {len(images)} images")
        else:
            print(f"Game {game_num}: Folder not found at {game_folder}")

    print(f"\nTotal images found: {len(all_images)}")
    print(f"After rotation: {len(all_images) * 2} images")
    print()

    # Process images
    stats = {
        "white_saved": 0,
        "black_saved": 0,
        "errors": 0
    }

    print("Processing images...")
    for game_num, img_path in tqdm(all_images, desc="Processing"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"\nWarning: Could not read {img_path}")
                stats["errors"] += 1
                continue

            # Generate output filenames
            frame_name = img_path.stem  # e.g., "frame_000200"
            white_filename = f"game{game_num}_{frame_name}_white.jpg"
            black_filename = f"game{game_num}_{frame_name}_black.jpg"

            # Save original (white viewpoint)
            white_path = output_path / white_filename
            cv2.imwrite(str(white_path), img)
            stats["white_saved"] += 1

            # Create and save rotated (black viewpoint)
            rotated_img = rotate_180(img)
            black_path = output_path / black_filename
            cv2.imwrite(str(black_path), rotated_img)
            stats["black_saved"] += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            stats["errors"] += 1

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"White viewpoint images saved: {stats['white_saved']}")
    print(f"Black viewpoint images saved: {stats['black_saved']}")
    print(f"Total images saved: {stats['white_saved'] + stats['black_saved']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nOutput directory: {output_path}")
    print("=" * 60)

    return stats


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    # Run preprocessing
    preprocess_real_images(
        project_dir=str(script_dir),
        games=[2, 4, 5, 6, 7]
    )


if __name__ == "__main__":
    main()
