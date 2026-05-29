"""
Preprocess Synthetic Chess Images for CUT Training

This script:
1. Collects all synthetic images from synthetic_gameX/cropped folders
2. Copies original images (white viewpoint)
3. Creates 180° rotated versions (black viewpoint)
4. Deletes comparison.png files (redundant)
5. Organizes them into a training-ready structure

Output structure:
    dataset/
    └── synthetic/
        ├── game2_frame_000200_middle_white.png
        ├── game2_frame_000200_middle_black.png  (180° rotated)
        ├── game2_frame_000200_left_white.png
        ├── game2_frame_000200_left_black.png
        └── ...

Usage:
    python preprocess_synthetic_images.py
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm


def rotate_180(image):
    """Rotate image 180 degrees"""
    return cv2.rotate(image, cv2.ROTATE_180)


def delete_comparison_files(project_path: Path, games: list) -> int:
    """Delete all comparison.png files from synthetic folders"""
    deleted_count = 0

    print("Deleting comparison files...")
    for game_num in games:
        cropped_folder = project_path / f"synthetic_game{game_num}" / "cropped"
        if cropped_folder.exists():
            comparison_files = list(cropped_folder.glob("*_comparison.png"))
            for comp_file in comparison_files:
                try:
                    comp_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {comp_file}: {e}")

    print(f"Deleted {deleted_count} comparison files")
    return deleted_count


def preprocess_synthetic_images(
    project_dir: str,
    output_dir: str = None,
    games: list = [2, 4, 5, 6, 7],
    delete_comparisons: bool = True
):
    """
    Preprocess synthetic chess images for CUT training.

    Args:
        project_dir: Path to the project directory
        output_dir: Path for output (default: project_dir/dataset/synthetic)
        games: List of game numbers to process
        delete_comparisons: Whether to delete comparison.png files
    """
    project_path = Path(project_dir)

    if output_dir is None:
        output_path = project_path / "dataset" / "synthetic"
    else:
        output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preprocessing Synthetic Chess Images")
    print("=" * 60)
    print(f"Project directory: {project_path}")
    print(f"Output directory: {output_path}")
    print(f"Games to process: {games}")
    print()

    # Delete comparison files first if requested
    if delete_comparisons:
        delete_comparison_files(project_path, games)
        print()

    # Collect all synthetic image paths (left, middle, right only)
    all_images = []
    view_types = ["left", "middle", "right"]

    for game_num in games:
        cropped_folder = project_path / f"synthetic_game{game_num}" / "cropped"
        if cropped_folder.exists():
            for view_type in view_types:
                images = list(cropped_folder.glob(f"*_{view_type}.png"))
                all_images.extend([(game_num, view_type, img) for img in images])

            # Count unique frames
            middle_count = len(list(cropped_folder.glob("*_middle.png")))
            print(f"Game {game_num}: Found {middle_count} positions x 3 views = {middle_count * 3} images")
        else:
            print(f"Game {game_num}: Folder not found at {cropped_folder}")

    print(f"\nTotal synthetic images found: {len(all_images)}")
    print(f"After rotation: {len(all_images) * 2} images")
    print()

    # Process images
    stats = {
        "white_saved": 0,
        "black_saved": 0,
        "errors": 0
    }

    print("Processing images...")
    for game_num, view_type, img_path in tqdm(all_images, desc="Processing"):
        try:
            # Read image
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"\nWarning: Could not read {img_path}")
                stats["errors"] += 1
                continue

            # Generate output filenames
            frame_name = img_path.stem.replace(f"_{view_type}", "")  # e.g., "frame_000200"
            white_filename = f"game{game_num}_{frame_name}_{view_type}_white.png"
            black_filename = f"game{game_num}_{frame_name}_{view_type}_black.png"

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
    preprocess_synthetic_images(
        project_dir=str(script_dir),
        games=[2, 4, 5, 6, 7],
        delete_comparisons=True
    )


if __name__ == "__main__":
    main()
