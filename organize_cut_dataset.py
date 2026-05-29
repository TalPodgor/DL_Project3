"""
Organize Dataset for CUT Training

This script organizes the preprocessed real and synthetic images
into the CUT training structure with game-based train/test split.

Split Strategy:
    - Training: Games 4, 5, 6, 7
    - Testing:  Game 2

Output structure:
    dataset/
    ├── trainA/    (synthetic training images)
    ├── trainB/    (real training images)
    ├── testA/     (synthetic test images)
    └── testB/     (real test images)

Usage:
    python organize_cut_dataset.py
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def organize_cut_dataset(
    project_dir: str,
    train_games: list = [4, 5, 6, 7],
    test_games: list = [2],
    copy_files: bool = True  # Set to False to move instead of copy
):
    """
    Organize dataset into CUT folder structure.

    Args:
        project_dir: Path to the project directory
        train_games: List of game numbers for training
        test_games: List of game numbers for testing
        copy_files: If True, copy files; if False, move files
    """
    project_path = Path(project_dir)
    dataset_path = project_path / "dataset"

    # Source directories
    real_dir = dataset_path / "real"
    synthetic_dir = dataset_path / "synthetic"

    # Output directories
    train_a_dir = dataset_path / "trainA"  # Synthetic training
    train_b_dir = dataset_path / "trainB"  # Real training
    test_a_dir = dataset_path / "testA"    # Synthetic test
    test_b_dir = dataset_path / "testB"    # Real test

    print("=" * 60)
    print("Organizing Dataset for CUT Training")
    print("=" * 60)
    print(f"Project directory: {project_path}")
    print(f"Training games: {train_games}")
    print(f"Testing games: {test_games}")
    print(f"Mode: {'Copy' if copy_files else 'Move'}")
    print()

    # Create output directories
    for dir_path in [train_a_dir, train_b_dir, test_a_dir, test_b_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # File operation function
    file_op = shutil.copy2 if copy_files else shutil.move

    stats = {
        "trainA": 0,
        "trainB": 0,
        "testA": 0,
        "testB": 0,
        "errors": 0
    }

    # Process real images
    print("Processing real images...")
    if real_dir.exists():
        real_images = list(real_dir.glob("*.jpg"))
        for img_path in tqdm(real_images, desc="Real images"):
            try:
                # Extract game number from filename (e.g., "game2_frame_000200_white.jpg")
                filename = img_path.name
                game_num = int(filename.split("_")[0].replace("game", ""))

                if game_num in train_games:
                    file_op(str(img_path), str(train_b_dir / filename))
                    stats["trainB"] += 1
                elif game_num in test_games:
                    file_op(str(img_path), str(test_b_dir / filename))
                    stats["testB"] += 1
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                stats["errors"] += 1
    else:
        print(f"Warning: Real directory not found at {real_dir}")

    # Process synthetic images
    print("\nProcessing synthetic images...")
    if synthetic_dir.exists():
        synthetic_images = list(synthetic_dir.glob("*.png"))
        for img_path in tqdm(synthetic_images, desc="Synthetic images"):
            try:
                # Extract game number from filename (e.g., "game2_frame_000200_middle_white.png")
                filename = img_path.name
                game_num = int(filename.split("_")[0].replace("game", ""))

                if game_num in train_games:
                    file_op(str(img_path), str(train_a_dir / filename))
                    stats["trainA"] += 1
                elif game_num in test_games:
                    file_op(str(img_path), str(test_a_dir / filename))
                    stats["testA"] += 1
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                stats["errors"] += 1
    else:
        print(f"Warning: Synthetic directory not found at {synthetic_dir}")

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Training split (Games {train_games}):")
    print(f"  trainA (synthetic): {stats['trainA']} images")
    print(f"  trainB (real):      {stats['trainB']} images")
    print()
    print(f"Testing split (Games {test_games}):")
    print(f"  testA (synthetic):  {stats['testA']} images")
    print(f"  testB (real):       {stats['testB']} images")
    print()
    print(f"Total images organized: {sum(stats.values()) - stats['errors']}")
    print(f"Errors: {stats['errors']}")
    print()
    print("Output structure:")
    print(f"  {train_a_dir}")
    print(f"  {train_b_dir}")
    print(f"  {test_a_dir}")
    print(f"  {test_b_dir}")
    print("=" * 60)

    return stats


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    # Run organization
    organize_cut_dataset(
        project_dir=str(script_dir),
        train_games=[4, 5, 6, 7],
        test_games=[2],
        copy_files=True  # Copy to preserve originals
    )


if __name__ == "__main__":
    main()
