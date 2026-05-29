"""
Post-Processing Script for Perfect Board Cropping
==================================================
Applies perspective transformation to ensure board corners align
perfectly with image corners (no background).

This script reads the raw renders and metadata from synthetic_chess_generator.py
and produces perfectly cropped images suitable for CUT/GAN training.

Usage:
    python postprocess_crop.py --input_dir ./synthetic_output --output_dir ./cropped_output --size 512
"""

import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path


def load_metadata(metadata_path):
    """Load metadata JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def order_corners(corners):
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left.
    This ensures consistent perspective transformation.
    """
    corners = np.array(corners, dtype=np.float32)

    # Sort by y-coordinate (top points first)
    sorted_by_y = corners[np.argsort(corners[:, 1])]

    # Top two points
    top_points = sorted_by_y[:2]
    # Bottom two points
    bottom_points = sorted_by_y[2:]

    # Sort top points by x (left first)
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x (left first)
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def perspective_crop(image, corners, output_size):
    """
    Apply perspective transformation to crop the board perfectly.

    Args:
        image: Input image (numpy array)
        corners: List of 4 corner points [[x,y], ...] in order: TL, TR, BR, BL
        output_size: Output image size (square)

    Returns:
        Cropped and warped image
    """
    # Ensure corners are properly ordered
    src_corners = order_corners(corners)

    # Destination corners (perfect square)
    dst_corners = np.array([
        [0, 0],                          # Top-left
        [output_size - 1, 0],            # Top-right
        [output_size - 1, output_size - 1],  # Bottom-right
        [0, output_size - 1]             # Bottom-left
    ], dtype=np.float32)

    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # Apply transformation
    warped = cv2.warpPerspective(
        image, M, (output_size, output_size),
        flags=cv2.INTER_LANCZOS4,  # High-quality interpolation
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return warped


def process_single_image(raw_image_path, corners, output_path, output_size):
    """Process a single image: load, crop, save."""
    # Load image
    image = cv2.imread(raw_image_path)
    if image is None:
        print(f"    Error: Could not load {raw_image_path}")
        return False

    # Apply perspective crop
    cropped = perspective_crop(image, corners, output_size)

    # Save result
    cv2.imwrite(output_path, cropped)
    return True


def process_fen_renders(metadata, input_dir, output_dir, output_size):
    """Process all renders for a single FEN."""
    fen_id = metadata['fen_id']
    view_side = metadata['view_side']

    print(f"\nProcessing FEN: {fen_id} ({view_side} view)")

    results = []

    for view_data in metadata['views']:
        view_name = view_data['view']
        raw_path = view_data['raw_image']
        corners = view_data['corners_2d']

        # Handle both absolute and relative paths
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(input_dir, os.path.basename(raw_path))

        # Output filename
        output_filename = f"{fen_id}_{view_name}.png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"  Processing {view_name}...")
        print(f"    Input:  {raw_path}")
        print(f"    Output: {output_path}")
        print(f"    Corners: {corners}")

        success = process_single_image(raw_path, corners, output_path, output_size)

        if success:
            print(f"    [OK] Cropped to {output_size}x{output_size}")
            results.append({
                'view': view_name,
                'path': output_path,
                'success': True
            })
        else:
            print(f"    [FAILED]")
            results.append({
                'view': view_name,
                'path': output_path,
                'success': False
            })

    return results


def find_metadata_files(input_dir):
    """Find all metadata JSON files in input directory."""
    metadata_files = []
    for f in os.listdir(input_dir):
        if f.endswith('_metadata.json'):
            metadata_files.append(os.path.join(input_dir, f))
    return sorted(metadata_files)


def create_comparison_grid(input_dir, output_dir, fen_id, output_size):
    """Create a side-by-side comparison image (raw vs cropped) for verification."""
    views = ['left', 'middle', 'right']

    # Load images
    raw_images = []
    cropped_images = []

    for view in views:
        raw_path = os.path.join(input_dir, f"{fen_id}_{view}_raw.png")
        cropped_path = os.path.join(output_dir, f"{fen_id}_{view}.png")

        if os.path.exists(raw_path) and os.path.exists(cropped_path):
            raw = cv2.imread(raw_path)
            cropped = cv2.imread(cropped_path)

            # Resize raw to match cropped for comparison
            raw_resized = cv2.resize(raw, (output_size, output_size))

            raw_images.append(raw_resized)
            cropped_images.append(cropped)

    if len(raw_images) != 3:
        print(f"  Warning: Could not create comparison for {fen_id}")
        return None

    # Create grid: top row = raw, bottom row = cropped
    top_row = np.hstack(raw_images)
    bottom_row = np.hstack(cropped_images)

    # Add labels
    label_height = 40
    label_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Create labeled rows
    top_labeled = np.zeros((output_size + label_height, output_size * 3, 3), dtype=np.uint8)
    bottom_labeled = np.zeros((output_size + label_height, output_size * 3, 3), dtype=np.uint8)

    top_labeled[label_height:, :] = top_row
    bottom_labeled[label_height:, :] = bottom_row

    # Add text labels
    cv2.putText(top_labeled, "RAW RENDERS (with background)", (10, 28), font, 0.8, label_color, 2)
    cv2.putText(bottom_labeled, "CROPPED (board fills frame)", (10, 28), font, 0.8, label_color, 2)

    for i, view in enumerate(views):
        x_offset = i * output_size + output_size // 2 - 30
        cv2.putText(top_labeled, view.upper(), (x_offset, 28), font, 0.6, (200, 200, 200), 1)
        cv2.putText(bottom_labeled, view.upper(), (x_offset, 28), font, 0.6, (200, 200, 200), 1)

    # Stack vertically
    comparison = np.vstack([top_labeled, bottom_labeled])

    # Save comparison
    comparison_path = os.path.join(output_dir, f"{fen_id}_comparison.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"\n  Comparison saved: {comparison_path}")

    return comparison_path


def main():
    parser = argparse.ArgumentParser(description='Post-process synthetic chess renders')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw renders and metadata')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for cropped images')
    parser.add_argument('--size', type=int, default=512,
                        help='Output image size (default: 512)')
    parser.add_argument('--comparison', action='store_true',
                        help='Generate comparison images')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("POST-PROCESSING: PERSPECTIVE CROP")
    print("="*70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output size:      {args.size}x{args.size}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all metadata files
    metadata_files = find_metadata_files(args.input_dir)

    if not metadata_files:
        print("\nNo metadata files found!")
        return

    print(f"\nFound {len(metadata_files)} FEN(s) to process")

    # Process each FEN
    all_results = []
    for metadata_path in metadata_files:
        metadata = load_metadata(metadata_path)
        results = process_fen_renders(metadata, args.input_dir, args.output_dir, args.size)
        all_results.extend(results)

        # Generate comparison if requested
        if args.comparison:
            create_comparison_grid(args.input_dir, args.output_dir, metadata['fen_id'], args.size)

    # Summary
    successful = sum(1 for r in all_results if r['success'])
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {successful}/{len(all_results)} images")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
