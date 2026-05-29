"""
Synthetic Chess Data Generation - Main Orchestrator
====================================================
Complete pipeline for generating perfectly cropped synthetic chessboard images.

This script:
1. Calls Blender to render raw images with corner coordinates
2. Applies perspective cropping to ensure perfect board alignment
3. Organizes outputs for CUT/GAN training

TEST MODE (3 hardcoded FENs):
    python run_generation.py --test

BATCH MODE (process CSV file):
    python run_generation.py --input_csv game_fens.csv --output_dir ./dataset

Requirements:
    - Blender installed and accessible via command line
    - OpenCV (pip install opencv-python)
    - chess-set.blend in the same directory
"""

import subprocess
import os
import sys
import argparse
import json
from pathlib import Path
import shutil

# ==========================
# CONFIGURATION
# ==========================

# Path to Blender executable - UPDATE THIS FOR YOUR SYSTEM
# Windows example: r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
# Linux example: "/home/user/blender-4.0/blender"
# macOS example: "/Applications/Blender.app/Contents/MacOS/Blender"
BLENDER_PATH = "blender"  # Assumes blender is in PATH

# Default settings
DEFAULT_RESOLUTION = 800  # Raw render resolution (cropped to OUTPUT_SIZE)
DEFAULT_OUTPUT_SIZE = 512  # Final cropped image size
DEFAULT_SAMPLES = 64       # Cycles render samples (lower = faster, noisier)

# Test FEN positions (fallback if no CSV provided)
TEST_FENS = [
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "id": "starting_position",
        "description": "Starting position - all pieces visible"
    },
    {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
        "id": "italian_game",
        "description": "Italian Game opening - early game"
    },
    {
        "fen": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1",
        "id": "middlegame_complex",
        "description": "Complex middlegame - many pieces, both sides castled"
    },
]


def load_fens_from_csv(csv_path, limit=None):
    """Load FEN positions from a game CSV file."""
    import csv

    fens = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            fens.append({
                'fen': row['fen'],
                'id': f"frame_{int(row['from_frame']):06d}",
                'frame': int(row['from_frame']),
                'description': f"Frame {row['from_frame']}"
            })

    return fens


def find_blender():
    """Find Blender executable."""
    # Check if custom path is set
    if os.path.isfile(BLENDER_PATH):
        return BLENDER_PATH

    # Try common locations
    common_paths = [
        # Windows
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        # Linux
        "/usr/bin/blender",
        "/snap/bin/blender",
        os.path.expanduser("~/blender/blender"),
        # macOS
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]

    for path in common_paths:
        if os.path.isfile(path):
            return path

    # Try system PATH
    try:
        result = subprocess.run(["blender", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            return "blender"
    except FileNotFoundError:
        pass

    return None


def run_blender_generation(blender_path, blend_file, fen, fen_id, output_dir, resolution, samples, view='black'):
    """Run Blender to generate raw renders."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_chess_generator.py")

    cmd = [
        blender_path,
        blend_file,
        "--background",
        "--python", script_path,
        "--",
        "--fen", fen,
        "--fen_id", fen_id,
        "--output_dir", output_dir,
        "--resolution", str(resolution),
        "--samples", str(samples),
        "--view", view
    ]

    print(f"\n  Running Blender...")
    print(f"  Command: {' '.join(cmd[:5])} ... --fen \"{fen[:30]}...\"")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per FEN
        )

        if result.returncode != 0:
            print(f"  [ERROR] Blender failed!")
            print(f"  STDERR: {result.stderr[-500:] if result.stderr else 'None'}")
            return False

        print(f"  [OK] Blender rendering complete")
        return True

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Blender timed out!")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def run_postprocessing(input_dir, output_dir, output_size, create_comparison=True):
    """Run perspective cropping post-processing."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprocess_crop.py")

    cmd = [
        sys.executable,  # Use same Python interpreter
        script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--size", str(output_size),
    ]

    if create_comparison:
        cmd.append("--comparison")

    print(f"\n  Running post-processing...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  [ERROR] Post-processing failed!")
            print(f"  STDERR: {result.stderr[-500:] if result.stderr else 'None'}")
            return False

        print(result.stdout)
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def generate_single_fen(blender_path, blend_file, fen, fen_id, raw_dir, final_dir,
                        resolution, output_size, samples, view='black'):
    """Generate images for a single FEN position."""
    print(f"\n{'='*70}")
    print(f"GENERATING: {fen_id}")
    print(f"{'='*70}")
    print(f"FEN: {fen}")

    # Step 1: Blender rendering
    success = run_blender_generation(
        blender_path, blend_file, fen, fen_id,
        raw_dir, resolution, samples, view
    )

    if not success:
        return False

    # Step 2: Post-processing (cropping)
    success = run_postprocessing(raw_dir, final_dir, output_size, create_comparison=True)

    return success


def run_test_mode(blender_path, blend_file, output_base_dir, resolution, output_size, samples,
                  test_csv=None, num_test=3, view='black'):
    """Run test generation with FEN positions from CSV or hardcoded defaults."""

    # Load test FENs
    if test_csv and os.path.exists(test_csv):
        test_fens = load_fens_from_csv(test_csv, limit=num_test)
        source = f"CSV: {os.path.basename(test_csv)}"
    else:
        test_fens = TEST_FENS[:num_test]
        source = "hardcoded defaults"

    print("\n" + "="*70)
    print(f"TEST MODE: Generating {len(test_fens)} test positions")
    print(f"Source: {source}")
    print(f"View: {view} (pieces closer to camera)")
    print("="*70)

    raw_dir = os.path.join(output_base_dir, "raw")
    final_dir = os.path.join(output_base_dir, "cropped")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    results = []

    for i, test_fen in enumerate(test_fens, 1):
        print(f"\n\n{'#'*70}")
        print(f"# TEST {i}/{len(test_fens)}: {test_fen['description']}")
        print(f"# FEN: {test_fen['fen']}")
        print(f"{'#'*70}")

        success = generate_single_fen(
            blender_path, blend_file,
            test_fen['fen'], test_fen['id'],
            raw_dir, final_dir,
            resolution, output_size, samples,
            view=view
        )

        results.append({
            'fen_id': test_fen['id'],
            'fen': test_fen['fen'],
            'success': success
        })

    # Print summary
    print("\n\n" + "="*70)
    print("TEST GENERATION SUMMARY")
    print("="*70)

    for r in results:
        status = "[OK]" if r['success'] else "[FAILED]"
        print(f"  {status} {r['fen_id']}")

    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal: {successful}/{len(results)} successful")
    print(f"\nOutput directories:")
    print(f"  Raw renders: {raw_dir}")
    print(f"  Cropped images: {final_dir}")

    if successful > 0:
        print(f"\nNext steps:")
        print(f"  1. Check the comparison images in {final_dir}")
        print(f"  2. Verify board corners align perfectly with image corners")
        print(f"  3. If satisfied, run full dataset generation")

    return results


def run_batch_mode(blender_path, blend_file, input_csv, output_dir, resolution, output_size, samples, view='black'):
    """Process FENs from a CSV file."""

    raw_dir = os.path.join(output_dir, "raw")
    final_dir = os.path.join(output_dir, "cropped")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Load all FENs from CSV
    fens = load_fens_from_csv(input_csv)

    print(f"\n{'='*70}")
    print(f"BATCH MODE: Processing {len(fens)} FEN positions")
    print(f"Source: {os.path.basename(input_csv)}")
    print(f"View: {view}")
    print(f"{'='*70}")

    results = []
    for i, fen_data in enumerate(fens, 1):
        print(f"\n[{i}/{len(fens)}] Processing {fen_data['id']}...")

        success = generate_single_fen(
            blender_path, blend_file,
            fen_data['fen'], fen_data['id'],
            raw_dir, final_dir,
            resolution, output_size, samples,
            view=view
        )

        results.append({'fen_id': fen_data['id'], 'success': success})

    successful = sum(1 for r in results if r['success'])
    print(f"\n\nBatch complete: {successful}/{len(results)} successful")
    print(f"Output: {final_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate perfectly cropped synthetic chessboard images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test with CSV (first 3 FENs):
    python run_generation.py --test --test_csv game2.csv

  Test with custom count:
    python run_generation.py --test --test_csv game2.csv --num_test 5

  Batch process entire CSV:
    python run_generation.py --input_csv game2.csv --output ./dataset

  With custom Blender path:
    python run_generation.py --test --test_csv game2.csv --blender "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"
        """
    )

    parser.add_argument('--test', action='store_true',
                        help='Run test mode with limited FEN positions')
    parser.add_argument('--test_csv', type=str,
                        help='CSV file to use for test mode (default: use hardcoded FENs)')
    parser.add_argument('--num_test', type=int, default=3,
                        help='Number of FENs to process in test mode (default: 3)')
    parser.add_argument('--input_csv', type=str,
                        help='CSV file for batch processing (processes ALL FENs)')
    parser.add_argument('--output', type=str, default='./synthetic_output',
                        help='Output directory (default: ./synthetic_output)')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION,
                        help=f'Raw render resolution (default: {DEFAULT_RESOLUTION})')
    parser.add_argument('--size', type=int, default=DEFAULT_OUTPUT_SIZE,
                        help=f'Final cropped image size (default: {DEFAULT_OUTPUT_SIZE})')
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                        help=f'Render samples (default: {DEFAULT_SAMPLES})')
    parser.add_argument('--blend_file', type=str, default='chess-set.blend',
                        help='Path to Blender file (default: chess-set.blend)')
    parser.add_argument('--blender', type=str,
                        help='Path to Blender executable')
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'],
                        help='Camera view side (default: black)')

    args = parser.parse_args()

    # Validate arguments
    if not args.test and not args.input_csv:
        parser.print_help()
        print("\nError: Must specify either --test or --input_csv")
        sys.exit(1)

    # Find Blender
    blender_path = args.blender or find_blender()
    if not blender_path:
        print("Error: Could not find Blender!")
        print("\nPlease either:")
        print("  1. Add Blender to your system PATH")
        print("  2. Use --blender to specify the path")
        print("  3. Edit BLENDER_PATH in this script")
        sys.exit(1)

    print(f"Using Blender: {blender_path}")

    # Check blend file
    blend_file = args.blend_file
    if not os.path.isfile(blend_file):
        # Try looking in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        blend_file = os.path.join(script_dir, args.blend_file)

    if not os.path.isfile(blend_file):
        print(f"Error: Could not find blend file: {args.blend_file}")
        sys.exit(1)

    print(f"Using blend file: {blend_file}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run appropriate mode
    if args.test:
        run_test_mode(
            blender_path, blend_file, args.output,
            args.resolution, args.size, args.samples,
            test_csv=args.test_csv,
            num_test=args.num_test,
            view=args.view
        )
    else:
        run_batch_mode(
            blender_path, blend_file, args.input_csv, args.output,
            args.resolution, args.size, args.samples
        )


if __name__ == "__main__":
    main()
