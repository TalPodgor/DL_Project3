"""
Creates side-by-side comparison images: Synthetic (real_A) vs Realistic (fake_B)
Output saved to test_output/comparisons/
"""

import os
from PIL import Image, ImageDraw, ImageFont

REAL_A_DIR = "test_output/images/real_A"
FAKE_B_DIR = "test_output/images/fake_B"
OUTPUT_DIR = "test_output/comparisons"
GRID_DIR  = "test_output/grid"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)

filenames = sorted(os.listdir(REAL_A_DIR))
print(f"Found {len(filenames)} images to compare")

# --- 1. Individual side-by-side comparisons ---
for fname in filenames:
    real_path = os.path.join(REAL_A_DIR, fname)
    fake_path = os.path.join(FAKE_B_DIR, fname)

    if not os.path.exists(fake_path):
        print(f"  Missing fake: {fname}")
        continue

    real_img = Image.open(real_path).convert("RGB")
    fake_img = Image.open(fake_path).convert("RGB")

    w, h = real_img.size
    label_h = 30
    comparison = Image.new("RGB", (w * 2 + 10, h + label_h), color=(30, 30, 30))

    # Paste images
    comparison.paste(real_img, (0, label_h))
    comparison.paste(fake_img, (w + 10, label_h))

    # Add labels
    draw = ImageDraw.Draw(comparison)
    draw.text((w // 2 - 40, 5), "SYNTHETIC", fill=(200, 200, 200))
    draw.text((w + 10 + w // 2 - 40, 5), "REALISTIC", fill=(100, 255, 100))

    out_path = os.path.join(OUTPUT_DIR, fname)
    comparison.save(out_path)

print(f"Saved {len(filenames)} individual comparisons to {OUTPUT_DIR}/")

# --- 2. Big grid of first 30 comparisons ---
GRID_COLS = 3
GRID_ROWS = 10
THUMB_W = 512
THUMB_H = 256  # height for each row (synthetic + realistic side by side)
PADDING = 5

grid_files = sorted(filenames)[:GRID_COLS * GRID_ROWS]
grid_w = GRID_COLS * (THUMB_W + PADDING)
grid_h = GRID_ROWS * (THUMB_H + PADDING + 20)
grid_img = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
draw = ImageDraw.Draw(grid_img)

for i, fname in enumerate(grid_files):
    col = i % GRID_COLS
    row = i // GRID_COLS
    x = col * (THUMB_W + PADDING)
    y = row * (THUMB_H + PADDING + 20)

    real_path = os.path.join(REAL_A_DIR, fname)
    fake_path = os.path.join(FAKE_B_DIR, fname)

    if not os.path.exists(fake_path):
        continue

    real_img = Image.open(real_path).convert("RGB").resize((THUMB_W // 2 - 2, THUMB_H - 20))
    fake_img = Image.open(fake_path).convert("RGB").resize((THUMB_W // 2 - 2, THUMB_H - 20))

    grid_img.paste(real_img, (x, y + 20))
    grid_img.paste(fake_img, (x + THUMB_W // 2 + 2, y + 20))

    label = fname.replace(".png", "").replace("game2_frame_", "")
    draw.text((x + 2, y + 2), label[:30], fill=(180, 180, 180))

grid_path = os.path.join(GRID_DIR, "overview_grid.png")
grid_img.save(grid_path)
print(f"Saved overview grid to {grid_path}")
print("\nDone! Open test_output/comparisons/ to see individual comparisons")
print("Open test_output/grid/overview_grid.png for a quick overview")
