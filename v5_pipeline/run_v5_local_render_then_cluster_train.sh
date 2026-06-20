#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/v5_local_render_then_cluster_train_${RUN_ID}.log"
STATUS_FILE="$LOG_DIR/v5_local_render_then_cluster_train_${RUN_ID}.status"
exec > >(tee -a "$LOG_FILE") 2>&1

BLENDER="${BLENDER:-/opt/homebrew/bin/blender}"
if [ ! -x "$BLENDER" ] && [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
  BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
fi
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

OUTDIR="${OUTDIR:-$PROJECT_DIR/datasets/chess_v5_oblique}"
REMOTE_REPO="${REMOTE_REPO:-/home/packler/chess_cut_project/contrastive-unpaired-translation}"
REMOTE_SRC="${REMOTE_SRC:-/home/packler/chess_cut_project/v5_cluster_src}"
REMOTE_HOST="${REMOTE_HOST:-bgu}"
NAME="${NAME:-chess_v5_oblique_fenseg_full}"
EP="${EP:-40}"
EPD="${EPD:-20}"
BS="${BS:-2}"
PIECEW="${PIECEW:-0.35}"
PCLS="${PCLS:-1.0}"
SAMPLES="${SAMPLES:-8}"
RESOLUTION="${RESOLUTION:-900}"

count_pairs() {
  local split="$1"
  find "$OUTDIR/$split" -maxdepth 1 -type f -name '*.png' \
    ! -name '*_seg.png' ! -name '*_depth.png' | wc -l | tr -d ' '
}

extract_job_id() {
  awk '{print $NF}'
}

echo "=== V5 local render -> cluster train ==="
echo "started=$(date)"
echo "project=$PROJECT_DIR"
echo "log=$LOG_FILE"
echo "status=$STATUS_FILE"
echo "blender=$BLENDER"
echo "python=$PYTHON_BIN"
echo "outdir=$OUTDIR"
echo "remote=$REMOTE_HOST:$REMOTE_REPO"
echo "name=$NAME ep=$EP epd=$EPD bs=$BS piecew=$PIECEW pcls=$PCLS"

if [ ! -x "$BLENDER" ]; then
  echo "Missing executable Blender at $BLENDER"
  exit 2
fi
if ! "$PYTHON_BIN" - <<'PY'
from PIL import Image
print("PIL_OK")
PY
then
  echo "Selected Python is missing Pillow: $PYTHON_BIN"
  exit 2
fi

echo "stage=render_train started=$(date)" | tee "$STATUS_FILE"
"$PYTHON_BIN" -u "$PROJECT_DIR/v5_pipeline/build_v5_dataset.py" \
  --project-dir "$PROJECT_DIR" \
  --split train \
  --out-dir "$OUTDIR" \
  --samples "$SAMPLES" \
  --resolution "$RESOLUTION" \
  --blender "$BLENDER" \
  --quiet

echo "stage=render_test started=$(date)" | tee -a "$STATUS_FILE"
"$PYTHON_BIN" -u "$PROJECT_DIR/v5_pipeline/build_v5_dataset.py" \
  --project-dir "$PROJECT_DIR" \
  --split test \
  --out-dir "$OUTDIR" \
  --samples "$SAMPLES" \
  --resolution "$RESOLUTION" \
  --blender "$BLENDER" \
  --quiet

TRAIN_COUNT="$(count_pairs train)"
TEST_COUNT="$(count_pairs test)"
echo "local_counts train=$TRAIN_COUNT test=$TEST_COUNT" | tee -a "$STATUS_FILE"

if [ "$TRAIN_COUNT" -lt 736 ] || [ "$TEST_COUNT" -lt 140 ]; then
  echo "Dataset incomplete; refusing to upload/submit training."
  exit 3
fi

echo "stage=upload started=$(date)" | tee -a "$STATUS_FILE"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/datasets/chess_v5_oblique' '$REMOTE_SRC'"
rsync -az --delete "$OUTDIR/" "$REMOTE_HOST:$REMOTE_REPO/datasets/chess_v5_oblique/"
rsync -az "$PROJECT_DIR/v5_pipeline/cluster_src/" "$REMOTE_HOST:$REMOTE_SRC/"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && bash '$REMOTE_SRC/install_v5_oblique_files.sh'"

echo "stage=submit_train started=$(date)" | tee -a "$STATUS_FILE"
TRAIN_SUBMIT="$(
  ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && DATAROOT=./datasets/chess_v5_oblique NAME='$NAME' EP='$EP' EPD='$EPD' BS='$BS' PIECEW='$PIECEW' PCLS='$PCLS' sbatch '$REMOTE_SRC/train_v5_oblique_hd.sbatch'"
)"
TRAIN_JOB="$(printf '%s\n' "$TRAIN_SUBMIT" | extract_job_id)"
echo "$TRAIN_SUBMIT" | tee -a "$STATUS_FILE"

TEST_SUBMIT="$(
  ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && DATAROOT=./datasets/chess_v5_oblique NUM_TEST=140 sbatch --dependency=afterok:$TRAIN_JOB '$REMOTE_SRC/test_v5_oblique_hd.sbatch' latest '$NAME'"
)"
TEST_JOB="$(printf '%s\n' "$TEST_SUBMIT" | extract_job_id)"
echo "$TEST_SUBMIT" | tee -a "$STATUS_FILE"

SCORE_SUBMIT="$(
  ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && DATAROOT=./datasets/chess_v5_oblique sbatch --dependency=afterok:$TEST_JOB '$REMOTE_SRC/score_v5_oblique.sbatch' latest '$NAME'"
)"
SCORE_JOB="$(printf '%s\n' "$SCORE_SUBMIT" | extract_job_id)"
echo "$SCORE_SUBMIT" | tee -a "$STATUS_FILE"

{
  echo "stage=submitted finished=$(date)"
  echo "train_job=$TRAIN_JOB"
  echo "test_job=$TEST_JOB"
  echo "score_job=$SCORE_JOB"
  echo "name=$NAME"
  echo "log=$LOG_FILE"
} | tee -a "$STATUS_FILE"

echo "=== DONE: submitted cluster chain ==="
