#!/bin/bash
set -euo pipefail

# Run from the CUT repository root on the cluster:
#   cd ~/chess_cut_project/contrastive-unpaired-translation
#   bash ~/chess_cut_project/v5_cluster_src/install_v5_oblique_files.sh

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "$SRC_DIR/v5_oblique_dataset.py" ./data/v5_oblique_dataset.py
cp "$SRC_DIR/paired_geom_hd_model.py" ./models/paired_geom_hd_model.py
cp "$SRC_DIR/square_eval.py" ./square_eval.py

echo "Installed V5 oblique dataset/model/evaluator into $(pwd)"
