# Parallax8 ABC Fine-Tune Experiment

Date: 2026-06-19

## Motivation

The external geometric diagnosis argues that board-plane homography can align
piece bases while leaving elevated piece heads/bodies misaligned. The strongest
existing dataset, `chess_v5_oblique_aligned_bright`, already uses per-game
camera elevations around 38-44 degrees, so a plain "fix camera angle" rerun
would mostly duplicate `bright_silABC`.

This experiment tests a narrower source-side geometry correction: keep the real
targets and labels fixed, but stretch the rendered synthetic piece silhouettes
upward by a small amount before fine-tuning from `bright_silABC`.

## Local Dataset

Script:

```bash
python3 v5_work/parallax_boost_dataset.py \
  --src datasets/chess_v5_oblique_aligned_bright \
  --out datasets/chess_v5_oblique_aligned_bright_parallax8 \
  --splits train,test \
  --max-extra 8 \
  --min-extra 2 \
  --force \
  --sheet v5_work/parallax8_sheet.png
```

Counts:

- train: 736 pairs, 736 seg, 736 depth
- test: 140 pairs, 140 seg, 140 depth

Preview:

- `v5_work/parallax8_sheet.png`

## Cluster Run

Dataset uploaded to:

```text
bgu:/home/packler/chess_cut_project/contrastive-unpaired-translation/datasets/chess_v5_oblique_aligned_bright_parallax8
```

Training name:

```text
chess_v5_bright_silABC_parallax8_ft15
```

Jobs:

- train: 18261047
- test: 18261048
- score: 18261049

Command-equivalent:

```bash
DATAROOT=./datasets/chess_v5_oblique_aligned_bright_parallax8 \
NAME=chess_v5_bright_silABC_parallax8_ft15 \
INIT=./checkpoints/chess_v5_bright_silABC/latest_net_G.pth \
BS=1 EP=10 EPD=5 \
PIECEW=0.7 PCLS=0.5 \
SEMSRC=fen_silhouette EDGE=1.0 \
VGG=3.0 VGGMAX=224 \
CX=0.5 CXSZ=32 \
PGAN=1.0 PFM=10.0 PVGG=5.0 \
PCROPS=64 PCROPSZ=96 \
sbatch ~/chess_cut_project/v5_cluster_src/train_v5_oblique_hd.sbatch
```

## Replacement Rule

Do not replace `bright_silABC` only because this is a new run. It must clearly
improve visual piece separation without introducing the synthetic/milky failure
seen in geometry-lock runs.

Minimum evidence to consider replacing ABC:

- visual board and crop sheets look at least as real as `bright_silABC`
- no obvious new ghosting from the source-side stretch
- square-eval metrics are not worse in occupancy/type in a meaningful way
- double-head/split-cue audits improve materially, not just by noise

## Result

Completed at 2026-06-20 01:16 IDT.

Pulled outputs:

- `v5_work/eval_parallax8_ft15/fake_B`
- `v5_work/reports_noC/report_chess_v5_bright_silABC_parallax8_ft15_latest.json`
- `v5_work/parallax8_vs_baselines_sheet.png`
- `v5_work/double_head_audit_parallax8`
- `v5_work/audit_parallax8_ft15.json`

Square-eval comparison:

| run | square | occupancy | type | whole-board occ exact | phantom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bright_silAB` | 0.9160 | 0.9924 | 0.7849 | 0.6429 | 0.0057 |
| `bright_silABC` | 0.9221 | 0.9916 | 0.8040 | 0.7571 | 0.0057 |
| `parallax8_ft15` | 0.9094 | 0.9846 | 0.7879 | 0.4500 | 0.0167 |

Double-head audit:

| run | score mean | any-extra | pawns any-extra | officers any-extra |
| --- | ---: | ---: | ---: | ---: |
| `bright_silAB` | 2.4544 | 0.5342 | 0.5932 | 0.4481 |
| `bright_silABC` | 2.4004 | 0.5463 | 0.6236 | 0.4336 |
| `parallax8_ft15` | 2.5367 | 0.5593 | 0.6346 | 0.4496 |

Defect/style audit:

| run | transparent head | edge halo | detail ratio | merge delta |
| --- | ---: | ---: | ---: | ---: |
| `bright_silAB` | 0.0050 | 0.2724 | 0.8898 | 8.2086 |
| `bright_silABC` | 0.0022 | 0.3075 | 0.9129 | 8.4893 |
| `parallax8_ft15` | 0.0053 | 0.2727 | 0.8707 | 8.3487 |

Decision: reject `parallax8_ft15`. It does not improve visual structure and it
meaningfully hurts occupancy/phantom behavior. Do not submit another training
run based on this source-side stretch; it is evidence that a naive geometric
post-render stretch is not the missing fix.
