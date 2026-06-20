# Checkpoint manifest — final model `chess_v5_bright_silABC`

The final realistic-translation model is too large for Git and was trained on the
BGU cluster. Its weights live on the shared drive and must be placed locally to
run the realistic stage of `generate_chessboard_image`.

## Required file

| Field | Value |
|---|---|
| Model name | `chess_v5_bright_silABC` |
| Model type | `paired_geom_hd` (paired, geometry-conditioned, pix2pixHD-style) |
| Generator file | `latest_net_G.pth` |
| Expected path | `checkpoints/chess_v5_bright_silABC/latest_net_G.pth` |
| Approx. size | ~43 MB |
| Download link | **TODO: add Drive/Release link** (placeholder in `generate_chessboard_image.py::CHECKPOINT_URL`) |

Only the **generator** (`*_net_G.pth`) is needed for inference. The discriminator
(`*_net_D.pth`) and any auxiliary classifier (`square_eval.pth`) are training-only.

## Architecture / load contract (for reproducibility)

Built by the framework as:

```python
networks.define_G(
    g_in=21, g_out=3, ngf=64, netG="resnet_9blocks", normG="instance",
    use_dropout=False, init_type="xavier", init_gain=0.02,
    no_antialias=False, no_antialias_up=False, gpu_ids, opt)
# g_in = input_nc(3) + num_seg_classes(15) + num_geom_channels(3) = 21
```

The generator consumes `concat([synthetic_RGB, one_hot(seg, 15), geom(3)])` and
outputs realistic RGB in `[-1, 1]`. `chess_v5_infer/networks.py` reproduces this
exact architecture so the `state_dict` loads without the full framework.

Full training config: `v5_work/final_config/bright_silABC_train_opt.txt`.

## How to install the checkpoint

```bash
mkdir -p checkpoints/chess_v5_bright_silABC
# download latest_net_G.pth from the Drive link above into that folder:
#   checkpoints/chess_v5_bright_silABC/latest_net_G.pth
python run_project3_demo.py            # now produces all three ./results/*.png
```

Requires PyTorch (`pip install torch torchvision`) or the cluster `pytorch` env.

## Legacy weights (NOT the final model)

`trained_model/latest_net_{G,F,D}.pth` (committed, ~58 MB total) are the earlier
**CUT (unpaired)** baseline. They are kept for history only and are **not** used
by `generate_chessboard_image`. The `net_F` (NCE head) is the tell-tale of CUT.

## Note for maintainers

A later internal re-evaluation (2026‑06‑14) suggested a sibling variant,
`chess_v5_bright_silAB` (dropping the global Contextual Loss "C"), may edge out
`silABC` on blind realism judging. The submission deliberately keeps
**`silABC`** as the final model (consistent with the report and config); `silAB`
is noted as future work in the report's limitations.
