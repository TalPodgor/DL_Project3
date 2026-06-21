"""Local (numpy/PIL, no torch) validation of Fix D: lowering the silhouette
brightness gate so dark/black piece AA-edge pixels survive.

Replicates the exact palette + rgb_semantic_to_ids logic from
v5_oblique_dataset.py, sweeps the bright_enough gate, and reports per-color
(white vs black) silhouette coverage + any background leakage.
"""
import json, os, sys
import numpy as np
from PIL import Image

PIECE_TO_ID = {"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
CLASS_COLORS_LINEAR = {
    "P":(0.92,0.88,0.72),"N":(0.82,0.82,0.52),"B":(0.92,0.72,0.44),
    "R":(0.72,0.92,0.72),"Q":(0.72,0.82,0.98),"K":(0.90,0.70,0.98),
    "p":(0.18,0.12,0.08),"n":(0.32,0.18,0.08),"b":(0.20,0.32,0.12),
    "r":(0.10,0.24,0.34),"q":(0.34,0.12,0.30),"k":(0.34,0.10,0.10),
}
def lin2srgb_u8(v):
    v=np.asarray(v,np.float32)
    s=np.where(v<=0.0031308,12.92*v,1.055*np.power(v,1/2.4)-0.055)
    return np.clip(np.round(s*255),0,255).astype(np.int16)
PALETTE=np.stack([lin2srgb_u8(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID])
PALETTE_IDS=np.asarray([PIECE_TO_ID[p] for p in PIECE_TO_ID],np.uint8)
print("sRGB palette (max-channel per piece color):")
for p,row in zip(PIECE_TO_ID,PALETTE):
    print(f"  {p}: {tuple(int(x) for x in row)} max={int(row.max())}")

def rgb_to_ids(seg_rgb, threshold=75, bright=70):
    arr=np.asarray(seg_rgb.convert("RGB"),np.int16); flat=arr.reshape(-1,3)
    diff=flat[:,None,:]-PALETTE[None,:,:]; dist2=(diff*diff).sum(2)
    nearest=np.argmin(dist2,1); best=dist2[np.arange(flat.shape[0]),nearest]
    ids=np.zeros(flat.shape[0],np.uint8)
    bright_enough=np.max(flat,1)>=bright
    keep=(best<=threshold*threshold)&bright_enough
    ids[keep]=PALETTE_IDS[nearest[keep]]
    return ids.reshape(arr.shape[:2])

def fen_grid(fen,viewpoint):
    ids=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            ids[r,c]=PIECE_TO_ID[ch]; c+=1
    if viewpoint=="black": ids=np.rot90(ids,2)
    return ids

d="datasets/chess_v5_oblique_aligned"
labels=json.load(open(os.path.join(os.path.dirname(__file__),"..",d,"labels.json")))
name=sys.argv[1] if len(sys.argv)>1 else "game2_frame_000200_black_middle"
meta=labels[name]; split=meta["split"]
base=os.path.join(os.path.dirname(__file__),"..",d,split,name)
seg=Image.open(base+"_seg.png").convert("RGB")
tile=seg.size[0]; cell=tile//8
grid=fen_grid(meta["fen"],meta["viewpoint"])
print(f"\nboard={name} split={split} tile={tile} cell={cell} fen={meta['fen']}")

def coverage(ids):
    """per-cell occupied-piece coverage, split white vs black, + bg leak."""
    wcov=[]; bcov=[]; bg_leak=0; bg_cells=0
    for r in range(8):
        for c in range(8):
            blk=ids[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            pid=int(grid[r,c]); frac=(blk>=3).mean()
            if pid==0:
                bg_cells+=1; bg_leak+=(blk>=3).sum()
            elif pid>=9:  # black piece
                bcov.append(frac)
            else:
                wcov.append(frac)
    return (np.mean(wcov) if wcov else 0, np.mean(bcov) if bcov else 0,
            bg_leak, bg_cells)

print(f"\n{'gate':>5} {'thr':>4} | white_cov black_cov | bg_leak_px (over %d empty cells)"%(64-int((grid>0).sum())))
for bright in (70,50,40,30,20,10):
    ids=rgb_to_ids(seg,75,bright)
    w,b,leak,bgc=coverage(ids)
    print(f"{bright:>5} {75:>4} | {w:8.3f} {b:9.3f} | {leak:8d}")

# also sweep distance threshold at the chosen gate
print("\n-- distance-threshold sweep at gate=30 --")
for thr in (60,75,90,110):
    ids=rgb_to_ids(seg,thr,30)
    w,b,leak,bgc=coverage(ids)
    print(f"  thr={thr:>3} | white={w:.3f} black={b:.3f} bg_leak={leak}")

# save side-by-side overlays for visual check (gate 70 vs 30)
def overlay(ids):
    rgb=np.asarray(seg.convert("RGB")).copy()
    mask=ids>=3
    rgb[mask]=(0.4*rgb[mask]+0.6*np.array([255,0,255])).astype(np.uint8)
    return rgb
out=np.concatenate([np.asarray(seg.convert("RGB")),
                    overlay(rgb_to_ids(seg,75,70)),
                    overlay(rgb_to_ids(seg,75,30))],axis=1)
op=os.path.join(os.path.dirname(__file__),"fix_d_overlay.png")
Image.fromarray(out).save(op)
print(f"\nsaved overlay (seg | gate70 | gate30) -> {op}")
