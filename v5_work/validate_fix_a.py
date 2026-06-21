"""Decide Fix A identity source: is the silhouette per-pixel color->id reliable
enough to use directly (overhang-correct), or must identity come from FEN?

Measures, across many boards:
  - per occupied cell: dominant non-zero silhouette id vs FEN id (agreement)
  - occupied-cell coverage distribution + guardrail trigger rate (<4% cell)
  - overhang: fraction of silhouette piece-pixels landing in FEN-empty cells
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
ID_TO_PIECE={v:k for k,v in PIECE_TO_ID.items()}

def rgb_to_ids(seg_rgb, threshold=75, bright=30):
    arr=np.asarray(seg_rgb.convert("RGB"),np.int16); flat=arr.reshape(-1,3)
    diff=flat[:,None,:]-PALETTE[None,:,:]; dist2=(diff*diff).sum(2)
    nearest=np.argmin(dist2,1); best=dist2[np.arange(flat.shape[0]),nearest]
    ids=np.zeros(flat.shape[0],np.uint8)
    keep=(best<=threshold*threshold)&(np.max(flat,1)>=bright)
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

root=os.path.join(os.path.dirname(__file__),"..","datasets","chess_v5_oblique_aligned")
labels=json.load(open(os.path.join(root,"labels.json")))
items=list(labels.items())
np.random.seed(0); np.random.shuffle(items)
N=int(sys.argv[1]) if len(sys.argv)>1 else 60
items=items[:N]

cell=64
tot_cells=0; agree=0; type_agree=0   # type_agree ignores color (w/b), checks P/N/B/R/Q/K shape class
cov=[]; guardrail=0
sil_piece_px=0; overhang_px=0
miss_cells=0  # occupied cell with ZERO silhouette pixels
conf={}  # (fen_piece, dominant_sil_piece) -> count for mismatches

for name,meta in items:
    p=os.path.join(root,meta["split"],name+"_seg.png")
    if not os.path.exists(p): continue
    seg=Image.open(p).convert("RGB")
    ids=rgb_to_ids(seg)
    grid=fen_grid(meta["fen"],meta["viewpoint"])
    sil_piece_px+=int((ids>=3).sum())
    for r in range(8):
        for c in range(8):
            blk=ids[r*cell:(r+1)*cell,c*cell:(c+1)*cell]
            fid=int(grid[r,c])
            if fid==0:
                overhang_px+=int((blk>=3).sum()); continue
            tot_cells+=1
            frac=(blk>=3).mean(); cov.append(frac)
            if frac<0.04: guardrail+=1
            nz=blk[blk>=3]
            if nz.size==0: miss_cells+=1; continue
            dom=int(np.bincount(nz).argmax())
            if dom==fid: agree+=1
            else:
                conf[(ID_TO_PIECE[fid],ID_TO_PIECE[dom])]=conf.get((ID_TO_PIECE[fid],ID_TO_PIECE[dom]),0)+1
            # type agreement (lowercase): same shape regardless of color
            if ID_TO_PIECE[dom].lower()==ID_TO_PIECE[fid].lower(): type_agree+=1

cov=np.array(cov)
print(f"boards={len(items)} occupied_cells={tot_cells}")
print(f"dominant sil-id == FEN-id (exact, incl. color): {agree/tot_cells:.4f}")
print(f"dominant sil-id type match (P/N/B/R/Q/K, ignore color): {type_agree/tot_cells:.4f}")
print(f"occupied cells with ZERO silhouette px: {miss_cells} ({miss_cells/tot_cells:.4f})")
print(f"guardrail trigger (cov<4%): {guardrail} ({guardrail/tot_cells:.4f})")
print(f"coverage: mean={cov.mean():.3f} p10={np.percentile(cov,10):.3f} p50={np.percentile(cov,50):.3f}")
print(f"overhang px into FEN-empty cells: {overhang_px} = {overhang_px/max(sil_piece_px,1):.4f} of all sil-piece px")
if conf:
    print("top mismatches (fen->dominant_sil):")
    for k,v in sorted(conf.items(),key=lambda x:-x[1])[:12]:
        print(f"   {k[0]} -> {k[1]} : {v}")
