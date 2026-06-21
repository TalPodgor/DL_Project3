"""Build + validate the chosen Fix A map (fen_silhouette_seg):
   - empty-board ids (light/dark) outside the silhouette
   - per-pixel render id inside the silhouette (overhang-correct identity+shape)
   - FEN base-patch guardrail: every FEN piece keeps an identity footprint
Reports occupancy guarantee, overhang preservation, halo reduction vs full-cell,
and saves a color-coded visualization.
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
EMPTY_LIGHT, EMPTY_DARK = 1, 2
def lin2srgb_u8(v):
    v=np.asarray(v,np.float32)
    s=np.where(v<=0.0031308,12.92*v,1.055*np.power(v,1/2.4)-0.055)
    return np.clip(np.round(s*255),0,255).astype(np.int16)
PALETTE=np.stack([lin2srgb_u8(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID])
PALETTE_IDS=np.asarray([PIECE_TO_ID[p] for p in PIECE_TO_ID],np.uint8)

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

def fen_silhouette_seg(grid, sil_ids, tile, base_floor=0.04):
    """grid: 8x8 FEN ids; sil_ids: HxW render ids (0 bg). Returns HxW seg ids."""
    cell=tile//8
    out=np.zeros((tile,tile),np.uint8)
    # 1) empty-board base
    for r in range(8):
        for c in range(8):
            out[r*cell:(r+1)*cell,c*cell:(c+1)*cell]= EMPTY_LIGHT if (r+c)%2==0 else EMPTY_DARK
    # 2) visible piece identity (overhang-correct)
    pm=sil_ids>=3
    out[pm]=sil_ids[pm]
    # 3) FEN base-patch guardrail: ensure each occupied cell carries >=floor of its own id
    bh0=int(cell*0.45)  # bottom 55% of the cell (piece base region)
    bx0=int(cell*0.15); bx1=int(cell*0.85)
    for r in range(8):
        for c in range(8):
            fid=int(grid[r,c])
            if fid==0: continue
            blk=out[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            own=(blk==fid).mean()
            if own<base_floor:
                # paint own id into base patch, but don't clobber a *different* visible piece
                sub=blk[bh0:cell, bx0:bx1]
                free=(sub<3)|(sub==fid)
                sub[free]=fid
    return out

def colorize(ids):
    """ids HxW -> RGB using palette for pieces, gray for empties."""
    h,w=ids.shape; out=np.zeros((h,w,3),np.uint8)
    out[ids==EMPTY_LIGHT]=(70,70,70); out[ids==EMPTY_DARK]=(35,35,35)
    for pid in range(3,15):
        out[ids==pid]=PALETTE[pid-3]
    return out

root=os.path.join(os.path.dirname(__file__),"..","datasets","chess_v5_oblique_aligned")
labels=json.load(open(os.path.join(root,"labels.json")))
items=list(labels.items()); np.random.seed(0); np.random.shuffle(items)
N=int(sys.argv[1]) if len(sys.argv)>1 else 80
sample=items[:N]; cell=64

tot=0; floor_ok=0; overhang_kept=0; overhang_tot=0; halo_full=0; halo_new=0
for name,meta in sample:
    p=os.path.join(root,meta["split"],name+"_seg.png")
    if not os.path.exists(p): continue
    seg=Image.open(p).convert("RGB"); tile=seg.size[0]
    sil=rgb_to_ids(seg); grid=fen_grid(meta["fen"],meta["viewpoint"])
    out=fen_silhouette_seg(grid,sil,tile)
    for r in range(8):
        for c in range(8):
            blk=out[r*cell:(r+1)*cell,c*cell:(c+1)*cell]
            fid=int(grid[r,c])
            if fid>=3:
                tot+=1
                if (blk==fid).mean()>=0.04: floor_ok+=1
            else:
                # empty cell: how many piece px (halo). full-cell would paint 0 here anyway,
                # but measure piece coverage that is *legit overhang* preserved
                overhang_tot+=int((sil[r*cell:(r+1)*cell,c*cell:(c+1)*cell]>=3).sum())
                overhang_kept+=int((blk>=3).sum())
print(f"boards~{N} occupied_cells={tot}")
print(f"occupied cells with >=4% own-id footprint AFTER guardrail: {floor_ok/tot:.4f} (target ~1.0)")
print(f"overhang piece-px preserved in empty cells: {overhang_kept}/{overhang_tot} = {overhang_kept/max(overhang_tot,1):.4f}")

# visualize 3 boards: raw seg | full-cell FEN | fen_silhouette
def fullcell(grid,tile):
    out=np.zeros((tile,tile),np.uint8); cell=tile//8
    for r in range(8):
        for c in range(8):
            fid=int(grid[r,c])
            out[r*cell:(r+1)*cell,c*cell:(c+1)*cell]= fid if fid>=3 else (EMPTY_LIGHT if (r+c)%2==0 else EMPTY_DARK)
    return out
rows=[]
for name,meta in sample[:3]:
    p=os.path.join(root,meta["split"],name+"_seg.png")
    seg=Image.open(p).convert("RGB"); tile=seg.size[0]
    sil=rgb_to_ids(seg); grid=fen_grid(meta["fen"],meta["viewpoint"])
    out=fen_silhouette_seg(grid,sil,tile)
    row=np.concatenate([np.asarray(seg.convert("RGB")),colorize(fullcell(grid,tile)),colorize(out)],axis=1)
    rows.append(row)
op=os.path.join(os.path.dirname(__file__),"fix_a_overlay.png")
Image.fromarray(np.concatenate(rows,axis=0)).save(op)
print(f"saved (raw seg | FULL-CELL fen | FEN_SILHOUETTE) x3 -> {op}")
