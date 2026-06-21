"""Self-consistent piece-defect audit (numpy/PIL, no torch). Run identically on a
baseline fake_B dir and a new-model fake_B dir to get honest before/after deltas.

Per occupied cell (FEN), within the per-class crop window, using the synthetic
silhouette (from {name}_seg.png) as the piece region:
  - transparent_head : head-region luminance contrast ratio fake/real (low = washed/see-through)
  - edge_halo        : excess fake piece-like energy in the ring OUTSIDE the silhouette
  - ghost_echo       : secondary vertical luminance peak above the piece top in fake not in real
  - merge (adjacent) : cross-boundary luminance continuity fake vs real for adjacent pieces

Usage: python3 audit_defects.py <fake_B_dir> <real_B_dir> <dataset_dir> <out_json> [tag]
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
WIN_BY_CLASS={"P":112,"N":128,"B":160,"R":128,"Q":160,"K":160,
              "p":112,"n":128,"b":160,"r":128,"q":160,"k":160}
def lin2srgb_u8(v):
    v=np.asarray(v,np.float32)
    s=np.where(v<=0.0031308,12.92*v,1.055*np.power(v,1/2.4)-0.055)
    return np.clip(np.round(s*255),0,255).astype(np.int16)
PALETTE=np.stack([lin2srgb_u8(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID])
PALETTE_IDS=np.asarray([PIECE_TO_ID[p] for p in PIECE_TO_ID],np.uint8)
ID2P={v:k for k,v in PIECE_TO_ID.items()}

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

def lum(img):  # HxWx3 uint8 -> HxW float
    a=np.asarray(img.convert("RGB"),np.float32)
    return 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]

def dilate(mask, k=3):
    m=mask.astype(np.float32); out=m.copy()
    for dy in range(-k,k+1):
        for dx in range(-k,k+1):
            out=np.maximum(out, np.roll(np.roll(m,dy,0),dx,1))
    return out>0.5

def grad_mag(L):
    gy=np.zeros_like(L); gx=np.zeros_like(L)
    gy[1:-1,:]=L[2:,:]-L[:-2,:]; gx[:,1:-1]=L[:,2:]-L[:,:-2]
    return np.sqrt(gx*gx+gy*gy)

def audit(fake_dir, real_dir, ds_dir, tag=""):
    labels=json.load(open(os.path.join(ds_dir,"labels.json")))
    files=sorted(f for f in os.listdir(fake_dir) if f.endswith(".png"))
    rows=[]; merges=[]
    for f in files:
        name=f[:-4]
        if name not in labels: continue
        meta=labels[name]
        segp=os.path.join(ds_dir,meta["split"],name+"_seg.png")
        if not os.path.exists(segp): continue
        fake=Image.open(os.path.join(fake_dir,f)); real=Image.open(os.path.join(real_dir,f))
        Lf=lum(fake); Lr=lum(real); tile=Lf.shape[0]; cell=tile//8
        sil=rgb_to_ids(Image.open(segp))
        grid=fen_grid(meta["fen"],meta["viewpoint"])
        Gf=grad_mag(Lf); Gr=grad_mag(Lr)
        cell_lum_f={}; cell_lum_r={}
        for r in range(8):
            for c in range(8):
                fid=int(grid[r,c])
                if fid<3: continue
                p=ID2P[fid]; win=WIN_BY_CLASS[p]; h=win//2
                cy=r*cell+32; cx=c*cell+32
                y0,y1=max(0,cy-h),min(tile,cy+h); x0,x1=max(0,cx-h),min(tile,cx+h)
                S=sil[y0:y1,x0:x1]>=3
                if S.sum()<20: continue
                lf=Lf[y0:y1,x0:x1]; lr=Lr[y0:y1,x0:x1]
                gf=Gf[y0:y1,x0:x1]; gr=Gr[y0:y1,x0:x1]
                ys=np.where(S.any(1))[0]
                top=ys.min(); bot=ys.max(); ht=bot-top+1
                head=np.zeros_like(S); head[top:top+max(1,int(ht*0.35))]=True; head&=S
                # transparent head: contrast (std) of luminance in head silhouette
                fhc=float(lf[head].std()) if head.sum()>5 else 0.0
                rhc=float(lr[head].std()) if head.sum()>5 else 0.0
                trans=fhc/(rhc+1e-3)
                # edge halo: piece-like energy in ring outside silhouette (fake vs real)
                ring=dilate(S,4)&(~dilate(S,1))
                halo=float((gf[ring].mean() if ring.sum() else 0)-(gr[ring].mean() if ring.sum() else 0))
                # ghost/echo: secondary luminance bump ABOVE the piece top, in fake not real
                gh0=max(0,top-int(ht*0.4))
                fcol=lf[gh0:top].mean() if top>gh0 else 0.0
                rcol=lr[gh0:top].mean() if top>gh0 else 0.0
                base=lf[S].mean()
                ghost=float(max(0.0,fcol-rcol))
                # detail: gradient energy ratio inside silhouette
                detail=float(gf[S].mean()/(gr[S].mean()+1e-3))
                rows.append(dict(name=name,piece=p,color=("w" if p.isupper() else "b"),
                                 trans=trans,fhc=fhc,rhc=rhc,halo=halo,ghost=ghost,detail=detail))
                cell_lum_f[(r,c)]=lf; cell_lum_r[(r,c)]=lr
        # adjacent merge along files (columns) within a rank
        for r in range(8):
            for c in range(7):
                if (r,c) in cell_lum_f and (r,c+1) in cell_lum_f:
                    bx=(c+1)*cell
                    band_f=Lf[:, bx-3:bx+3]; band_r=Lr[:, bx-3:bx+3]
                    # low std across boundary in fake but real has a gap = merged
                    merges.append(float(band_r[band_r>40].std() - band_f[band_f>40].std()))
    rows=rows or [dict(trans=0,fhc=0,rhc=0,halo=0,ghost=0,detail=0,color="w",piece="P")]
    def arr(k): return np.array([x[k] for x in rows],float)
    summ={
        "tag":tag,"n_pieces":len(rows),
        "transparent_head_ratio_mean":float(arr("trans").mean()),
        "transparent_head_flag_rate":float((arr("trans")<0.6).mean()),
        "edge_halo_delta_mean":float(arr("halo").mean()),
        "edge_halo_flag_rate":float((arr("halo")>0.5).mean()),
        "ghost_echo_mean":float(arr("ghost").mean()),
        "ghost_echo_flag_rate":float((arr("ghost")>3.0).mean()),
        "detail_ratio_mean":float(arr("detail").mean()),
        "merge_delta_mean":float(np.mean(merges) if merges else 0.0),
    }
    # per-color (white vs black) transparent + detail
    for col in ("w","b"):
        sub=[x for x in rows if x["color"]==col]
        if sub:
            summ[f"trans_ratio_{col}"]=float(np.mean([x["trans"] for x in sub]))
            summ[f"detail_ratio_{col}"]=float(np.mean([x["detail"] for x in sub]))
    return summ

if __name__=="__main__":
    fake,real,ds,out=sys.argv[1:5]
    tag=sys.argv[5] if len(sys.argv)>5 else os.path.basename(fake.rstrip("/"))
    s=audit(fake,real,ds,tag)
    json.dump(s,open(out,"w"),indent=2)
    print(json.dumps(s,indent=2))
