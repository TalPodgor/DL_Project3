"""Objective double-head / vanishing-head detector for generated pieces, vs real.
Works for SHORT pieces (pawns) and tall ones. Reports rates by piece type + color.

Per piece: build a vertical luminance profile in the silhouette column. Piece is
darker than the wooden board. Detect:
  - double/ghost  : >=2 vertically-separated dark lobes in GEN that real does NOT have
  - vanishing/top : GEN top region fades to board level while REAL top stays dark
Usage: python3 detect_head_artifacts.py <fake_dir> <real_dir> <ds_dir> [tag]
"""
import json, os, sys
import numpy as np
from PIL import Image

PIECE_TO_ID={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
CLASS_COLORS_LINEAR={"P":(0.92,0.88,0.72),"N":(0.82,0.82,0.52),"B":(0.92,0.72,0.44),
 "R":(0.72,0.92,0.72),"Q":(0.72,0.82,0.98),"K":(0.90,0.70,0.98),"p":(0.18,0.12,0.08),
 "n":(0.32,0.18,0.08),"b":(0.20,0.32,0.12),"r":(0.10,0.24,0.34),"q":(0.34,0.12,0.30),"k":(0.34,0.10,0.10)}
def l2s(v):
    v=np.asarray(v,np.float32); s=np.where(v<=0.0031308,12.92*v,1.055*np.power(v,1/2.4)-0.055)
    return np.clip(np.round(s*255),0,255).astype(np.int16)
PALETTE=np.stack([l2s(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID]); PIDS=np.array([PIECE_TO_ID[p] for p in PIECE_TO_ID],np.uint8)
ID2P={v:k for k,v in PIECE_TO_ID.items()}
def sids(seg,thr=75,br=30):
    a=np.asarray(seg.convert("RGB"),np.int16); f=a.reshape(-1,3)
    d=((f[:,None,:]-PALETTE[None])**2).sum(2); n=d.argmin(1); b=d[np.arange(len(f)),n]
    o=np.zeros(len(f),np.uint8); k=(b<=thr*thr)&(f.max(1)>=br); o[k]=PIDS[n[k]]
    return o.reshape(a.shape[:2])
def fen_grid(fen,vp):
    g=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            g[r,c]=PIECE_TO_ID[ch]; c+=1
    if vp=="black": g=np.rot90(g,2)
    return g
def lum(a): return 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
WIN={"P":104,"p":104,"N":120,"n":120,"B":150,"b":150,"R":120,"r":120,"Q":150,"q":150,"K":150,"k":150}

def lobes(profile, board, delta=22, minlen=3):
    """count vertically separated dark runs (piece darker than board by delta)."""
    dark=profile < (board-delta)
    runs=0; cur=0; lens=[]
    for v in dark:
        if v: cur+=1
        else:
            if cur>=minlen: runs+=1; lens.append(cur)
            cur=0
    if cur>=minlen: runs+=1; lens.append(cur)
    return runs,lens

def analyze(fake,real,ds,tag):
    labels=json.load(open(os.path.join(ds,"labels.json")))
    files=sorted(f for f in os.listdir(fake) if f.endswith(".png"))
    from collections import defaultdict
    cnt=defaultdict(int); dbl=defaultdict(int); van=defaultdict(int)
    for f in files:
        name=f[:-4]
        if name not in labels: continue
        m=labels[name]; segp=os.path.join(ds,m["split"],name+"_seg.png")
        if not os.path.exists(segp): continue
        fk=np.asarray(Image.open(os.path.join(fake,f)).convert("RGB"),np.float32)
        rl=np.asarray(Image.open(os.path.join(real,f)).convert("RGB"),np.float32)
        Lf=lum(fk); Lr=lum(rl); tile=Lf.shape[0]; cell=tile//8
        sil=sids(Image.open(segp)); g=fen_grid(m["fen"],m["viewpoint"])
        for r in range(8):
            for c in range(8):
                pid=int(g[r,c])
                if pid<3: continue
                p=ID2P[pid]; win=WIN[p]; h=win//2
                cy=r*cell+cell//2; cx=c*cell+cell//2
                y0,y1=max(0,cy-h),min(tile,cy+h); x0,x1=max(0,cx-h),min(tile,cx+h)
                S=sil[y0:y1,x0:x1]>=3
                if S.sum()<25: continue
                cols=np.where(S.any(0))[0]
                xc0,xc1=cols.min(),cols.max(); band=slice(xc0,xc1+1)
                lf=Lf[y0:y1,x0:x1]; lr=Lr[y0:y1,x0:x1]
                # vertical profile = mean luminance across the silhouette-x band per row, masked to silhouette rows
                prof_f=np.array([lf[i,band][S[i,band]].mean() if S[i,band].any() else np.nan for i in range(S.shape[0])])
                prof_r=np.array([lr[i,band][S[i,band]].mean() if S[i,band].any() else np.nan for i in range(S.shape[0])])
                # board estimate: median luminance of the surround OUTSIDE the silhouette x-band
                bm=np.full(S.shape,True); bm[:, max(0,xc0-2):xc1+3]=False
                board=np.median(lf[bm]) if bm.any() else 180.0
                ys=np.where(S.any(1))[0]; top,bot=ys.min(),ys.max(); ht=bot-top+1
                pf=prof_f[top:bot+1]; pr=prof_r[top:bot+1]
                pf=np.where(np.isnan(pf),board,pf); pr=np.where(np.isnan(pr),board,pr)
                rf,_=lobes(pf,board); rr,_=lobes(pr,board)
                cnt[p]+=1
                # double: gen has >=2 lobes AND more lobes than real
                if rf>=2 and rf>rr: dbl[p]+=1
                # vanishing: gen top third near board while real top is dark
                tt=max(1,int(ht*0.33))
                gen_top=pf[:tt].mean(); real_top=pr[:tt].mean(); gen_bot=pf[-tt:].mean()
                if (board-gen_top)< 0.4*(board-gen_bot) and (board-real_top)>0.5*(board-pr[-tt:].mean()+1e-6) and (board-real_top)>15:
                    van[p]+=1
    def agg(keys):
        n=sum(cnt[k] for k in keys); d=sum(dbl[k] for k in keys); v=sum(van[k] for k in keys)
        return n,d,v
    print(f"=== {tag} ===")
    print(f"{'piece':>6} {'n':>4} {'double%':>8} {'vanish%':>8}")
    for p in ["P","N","B","R","Q","K","p","n","b","r","q","k"]:
        if cnt[p]: print(f"{p:>6} {cnt[p]:>4} {100*dbl[p]/cnt[p]:>7.1f}% {100*van[p]/cnt[p]:>7.1f}%")
    for label,keys in [("PAWNS",["P","p"]),("WHITE",list("PNBRQK")),("BLACK",list("pnbrqk")),("ALL",list("PNBRQKpnbrqk"))]:
        n,d,v=agg(keys)
        if n: print(f"{label:>6} {n:>4} {100*d/n:>7.1f}% {100*v/n:>7.1f}%")

if __name__=="__main__":
    fake,real,ds=sys.argv[1:4]; tag=sys.argv[4] if len(sys.argv)>4 else "model"
    analyze(fake,real,ds,tag)
