"""WHY is the 'double-head' stuck ~29-31%? Decisive test: is the excess lobe driven
by CROWDING (adjacent/overlapping neighbours = data-ceiling merging) or does it happen
on ISOLATED pieces too (a true per-piece ghost the model invents)?

Reuses the exact lobe logic from detect_head_artifacts. For each occupied piece in
silABC: gen-lobes rf, real-lobes rr, and #occupied 4-neighbours in the (oriented) grid.
"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from detect_head_artifacts import sids, fen_grid, lum, lobes, WIN, ID2P

fake=sys.argv[1]; real=sys.argv[2]; ds=sys.argv[3]
labels=json.load(open(os.path.join(ds,"labels.json")))
files=sorted(f for f in os.listdir(fake) if f.endswith(".png"))

rows=[]  # (piece, color, rf, rr, nbrs, is_double)
real_multi=0; nreal=0
for f in files:
    name=f[:-4]
    if name not in labels: continue
    m=labels[name]; segp=os.path.join(ds,m["split"],name+"_seg.png")
    if not os.path.exists(segp): continue
    Lf=lum(np.asarray(Image.open(os.path.join(fake,f)).convert("RGB"),np.float32))
    Lr=lum(np.asarray(Image.open(os.path.join(real,f)).convert("RGB"),np.float32))
    tile=Lf.shape[0]; cell=tile//8
    sil=sids(Image.open(segp)); g=fen_grid(m["fen"],m["viewpoint"])
    occ=(g>=3)
    for r in range(8):
        for c in range(8):
            pid=int(g[r,c])
            if pid<3: continue
            p=ID2P[pid]; win=WIN[p]; h=win//2
            cy=r*cell+cell//2; cx=c*cell+cell//2
            y0,y1=max(0,cy-h),min(tile,cy+h); x0,x1=max(0,cx-h),min(tile,cx+h)
            S=sil[y0:y1,x0:x1]>=3
            if S.sum()<25: continue
            lf=Lf[y0:y1,x0:x1]; lr=Lr[y0:y1,x0:x1]
            cols=np.where(S.any(0))[0]; xc0,xc1=cols.min(),cols.max(); band=slice(xc0,xc1+1)
            prof_f=np.array([lf[i,band][S[i,band]].mean() if S[i,band].any() else np.nan for i in range(S.shape[0])])
            prof_r=np.array([lr[i,band][S[i,band]].mean() if S[i,band].any() else np.nan for i in range(S.shape[0])])
            bm=np.full(S.shape,True); bm[:, max(0,xc0-2):xc1+3]=False
            board=np.median(lf[bm]) if bm.any() else 180.0
            ys=np.where(S.any(1))[0]; top,bot=ys.min(),ys.max()
            pf=np.where(np.isnan(prof_f[top:bot+1]),board,prof_f[top:bot+1])
            pr=np.where(np.isnan(prof_r[top:bot+1]),board,prof_r[top:bot+1])
            rf,_=lobes(pf,board); rr,_=lobes(pr,board)
            # occupied 4-neighbours in the oriented grid
            nbrs=0
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr2,cc2=r+dr,c+dc
                if 0<=rr2<8 and 0<=cc2<8 and occ[rr2,cc2]: nbrs+=1
            rows.append((p, "w" if p.isupper() else "b", rf, rr, nbrs, int(rf>=2 and rf>rr)))
            real_multi += int(rr>=2); nreal+=1

R=np.array([(x[2],x[3],x[4],x[5]) for x in rows],float)  # rf,rr,nbrs,dbl
def rate(mask):
    return (R[mask,3].mean()*100 if mask.sum() else 0), int(mask.sum())
print(f"pieces={len(R)}")
print(f"REAL itself has >=2 lobes: {100*real_multi/max(nreal,1):.1f}%  (inherent scene multi-lobe / adjacency)")
print(f"mean gen-lobes={R[:,0].mean():.2f}  mean real-lobes={R[:,1].mean():.2f}")
print()
print("DOUBLE-rate by #occupied neighbours:")
for k in [0,1,2,3,4]:
    rt,n=rate(R[:,2]==k)
    if n: print(f"  neighbours={k}: {rt:5.1f}%   (n={n})")
rt0,n0=rate(R[:,2]==0); rtc,nc=rate(R[:,2]>=1)
print(f"  ISOLATED (0 nbrs): {rt0:.1f}%  vs  CROWDED (>=1): {rtc:.1f}%")
print()
print("DOUBLE-rate, PAWNS only, by crowding:")
PM=np.array([x[0] in ("P","p") for x in rows])
rtp0,_=rate(PM&(R[:,2]==0)); rtpc,_=rate(PM&(R[:,2]>=1))
print(f"  pawns isolated: {rtp0:.1f}%   pawns crowded(>=1): {rtpc:.1f}%")
print()
print("DOUBLE-rate, BLACK officers (n,b,r,q,k) by crowding:")
BO=np.array([x[0] in ("n","b","r","q","k") for x in rows])
rtb0,nb0=rate(BO&(R[:,2]==0)); rtbc,nbc=rate(BO&(R[:,2]>=1))
print(f"  black-officer isolated: {rtb0:.1f}% (n={nb0})   crowded(>=1): {rtbc:.1f}% (n={nbc})")
