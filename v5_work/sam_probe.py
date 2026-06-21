"""Feasibility probe: SAM point-prompted at FEN-known squares on REAL boards.
Tests whether SAM gives clean per-piece masks on 40-60px soft pieces.
Usage: python3 sam_probe.py --reals <dir> --labels <labels.json> --ckpt <vit_b.pth> --out <dir> --n 3
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

P={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
ID2P={v:k for k,v in P.items()}
def fen_grid(fen,vp):
    g=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            g[r,c]=P[ch]; c+=1
    if vp=="black": g=np.rot90(g,2)
    return g
COLORS=np.array([(230,50,50),(50,200,50),(60,120,255),(240,220,40),(230,120,30),
                 (200,60,220),(40,220,220),(255,150,180),(150,90,40),(120,255,150)],np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reals",required=True); ap.add_argument("--labels",required=True)
    ap.add_argument("--ckpt",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--num",type=int,default=3); ap.add_argument("--boards",default="")
    a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    labels=json.load(open(a.labels))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sam=sam_model_registry["vit_b"](checkpoint=a.ckpt).to(dev); sam.eval()
    pred=SamPredictor(sam)
    avail=[f for f in sorted(os.listdir(a.reals)) if f.endswith(".png") and f[:-4] in labels]
    names=[n.strip() for n in a.boards.split(",") if n.strip()] if a.boards else None
    if names: avail=[f for f in avail if f[:-4] in names]
    avail=avail[:a.num]
    print("device",dev,"| boards:",[f[:-4] for f in avail])
    for f in avail:
        img=np.asarray(Image.open(os.path.join(a.reals,f)).convert("RGB"))
        H=img.shape[0]; cell=H//8
        pred.set_image(img)
        g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
        overlay=img.copy().astype(np.float32); ci=0; areas=[]
        for r in range(8):
            for c in range(8):
                if g[r,c]==0: continue
                cx=c*cell+cell//2; cy=r*cell+cell//2
                pts=np.array([[cx,cy],[cx,cy-int(0.45*cell)]]); lab=np.array([1,1])
                masks,scores,_=pred.predict(point_coords=pts,point_labels=lab,multimask_output=True)
                # choose smallest-area mask that contains the upper (piece-body) point, else max score
                up=(int(cy-0.45*cell),cx)
                cand=[(m.sum(),m) for m in masks if m[max(0,up[0]),up[1]]]
                if cand: _,m=min(cand,key=lambda t:t[0])
                else: m=masks[int(np.argmax(scores))]
                areas.append(int(m.sum())/ (cell*cell))
                col=COLORS[ci%len(COLORS)]; ci+=1
                overlay[m]=0.45*overlay[m]+0.55*col
        side=np.concatenate([img,overlay.clip(0,255).astype(np.uint8)],axis=1)
        Image.fromarray(side).save(os.path.join(a.out,f"sammask_{f}"))
        ar=np.array(areas)
        print(f"{f[:-4]}: pieces={len(areas)} mask_area/cell mean={ar.mean():.2f} min={ar.min():.2f} max={ar.max():.2f}")
    print("wrote overlays to",a.out)

if __name__=="__main__": main()
