"""Build a per-piece REAL mask/sprite bank with SAM, box-prompted from the synthetic
seg silhouette (aligned) + negative neighbor points + a contrast gate.

Dataset layout: <data>/<split>/<name>.png = [synthetic | real] side-by-side;
<name>_seg.png = RGB semantic silhouette (tile-sized). labels.json gives fen/viewpoint/split.

Usage:
  python3 mask_bank.py --data <dataset_dir> --split train --ckpt <pth> --model vit_b \
     --out <out_dir> --limit 0 --viz 6
Outputs: <out>/masks/<name>__r{r}c{c}_{P}.png (binary), <out>/sprites/<P>/<...>.png (RGBA),
         <out>/viz/<name>.png (overlay), <out>/stats.json
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

PIECE_TO_ID={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
ID2P={v:k for k,v in PIECE_TO_ID.items()}
CLASS_COLORS_LINEAR={"P":(0.92,0.88,0.72),"N":(0.82,0.82,0.52),"B":(0.92,0.72,0.44),
 "R":(0.72,0.92,0.72),"Q":(0.72,0.82,0.98),"K":(0.90,0.70,0.98),"p":(0.18,0.12,0.08),
 "n":(0.32,0.18,0.08),"b":(0.20,0.32,0.12),"r":(0.10,0.24,0.34),"q":(0.34,0.12,0.30),"k":(0.34,0.10,0.10)}
def l2s(v):
    v=np.asarray(v,np.float32); s=np.where(v<=0.0031308,12.92*v,1.055*np.power(v,1/2.4)-0.055)
    return np.clip(np.round(s*255),0,255).astype(np.int16)
PALETTE=np.stack([l2s(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID]); PIDS=np.array([PIECE_TO_ID[p] for p in PIECE_TO_ID],np.uint8)
def seg_ids(seg,thr=75,br=30):
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
COLORS=np.array([(230,50,50),(50,200,50),(60,120,255),(240,220,40),(230,120,30),
                 (200,60,220),(40,220,220),(255,150,180),(150,90,40),(120,255,150)],np.uint8)

def seg_box(ids, P, r, c, cell, H, W):
    """bbox prior for piece at (r,c) from seg pixels of that type in a local window."""
    pid=PIECE_TO_ID[P]
    x0w=int(max(0,(c-0.9)*cell)); x1w=int(min(W,(c+1.9)*cell))
    y0w=int(max(0,(r-1.7)*cell)); y1w=int(min(H,(r+1.1)*cell))
    sub=ids[y0w:y1w, x0w:x1w]
    ys,xs=np.where(sub==pid)
    if len(xs)>=8:
        bx0=x0w+xs.min(); bx1=x0w+xs.max(); by0=y0w+ys.min(); by1=y0w+ys.max()
    else:  # fallback: generic upright box rooted at the cell
        bx0=int((c+0.12)*cell); bx1=int((c+0.88)*cell)
        by0=int((r-1.1)*cell);  by1=int((r+0.95)*cell)
    pad=int(0.08*cell)
    return [max(0,bx0-pad),max(0,by0-pad),min(W,bx1+pad),min(H,by1+pad)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--split",default="train")
    ap.add_argument("--ckpt",required=True); ap.add_argument("--model",default="vit_b")
    ap.add_argument("--out",required=True); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--viz",type=int,default=6)
    a=ap.parse_args()
    labels=json.load(open(os.path.join(a.data,"labels.json")))
    for sub in ("masks","sprites","viz"): os.makedirs(os.path.join(a.out,sub),exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sam=sam_model_registry[a.model](checkpoint=a.ckpt).to(dev); sam.eval()
    pred=SamPredictor(sam)
    sdir=os.path.join(a.data,a.split)
    names=[f[:-4] for f in sorted(os.listdir(sdir)) if f.endswith(".png")
           and not f.endswith("_seg.png") and not f.endswith("_depth.png") and f[:-4] in labels]
    if a.limit>0: names=names[:a.limit]
    stats={"n_boards":0,"n_pieces":0,"accepted":0,"rej_area":0,"rej_contrast":0,
           "by_type":{}, "accept_by_color":{"white":0,"black":0},"total_by_color":{"white":0,"black":0}}
    for bi,name in enumerate(names):
        AB=np.asarray(Image.open(os.path.join(sdir,name+".png")).convert("RGB"))
        H,Wf=AB.shape[:2]; W=Wf//2; real=AB[:, W:Wf].copy()
        seg=Image.open(os.path.join(sdir,name+"_seg.png")).convert("RGB").resize((W,H),Image.NEAREST)
        ids=seg_ids(seg); cell=H//8
        g=fen_grid(labels[name]["fen"],labels[name]["viewpoint"])
        pred.set_image(real)
        # empty-square reference colors by parity
        emp={0:[],1:[]}
        for r in range(8):
            for c in range(8):
                if g[r,c]==0:
                    cy=r*cell+cell//2; cx=c*cell+cell//2; h=cell//4
                    emp[(r+c)%2].append(np.median(real[cy-h:cy+h,cx-h:cx+h].reshape(-1,3),0))
        sqref={k:(np.median(np.array(v),0) if v else np.array([180,180,150.])) for k,v in emp.items()}
        overlay=real.astype(np.float32); ci=0
        for r in range(8):
            for c in range(8):
                if g[r,c]==0: continue
                P=ID2P[int(g[r,c])]; color="black" if P.islower() else "white"
                stats["n_pieces"]+=1; stats["total_by_color"][color]+=1
                st=stats["by_type"].setdefault(P,{"total":0,"acc":0}); st["total"]+=1
                box=np.array(seg_box(ids,P,r,c,cell,H,W))
                negs=[]
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr,cc=r+dr,c+dc
                    if 0<=rr<8 and 0<=cc<8: negs.append([cc*cell+cell//2, rr*cell+cell//2])
                pc=np.array(negs) if negs else None; pl=np.zeros(len(negs)) if negs else None
                masks,scores,_=pred.predict(point_coords=pc,point_labels=pl,box=box,multimask_output=False)
                m=masks[0]; area=m.sum()/(cell*cell)
                pcol=np.median(real[m],0) if m.any() else np.array([0,0,0.])
                ref=sqref[(r+c)%2]; contrast=float(np.abs(pcol.astype(float)-ref).mean())
                bg_frac=float((np.abs(real[m].astype(float)-ref).mean(1)<14).mean()) if m.any() else 1.0
                ok=True
                if not (0.08<=area<=2.4): ok=False; stats["rej_area"]+=1
                elif bg_frac>0.6 and contrast<18: ok=False; stats["rej_contrast"]+=1
                if ok:
                    stats["accepted"]+=1; st["acc"]+=1; stats["accept_by_color"][color]+=1
                    ys,xs=np.where(m)
                    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
                    rgba=np.zeros((y1-y0,x1-x0,4),np.uint8); rgba[...,:3]=real[y0:y1,x0:x1]
                    rgba[...,3]=(m[y0:y1,x0:x1]*255).astype(np.uint8)
                    od=os.path.join(a.out,"sprites",P); os.makedirs(od,exist_ok=True)
                    Image.fromarray(rgba).save(os.path.join(od,f"{name}__r{r}c{c}.png"))
                    Image.fromarray((m*255).astype(np.uint8)).save(os.path.join(a.out,"masks",f"{name}__r{r}c{c}_{P}.png"))
                    col=COLORS[ci%len(COLORS)]; overlay[m]=0.45*overlay[m]+0.55*col
                else:
                    overlay[m]=0.6*overlay[m]+0.4*np.array([255,255,255]) # rejected -> white wash
                ci+=1
        stats["n_boards"]+=1
        if bi<a.viz:
            side=np.concatenate([real, np.full((H,6,3),20,np.uint8), overlay.clip(0,255).astype(np.uint8)],axis=1)
            Image.fromarray(side).save(os.path.join(a.out,"viz",name+".png"))
    cov=stats["accepted"]/max(1,stats["n_pieces"])
    stats["coverage"]=round(cov,4)
    json.dump(stats,open(os.path.join(a.out,"stats.json"),"w"),indent=2)
    print(f"device={dev} model={a.model} boards={stats['n_boards']} pieces={stats['n_pieces']} "
          f"accepted={stats['accepted']} coverage={cov:.3f} rej_area={stats['rej_area']} rej_contrast={stats['rej_contrast']}")
    print("by_type:", {k:f"{v['acc']}/{v['total']}" for k,v in sorted(stats['by_type'].items())})
    print("by_color accept:", stats["accept_by_color"], "of", stats["total_by_color"])

if __name__=="__main__": main()
