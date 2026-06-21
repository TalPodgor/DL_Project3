"""Trim loose SAM masks against the local empty-square color so only piece pixels
remain. Decisive test of whether clean cutouts are achievable. numpy/PIL, no GPU.

Reads <bank>/masks/<name>__r{r}c{c}_{P}.png (binary) + dataset real (right half of
<data>/<split>/<name>.png). Writes <out>/sprites_trim/<P>/..., <out>/stats_trim.json,
<out>/viz_trim/<name>.png. Reports square-contamination before/after.

Usage: python3 mask_trim.py --data <ds> --split train --bank <mask_bank_v1> --out <dir> --thr 20 --viz 8
"""
import argparse, json, os, re
import numpy as np
from PIL import Image
from collections import defaultdict

PIECE_TO_ID={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
def fen_grid(fen,vp):
    g=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            g[r,c]=PIECE_TO_ID[ch]; c+=1
    if vp=="black": g=np.rot90(g,2)
    return g
def opening(m):  # 3x3 erode then dilate, numpy only
    def shift_all(a):
        e=a.copy()
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            e &= np.roll(np.roll(a,dy,0),dx,1)
        return e
    def shift_any(a):
        d=a.copy()
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            d |= np.roll(np.roll(a,dy,0),dx,1)
        return d
    return shift_any(shift_all(m))
def contamination(real, m, ref):  # frac of masked px that look like the square
    if not m.any(): return 1.0
    return float((np.abs(real[m].astype(float)-ref).mean(1)<14).mean())

NAMERE=re.compile(r"^(.*)__r(\d)c(\d)_(.)\.png$")
COLORS=np.array([(230,50,50),(50,200,50),(60,120,255),(240,220,40),(230,120,30),
                 (200,60,220),(40,220,220),(255,150,180),(150,90,40),(120,255,150)],np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--split",default="train")
    ap.add_argument("--bank",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--thr",type=float,default=20.0); ap.add_argument("--viz",type=int,default=8)
    a=ap.parse_args()
    labels=json.load(open(os.path.join(a.data,"labels.json")))
    os.makedirs(os.path.join(a.out,"viz_trim"),exist_ok=True)
    maskdir=os.path.join(a.bank,"masks")
    byboard=defaultdict(list)
    for f in os.listdir(maskdir):
        mm=NAMERE.match(f)
        if mm: byboard[mm.group(1)].append((int(mm.group(2)),int(mm.group(3)),mm.group(4),f))
    st={"pieces":0,"usable_after":0,"contam_before":[],"contam_after":[],"kept_frac":[],"area_after":[],"by_type":defaultdict(lambda:[0,0])}
    boards=sorted(byboard); vizn=0
    for name in boards:
        if name not in labels: continue
        AB=np.asarray(Image.open(os.path.join(a.data,a.split,name+".png")).convert("RGB"))
        H,Wf=AB.shape[:2]; W=Wf//2; real=AB[:,W:Wf]; cell=H//8
        g=fen_grid(labels[name]["fen"],labels[name]["viewpoint"])
        emp={0:[],1:[]}
        for r in range(8):
            for c in range(8):
                if g[r,c]==0:
                    cy=r*cell+cell//2; cx=c*cell+cell//2; h=cell//4
                    emp[(r+c)%2].append(np.median(real[cy-h:cy+h,cx-h:cx+h].reshape(-1,3),0))
        sqref={k:(np.median(np.array(v),0) if v else np.array([180,180,150.])) for k,v in emp.items()}
        do_viz = vizn<a.viz; overlay=real.astype(np.float32).copy() if do_viz else None; ci=0
        for r,c,P,fn in byboard[name]:
            m=np.asarray(Image.open(os.path.join(maskdir,fn)).convert("L"))>127
            ref=sqref[(r+c)%2]
            st["contam_before"].append(contamination(real,m,ref))
            dist=np.abs(real.astype(float)-ref).mean(2)
            fg=opening((dist>a.thr)&m)
            kept=fg.sum()/max(1,m.sum()); area=fg.sum()/(cell*cell)
            st["pieces"]+=1; st["kept_frac"].append(float(kept)); st["area_after"].append(float(area))
            st["contam_after"].append(contamination(real,fg,ref))
            tt=st["by_type"][P]; tt[0]+=1
            usable = 0.05<=area<=2.0 and fg.sum()>=40
            if usable:
                st["usable_after"]+=1; tt[1]+=1
                ys,xs=np.where(fg)
                y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
                rgba=np.zeros((y1-y0,x1-x0,4),np.uint8); rgba[...,:3]=real[y0:y1,x0:x1]
                rgba[...,3]=(fg[y0:y1,x0:x1]*255).astype(np.uint8)
                od=os.path.join(a.out,"sprites_trim",P); os.makedirs(od,exist_ok=True)
                Image.fromarray(rgba).save(os.path.join(od,f"{name}__r{r}c{c}.png"))
            if do_viz:
                overlay[fg]=0.4*overlay[fg]+0.6*COLORS[ci%len(COLORS)]; ci+=1
        if do_viz:
            side=np.concatenate([real,np.full((H,6,3),20,np.uint8),overlay.clip(0,255).astype(np.uint8)],1)
            Image.fromarray(side).save(os.path.join(a.out,"viz_trim",name+".png")); vizn+=1
    cb=float(np.mean(st["contam_before"])); caf=float(np.mean(st["contam_after"]))
    out={"pieces":st["pieces"],"usable_after":st["usable_after"],
         "usable_frac":round(st["usable_after"]/max(1,st["pieces"]),3),
         "contam_before":round(cb,3),"contam_after":round(caf,3),
         "kept_frac_mean":round(float(np.mean(st["kept_frac"])),3),
         "area_after_mean":round(float(np.mean(st["area_after"])),3),
         "by_type_usable":{k:f"{v[1]}/{v[0]}" for k,v in sorted(st["by_type"].items())}}
    json.dump(out,open(os.path.join(a.out,"stats_trim.json"),"w"),indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
