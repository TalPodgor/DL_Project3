"""3-way crop comparison: [modelA | modelB | real] for given pieces.
Usage: python3 compare3.py <fakeA_dir> <fakeB_dir> <real_dir> <ds_dir> <out.png> <pieces> <n> [labelA labelB]
"""
import json, os, sys
import numpy as np
from PIL import Image
PIECE_TO_ID={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
def fen_grid(fen,vp):
    ids=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            ids[r,c]=PIECE_TO_ID[ch]; c+=1
    if vp=="black": ids=np.rot90(ids,2)
    return ids
ID2P={v:k for k,v in PIECE_TO_ID.items()}
WIN={"P":104,"p":104,"N":120,"n":120,"B":150,"b":150,"R":120,"r":120,"Q":150,"q":150,"K":150,"k":150}
fA,fB,real,ds,out,want=sys.argv[1:7]; N=int(sys.argv[7]); DISP=190
labA=sys.argv[8] if len(sys.argv)>8 else "A"; labB=sys.argv[9] if len(sys.argv)>9 else "B"
labels=json.load(open(os.path.join(ds,"labels.json")))
files=sorted(set(os.listdir(fA))&set(os.listdir(fB)))
def crop(img,cy,cx,win):
    h=win//2; pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    c=pad[cy:cy+2*h,cx:cx+2*h]; return np.asarray(Image.fromarray(c).resize((DISP,DISP),Image.BILINEAR))
inst=[]
for f in files:
    if not f.endswith(".png"): continue
    n=f[:-4]
    if n not in labels: continue
    g=fen_grid(labels[n]["fen"],labels[n]["viewpoint"])
    for r in range(8):
        for c in range(8):
            pp=ID2P.get(int(g[r,c]),"")
            if pp and pp in want: inst.append((f,r,c,pp))
np.random.seed(3); np.random.shuffle(inst); inst=inst[:N]
rows=[]
for f,r,c,p in inst:
    A=np.asarray(Image.open(os.path.join(fA,f)).convert("RGB")); B=np.asarray(Image.open(os.path.join(fB,f)).convert("RGB"))
    R=np.asarray(Image.open(os.path.join(real,f)).convert("RGB"))
    cell=A.shape[0]//8; cy=r*cell; cx=c*cell; win=WIN.get(p,104)
    sep=np.full((DISP,4,3),255,np.uint8)
    row=np.concatenate([crop(A,cy,cx,win),sep,crop(B,cy,cx,win),sep,crop(R,cy,cx,win)],axis=1)
    rows.append(row)
per=2; grid=[]
for i in range(0,len(rows),per):
    ch=rows[i:i+per]; w=max(x.shape[1] for x in ch)
    ch=[np.pad(x,((0,0),(0,w-x.shape[1]),(0,0)),constant_values=255) for x in ch]
    s=np.full((ch[0].shape[0],8,3),0,np.uint8); rr=ch[0]
    for x in ch[1:]: rr=np.concatenate([rr,s,x],axis=1)
    grid.append(rr)
w=max(x.shape[1] for x in grid); grid=[np.pad(x,((0,0),(0,w-x.shape[1]),(0,0)),constant_values=255) for x in grid]
rs=np.full((10,w,3),0,np.uint8); big=grid[0]
for x in grid[1:]: big=np.concatenate([big,rs,x],axis=0)
Image.fromarray(big).save(out)
print(f"saved -> {out}  cols per cell: [{labA} | {labB} | REAL]  pieces={want} n={len(inst)}")
