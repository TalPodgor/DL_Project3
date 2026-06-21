"""High-res 3-col piece crops: REAL | silAB (realistic+double-head) | glock (clean+washed).
Deterministic, representative piece selection (not cherry-picked worst).
Usage: python3 present_trio.py <real_dir> <silAB_dir> <glock_dir> <ds_dir> <out.png>
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
real_dir,aDir,bDir,ds_dir,outp=sys.argv[1:6]
labels=json.load(open(os.path.join(ds_dir,"labels.json")))
P={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
ID2P={v:k for k,v in P.items()}
WIN={"P":120,"p":120,"N":140,"n":140,"B":165,"b":165,"R":140,"r":140,"Q":165,"q":165,"K":165,"k":165}
def fen_grid(fen,vp):
    g=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            g[r,c]=P[ch]; c+=1
    if vp=="black": g=np.rot90(g,2)
    return g
files=sorted(set(os.listdir(real_dir))&set(os.listdir(aDir))&set(os.listdir(bDir)))
files=[f for f in files if f.endswith(".png") and f[:-4] in labels]
# representative pick: 3 black pawns, 1 knight, 1 bishop, 1 queen — first occurrences in sorted order
want=["p","p","p","n","b","q"]; picks=[]; used={}
for f in files:
    g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
    for r in range(8):
        for c in range(8):
            pc=ID2P.get(int(g[r,c]),"")
            if pc in want:
                idx=want.index(pc)
                if used.get((pc,want[:idx+1].count(pc)),False): continue
            # collect distinct (piece,occurrence) deterministically
    # simpler: gather all and filter below
# build list of all pieces then choose
allp=[]
for f in files:
    g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
    for r in range(8):
        for c in range(8):
            pc=ID2P.get(int(g[r,c]),"")
            if pc: allp.append((f,r,c,pc))
def take(pc,n):
    out=[x for x in allp if x[3]==pc][:n]; return out
picks=take("p",3)+take("n",1)+take("b",1)+take("q",1)
DISP=240; COLS=["REAL (target)","bright_silAB","glock_t1p5_train12"]; SRC=[real_dir,aDir,bDir]
def crop(d,f,r,c,pc):
    win=WIN[pc]; img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
    cell=img.shape[0]//8; h=win//2; cy=r*cell+cell//2; cx=c*cell+cell//2
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    return Image.fromarray(pad[cy:cy+2*h,cx:cx+2*h]).resize((DISP,DISP),Image.LANCZOS)
W=DISP*3; rows=[]
hd=Image.new("RGB",(W,26),(20,20,20)); d=ImageDraw.Draw(hd)
for i,l in enumerate(COLS): d.text((i*DISP+8,8),l,fill=(255,255,120))
rows.append(hd)
for f,r,c,pc in picks:
    s=Image.new("RGB",(W,DISP),(0,0,0))
    for i,dd in enumerate(SRC): s.paste(crop(dd,f,r,c,pc),(i*DISP,0))
    rows.append(s)
H=sum(x.height for x in rows); out=Image.new("RGB",(W,H),(0,0,0)); y=0
for x in rows: out.paste(x,(0,y)); y+=x.height
out.save(outp); print("wrote",outp,"pieces:",[(pc,f[:-4]) for f,r,c,pc in picks])
