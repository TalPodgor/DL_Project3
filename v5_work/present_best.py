"""Presentation montage (NOT blind): REAL | bright_silAB (new best) | bright_silABC (old).
Boards + black-officer crops. Usage:
  python3 present_best.py <real_dir> <ds_dir> <silAB_dir> <silABC_dir>
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

real_dir, ds_dir, d_silAB, d_silABC = sys.argv[1:5]
labels=json.load(open(os.path.join(ds_dir,"labels.json")))
PIECE_TO_ID={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
ID2P={v:k for k,v in PIECE_TO_ID.items()}
def fen_grid(fen,vp):
    ids=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            ids[r,c]=PIECE_TO_ID[ch]; c+=1
    if vp=="black": ids=np.rot90(ids,2)
    return ids
def npiece(m): return sum(ch.isalpha() for ch in m["fen"].split()[0])
files=sorted(set(os.listdir(real_dir))&set(os.listdir(d_silAB))&set(os.listdir(d_silABC)))
files=[f for f in files if f.endswith(".png") and f[:-4] in labels]
COLS=["REAL  (target)","bright_silAB  (NEW best)","bright_silABC  (old)"]
SRC=[real_dir,d_silAB,d_silABC]
def header(cw,w):
    im=Image.new("RGB",(w,30),(25,25,25)); d=ImageDraw.Draw(im)
    for i,l in enumerate(COLS): d.text((i*cw+10,9),l,fill=(255,255,120))
    return im

# ---- boards: 5 spanning sparse..dense ----
TS=330; W=TS*3
cand=sorted(files,key=lambda f:npiece(labels[f[:-4]]))
picks=[cand[int(x*(len(cand)-1))] for x in (0.10,0.34,0.55,0.78,0.95)]
def bimg(d,f): return Image.open(os.path.join(d,f)).convert("RGB").resize((TS,TS),Image.BILINEAR)
rows=[header(TS,W)]
for f in picks:
    s=Image.new("RGB",(W,TS),(0,0,0))
    for i,d in enumerate(SRC): s.paste(bimg(d,f),(i*TS,0))
    rows.append(s)
H=sum(r.height for r in rows); out=Image.new("RGB",(W,H),(0,0,0)); y=0
for r in rows: out.paste(r,(0,y)); y+=r.height
out.save("present_boards.png")

# ---- black-officer crops ----
WINS={"q":150,"r":120,"b":150,"n":120,"k":150}
inst=[]; rs=np.random.RandomState(5); order=list(files); rs.shuffle(order)
seen=set()
for f in order:
    g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
    for r in range(8):
        for c in range(8):
            p=ID2P.get(int(g[r,c]),"")
            if p in WINS and p not in seen:
                seen.add(p); inst.append((f,r,c,p,WINS[p]))
    if len(inst)>=5: break
# add a crowded window for merge demo
for f in order:
    g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
    done=False
    for r in range(8):
        for c in range(1,7):
            if g[r,c] and g[r,c-1] and g[r,c+1]:
                inst.append((f,r,c,"crowd",210)); done=True; break
        if done: break
    if done: break
DISP=250; Wc=DISP*3
def crop(d,f,r,c,win):
    img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
    cell=img.shape[0]//8; h=win//2; cy=r*cell+cell//2; cx=c*cell+cell//2
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    return Image.fromarray(pad[cy:cy+2*h,cx:cx+2*h]).resize((DISP,DISP),Image.BILINEAR)
rows=[header(DISP,Wc)]
for f,r,c,p,win in inst:
    s=Image.new("RGB",(Wc,DISP),(0,0,0))
    for i,d in enumerate(SRC): s.paste(crop(d,f,r,c,win),(i*DISP,0))
    rows.append(s)
H=sum(r.height for r in rows); out=Image.new("RGB",(Wc,H),(0,0,0)); y=0
for r in rows: out.paste(r,(0,y)); y+=r.height
out.save("present_crops.png")
print("wrote present_boards.png, present_crops.png")
print("crops:", [(p, f[:-4]) for f,r,c,p,win in inst])
