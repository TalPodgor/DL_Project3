"""Crop pieces (default: pawns) from generated vs real (vs synthetic) and tile
into a montage to eyeball double-head / vanishing-head artifacts.
Usage: python3 head_crops.py <fake_dir> <real_dir> <ds_dir> <out.png> [pieces] [n]
  pieces: e.g. 'Pp' (default) or 'PpKQ'
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
WIN={"P":104,"p":104,"N":120,"n":120,"B":150,"b":150,"R":120,"r":120,
     "Q":150,"q":150,"K":150,"k":150}

fake,real,ds,out=sys.argv[1:5]
want=sys.argv[5] if len(sys.argv)>5 else "Pp"
N=int(sys.argv[6]) if len(sys.argv)>6 else 18
labels=json.load(open(os.path.join(ds,"labels.json")))
files=sorted(f for f in os.listdir(fake) if f.endswith(".png"))

DISP=int(os.environ.get("DISP","110"))
def crop(img,cy,cx,win):
    h=win//2; t=img.shape[0]
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    cyp,cxp=cy+h,cx+h
    c=pad[cyp-h:cyp+h,cxp-h:cxp+h]
    return np.asarray(Image.fromarray(c).resize((DISP,DISP),Image.BILINEAR))

# gather piece instances spread across many boards
inst=[]
for f in files:
    name=f[:-4]
    if name not in labels: continue
    meta=labels[name]
    g=fen_grid(meta["fen"],meta["viewpoint"])
    abp=os.path.join(ds,meta["split"],name+".png")
    for r in range(8):
        for c in range(8):
            p=ID2P.get(int(g[r,c]),"")
            if p in want:
                inst.append((f,name,abp,r,c,p))
# spread: take every k-th so we sample many different boards
np.random.seed(1); np.random.shuffle(inst)
# balance white/black
sel=[]
for col in ("upper","lower"):
    pool=[x for x in inst if (x[5].isupper() if col=="upper" else x[5].islower())]
    sel+=pool[:N//2]
rows=[]
for f,name,abp,r,c,p in sel:
    fk=np.asarray(Image.open(os.path.join(fake,f)).convert("RGB"))
    rl=np.asarray(Image.open(os.path.join(real,f)).convert("RGB"))
    tile=fk.shape[0]; cell=tile//8
    cy=r*cell+cell//2; cx=c*cell+cell//2
    win=WIN.get(p,104)
    syn=None
    if os.path.exists(abp):
        ab=np.asarray(Image.open(abp).convert("RGB")); w2=ab.shape[1]//2
        syn=crop(ab[:,:w2],cy,cx,win)
    gc=crop(fk,cy,cx,win); rc=crop(rl,cy,cx,win)
    sep=np.full((DISP,4,3),255,np.uint8)
    parts=[]
    if syn is not None: parts+= [syn,sep]
    parts+=[gc,sep,rc]
    row=np.concatenate(parts,axis=1)
    # label strip
    lab=np.full((14,row.shape[1],3),255,np.uint8)
    rows.append((p,np.concatenate([lab,row],axis=0)))
# tile N instances per montage-row
per=int(os.environ.get("PER","3"))
grid=[]
for i in range(0,len(rows),per):
    chunk=[rows[j][1] for j in range(i,min(i+per,len(rows)))]
    wmax=max(x.shape[1] for x in chunk)
    chunk=[np.pad(x,((0,0),(0,wmax-x.shape[1]),(0,0)),constant_values=255) for x in chunk]
    bigsep=np.full((chunk[0].shape[0],8,3),0,np.uint8)
    r=chunk[0]
    for x in chunk[1:]: r=np.concatenate([r,bigsep,x],axis=1)
    grid.append(r)
wmax=max(x.shape[1] for x in grid)
grid=[np.pad(x,((0,0),(0,wmax-x.shape[1]),(0,0)),constant_values=255) for x in grid]
rowsep=np.full((10,wmax,3),0,np.uint8)
big=grid[0]
for x in grid[1:]: big=np.concatenate([big,rowsep,x],axis=0)
Image.fromarray(big).save(out)
order = "[syn|gen|real]" if syn is not None else "[gen|real]"
print(f"saved {len(rows)} crops {order} pieces={want} -> {out}")
print("legend per cell, left->right:", order)
