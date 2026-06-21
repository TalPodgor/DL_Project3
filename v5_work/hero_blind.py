"""High-res 2x2 blind hero crops (REAL / A / B / C) reusing blind_map.json mapping.
One PNG per piece -> hero_*.png. Usage:
  python3 hero_blind.py <real_dir> <ds_dir> <silABC_dir> <silAB_dir> <noPCLS_dir>
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

real_dir, ds_dir, d_silABC, d_silAB, d_noPCLS = sys.argv[1:6]
TRUE={"silABC":d_silABC,"silAB":d_silAB,"noPCLS":d_noPCLS}
slot=json.load(open("blind_map.json"))            # slot letter -> true model (hidden)
slot_dir={k:TRUE[v] for k,v in slot.items()}
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
files=sorted(set(os.listdir(real_dir))&set(os.listdir(d_silABC))&
             set(os.listdir(d_silAB))&set(os.listdir(d_noPCLS)))
files=[f for f in files if f.endswith(".png") and f[:-4] in labels]

# find one fairly-isolated instance per target black officer, + two crowded windows
def grid_of(f): return fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
singles={}
crowded=[]
rs=np.random.RandomState(11); order=list(files); rs.shuffle(order)
for f in order:
    g=grid_of(f)
    for r in range(8):
        for c in range(8):
            p=ID2P.get(int(g[r,c]),"")
            if p in "qrbnk" and p not in singles:
                singles[p]=(f,r,c,p,150)
    # crowded: a cell with >=2 occupied neighbors in same rank
    for r in range(8):
        for c in range(1,7):
            if g[r,c] and g[r,c-1] and g[r,c+1] and len(crowded)<2:
                crowded.append((f,r,c,"x",210))
    if len(singles)>=5 and len(crowded)>=2: break

items=[singles[p] for p in "qrbnk" if p in singles]+crowded
DISP=300
def crop(d,f,r,c,win):
    img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
    cell=img.shape[0]//8; h=win//2
    cy=r*cell+cell//2; cx=c*cell+cell//2
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    cr=pad[cy:cy+2*h,cx:cx+2*h]
    return Image.fromarray(cr).resize((DISP,DISP),Image.BILINEAR)
def panel(f,r,c,win):
    lab=["REAL","A","B","C"]; srcs=[real_dir,slot_dir["A"],slot_dir["B"],slot_dir["C"]]
    g=Image.new("RGB",(DISP*2,DISP*2+22),(15,15,15)); dr=ImageDraw.Draw(g)
    pos=[(0,22),(DISP,22),(0,DISP+22),(DISP,DISP+22)]
    for (px,py),l,s in zip(pos,lab,srcs):
        g.paste(crop(s,f,r,c,win),(px,py))
        dr.text((px+6,py+2),l,fill=(255,255,80))
    return g
for i,(f,r,c,p,win) in enumerate(items):
    panel(f,r,c,win).save(f"hero_{i}_{p}.png")
print("wrote", [f"hero_{i}_{it[3]}.png" for i,it in enumerate(items)])
