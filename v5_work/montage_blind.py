"""Blind montage: 3 models anonymized A/B/C (hidden random mapping) + REAL reference.
Emits a board-level grid and a black-officer crop grid. Mapping -> blind_map.json
(NOT printed). Usage:
  python3 montage_blind.py <real_dir> <ds_dir> <silABC_dir> <silAB_dir> <noPCLS_dir>
"""
import json, os, sys, secrets
import numpy as np
from PIL import Image, ImageDraw

real_dir, ds_dir, d_silABC, d_silAB, d_noPCLS = sys.argv[1:6]
TRUE = {"silABC": d_silABC, "silAB": d_silAB, "noPCLS": d_noPCLS}
labels = json.load(open(os.path.join(ds_dir, "labels.json")))

# --- hidden random mapping A/B/C -> model name (non-deterministic) ---
names = ["silABC", "silAB", "noPCLS"]
perm = list(names)
rng = secrets.SystemRandom()
rng.shuffle(perm)
slot = {"A": perm[0], "B": perm[1], "C": perm[2]}      # slot letter -> true model
slot_dir = {k: TRUE[v] for k, v in slot.items()}
json.dump(slot, open("blind_map.json", "w"))

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
def npiece(meta):
    return sum(ch.isalpha() for ch in meta["fen"].split()[0])

files=sorted(set(os.listdir(real_dir)) & set(os.listdir(d_silABC)) &
             set(os.listdir(d_silAB)) & set(os.listdir(d_noPCLS)))
files=[f for f in files if f.endswith(".png") and f[:-4] in labels]

def header(cols, w, label_list):
    img=Image.new("RGB",(w,28),(20,20,20)); dr=ImageDraw.Draw(img)
    for i,l in enumerate(label_list):
        dr.text((i*cols+cols//2-18, 8), l, fill=(255,255,255))
    return img

# ---------- BOARD-LEVEL GRID ----------
# pick boards spanning piece counts (sparse..dense), deterministic by sorted name
cand=sorted(files, key=lambda f: npiece(labels[f[:-4]]))
picks=[cand[int(x*(len(cand)-1))] for x in (0.08,0.30,0.52,0.74,0.94)]
TS=240
cols=["REAL","A","B","C"]
def board_img(d,f):
    return Image.open(os.path.join(d,f)).convert("RGB").resize((TS,TS),Image.BILINEAR)
W=TS*4
rowimgs=[header(TS,W,cols)]
for f in picks:
    strip=Image.new("RGB",(W,TS),(0,0,0))
    strip.paste(board_img(real_dir,f),(0,0))
    strip.paste(board_img(slot_dir["A"],f),(TS,0))
    strip.paste(board_img(slot_dir["B"],f),(TS*2,0))
    strip.paste(board_img(slot_dir["C"],f),(TS*3,0))
    rowimgs.append(strip)
H=sum(im.height for im in rowimgs)
out=Image.new("RGB",(W,H),(0,0,0)); y=0
for im in rowimgs: out.paste(im,(0,y)); y+=im.height
out.save("board_blind.png")

# ---------- BLACK-OFFICER CROP GRID ----------
WIN={"q":150,"r":120,"b":150,"n":120,"k":150}
inst=[]
for f in files:
    g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
    for r in range(8):
        for c in range(8):
            p=ID2P.get(int(g[r,c]),"")
            if p in WIN: inst.append((f,r,c,p))
rng2=np.random.RandomState(7); rng2.shuffle(inst); inst=inst[:8]
DISP=200
def crop(d,f,r,c,p):
    img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
    cell=img.shape[0]//8; win=WIN[p]; h=win//2
    cy=r*cell+cell//2; cx=c*cell+cell//2
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    cr=pad[cy:cy+2*h,cx:cx+2*h]
    return np.asarray(Image.fromarray(cr).resize((DISP,DISP),Image.BILINEAR))
Wc=DISP*4
rowimgs=[header(DISP,Wc,cols)]
for f,r,c,p in inst:
    strip=Image.new("RGB",(Wc,DISP),(0,0,0))
    strip.paste(Image.fromarray(crop(real_dir,f,r,c,p)),(0,0))
    strip.paste(Image.fromarray(crop(slot_dir["A"],f,r,c,p)),(DISP,0))
    strip.paste(Image.fromarray(crop(slot_dir["B"],f,r,c,p)),(DISP*2,0))
    strip.paste(Image.fromarray(crop(slot_dir["C"],f,r,c,p)),(DISP*3,0))
    rowimgs.append(strip)
H=sum(im.height for im in rowimgs)
out=Image.new("RGB",(Wc,H),(0,0,0)); y=0
for im in rowimgs: out.paste(im,(0,y)); y+=im.height
out.save("crop_blind.png")
print("wrote board_blind.png and crop_blind.png ; mapping hidden in blind_map.json")
print("boards used:", [p[:-4] for p in picks])
