"""Round-2 blind eval set for INDEPENDENT judges. Fresh hidden mapping X/Y/Z.
Montages -> blind_eval_round2/  ; key -> .blind_round2_key.json (OUTSIDE eval dir).
Usage: python3 blind_round2.py <real_dir> <ds_dir> <silABC_dir> <silAB_dir> <noPCLS_dir>
"""
import json, os, sys, secrets
import numpy as np
from PIL import Image, ImageDraw

real_dir, ds_dir, d_silABC, d_silAB, d_noPCLS = sys.argv[1:6]
TRUE={"silABC":d_silABC,"silAB":d_silAB,"noPCLS":d_noPCLS}
EVAL="blind_eval_round2"; os.makedirs(EVAL,exist_ok=True)
names=["silABC","silAB","noPCLS"]; rng=secrets.SystemRandom(); rng.shuffle(names)
slot={"X":names[0],"Y":names[1],"Z":names[2]}
slot_dir={k:TRUE[v] for k,v in slot.items()}
json.dump(slot, open(".blind_round2_key.json","w"))   # OUTSIDE eval dir

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
files=sorted(set(os.listdir(real_dir))&set(os.listdir(d_silABC))&
             set(os.listdir(d_silAB))&set(os.listdir(d_noPCLS)))
files=[f for f in files if f.endswith(".png") and f[:-4] in labels]
COLS=["REAL","X","Y","Z"]
def hdr(cw,w):
    im=Image.new("RGB",(w,30),(20,20,20)); d=ImageDraw.Draw(im)
    for i,l in enumerate(COLS): d.text((i*cw+cw//2-16,9),l,fill=(255,255,120))
    return im

# board montage: 6 boards spanning sparse..dense
cand=sorted(files,key=lambda f:npiece(labels[f[:-4]]))
picks=[cand[int(x*(len(cand)-1))] for x in (0.05,0.25,0.45,0.62,0.80,0.95)]
TS=260; W=TS*4
def bimg(d,f): return Image.open(os.path.join(d,f)).convert("RGB").resize((TS,TS),Image.BILINEAR)
rows=[hdr(TS,W)]
for f in picks:
    s=Image.new("RGB",(W,TS),(0,0,0))
    for i,col in enumerate(["REAL","X","Y","Z"]):
        s.paste(bimg(real_dir if col=="REAL" else slot_dir[col],f),(i*TS,0))
    rows.append(s)
H=sum(r.height for r in rows); out=Image.new("RGB",(W,H),(0,0,0)); y=0
for r in rows: out.paste(r,(0,y)); y+=r.height
out.save(os.path.join(EVAL,"boards.png"))

# hero 2x2 crops: REAL/X/Y/Z, diverse black officers + crowded windows
WINS={"q":150,"r":120,"b":150,"n":120,"k":150}
singles={}; crowded=[]
rs=np.random.RandomState(23); order=list(files); rs.shuffle(order)
for f in order:
    g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
    for r in range(8):
        for c in range(8):
            p=ID2P.get(int(g[r,c]),"")
            if p in WINS and p not in singles: singles[p]=(f,r,c,150)
    for r in range(8):
        for c in range(1,7):
            if g[r,c] and g[r,c-1] and g[r,c+1] and len(crowded)<3: crowded.append((f,r,c,210))
    if len(singles)>=5 and len(crowded)>=3: break
items=[singles[p] for p in "qrbnk" if p in singles]+crowded
DISP=300
def crop(d,f,r,c,win):
    img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
    cell=img.shape[0]//8; h=win//2; cy=r*cell+cell//2; cx=c*cell+cell//2
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    return Image.fromarray(pad[cy:cy+2*h,cx:cx+2*h]).resize((DISP,DISP),Image.BILINEAR)
def hero(f,r,c,win):
    g=Image.new("RGB",(DISP*2,DISP*2+24),(15,15,15)); dr=ImageDraw.Draw(g)
    cells=[("REAL",real_dir,0,24),("X",slot_dir["X"],DISP,24),
           ("Y",slot_dir["Y"],0,DISP+24),("Z",slot_dir["Z"],DISP,DISP+24)]
    for lab,src,px,py in cells:
        g.paste(crop(src,f,r,c,win),(px,py)); dr.text((px+6,py+3),lab,fill=(255,255,120))
    return g
for i,(f,r,c,win) in enumerate(items):
    hero(f,r,c,win).save(os.path.join(EVAL,f"hero_{i}.png"))
print("eval set:", sorted(os.listdir(EVAL)))
print("n_heroes:", len(items))
