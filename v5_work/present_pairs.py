"""Individual REAL|bright_silAB board pairs + one crop strip, each as its own PNG
so they render legibly inline. Usage:
  python3 present_pairs.py <real_dir> <ds_dir> <silAB_dir>
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
real_dir,ds_dir,fake_dir=sys.argv[1:4]
labels=json.load(open(os.path.join(ds_dir,"labels.json")))
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
def npiece(m): return sum(ch.isalpha() for ch in m["fen"].split()[0])
files=sorted(set(os.listdir(real_dir))&set(os.listdir(fake_dir)))
files=[f for f in files if f.endswith(".png") and f[:-4] in labels]
cand=sorted(files,key=lambda f:npiece(labels[f[:-4]]))
picks=[cand[int(x*(len(cand)-1))] for x in (0.12,0.42,0.68,0.92)]
TS=360
def lab(im,t):
    d=ImageDraw.Draw(im); d.rectangle([0,0,TS,22],fill=(20,20,20)); d.text((8,6),t,fill=(255,255,120)); return im
for i,f in enumerate(picks):
    pair=Image.new("RGB",(TS*2,TS),(0,0,0))
    R=Image.open(os.path.join(real_dir,f)).convert("RGB").resize((TS,TS),Image.BILINEAR)
    Fk=Image.open(os.path.join(fake_dir,f)).convert("RGB").resize((TS,TS),Image.BILINEAR)
    pair.paste(lab(R,"REAL  (target)"),(0,0)); pair.paste(lab(Fk,"bright_silAB  (best)"),(TS,0))
    pair.save(f"pair_{i}.png")
# crop strip: 4 pieces side by side, REAL over silAB (2 rows)
WINS={"q":160,"r":130,"b":160,"n":130,"k":160,"Q":160,"R":130,"B":160,"N":130}
rs=np.random.RandomState(2); order=list(files); rs.shuffle(order); chosen=[]
seen=set()
for f in order:
    g=fen_grid(labels[f[:-4]]["fen"],labels[f[:-4]]["viewpoint"])
    for r in range(8):
        for c in range(8):
            p=ID2P.get(int(g[r,c]),"")
            if p in WINS and p.lower() not in seen and len(chosen)<5:
                seen.add(p.lower()); chosen.append((f,r,c,p,WINS[p]))
    if len(chosen)>=5: break
DISP=200
def crop(d,f,r,c,win):
    img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
    cell=img.shape[0]//8; h=win//2; cy=r*cell+cell//2; cx=c*cell+cell//2
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    return Image.fromarray(pad[cy:cy+2*h,cx:cx+2*h]).resize((DISP,DISP),Image.BILINEAR)
strip=Image.new("RGB",(DISP*len(chosen),DISP*2+24),(15,15,15))
dr=ImageDraw.Draw(strip); dr.text((8,4),"top: REAL    bottom: bright_silAB",fill=(255,255,120))
for i,(f,r,c,p,win) in enumerate(chosen):
    strip.paste(crop(real_dir,f,r,c,win),(i*DISP,24))
    strip.paste(crop(fake_dir,f,r,c,win),(i*DISP,DISP+24))
strip.save("pair_crops.png")
print("wrote", [f"pair_{i}.png" for i in range(len(picks))], "and pair_crops.png")
print("boards:", [p[:-4] for p in picks])
