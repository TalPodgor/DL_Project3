"""Measure empty dark/light square color: REAL vs silAB vs synthetic-input.
Usage: python3 board_color_diag.py <real_dir> <silAB_dir> <realA_dir> <ds_dir>
realA_dir = synthetic input (results .../images/real_A)
"""
import json, os, sys
import numpy as np
from PIL import Image
real_dir,fake_dir,inA_dir,ds_dir=sys.argv[1:5]
labels=json.load(open(os.path.join(ds_dir,"labels.json")))
P={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
def fen_grid(fen,vp):
    g=np.zeros((8,8),np.uint8)
    for r,row in enumerate(fen.split()[0].split("/")):
        c=0
        for ch in row:
            if ch.isdigit(): c+=int(ch); continue
            g[r,c]=P[ch]; c+=1
    if vp=="black": g=np.rot90(g,2)
    return g
def empties(d, files, limit=60):
    dark=[]; light=[]
    for f in files[:limit]:
        n=f[:-4]
        if n not in labels: continue
        img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"),np.float32)
        tile=img.shape[0]; cell=tile//8
        g=fen_grid(labels[n]["fen"],labels[n]["viewpoint"])
        for r in range(8):
            for c in range(8):
                if g[r,c]!=0: continue
                y=r*cell+cell//2; x=c*cell+cell//2; h=cell//4
                patch=img[y-h:y+h,x-h:x+h].reshape(-1,3)
                med=np.median(patch,0)
                (dark if (r+c)%2==0 else light).append(med)
    dark=np.array(dark); light=np.array(light)
    # decide which parity is actually darker by luminance
    if dark.mean()> light.mean(): dark,light=light,dark
    return dark.mean(0), light.mean(0)
files=sorted(set(os.listdir(real_dir))&set(os.listdir(fake_dir)))
files=[f for f in files if f.endswith(".png")]
print(f"{'source':16s} {'DARK sq (R,G,B)':>22s}  warmth(R-B)   {'LIGHT sq (R,G,B)':>22s}")
for name,d in [("REAL target",real_dir),("bright_silAB",fake_dir),("synthetic in",inA_dir)]:
    if not os.path.isdir(d): print(f"{name:16s} (dir missing: {d})"); continue
    dk,lt=empties(d,files)
    print(f"{name:16s} {str(tuple(dk.round(1))):>22s}   {dk[0]-dk[2]:+6.1f}     {str(tuple(lt.round(1))):>22s}")
