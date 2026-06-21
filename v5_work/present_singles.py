"""One image per piece (single row, large panels) so it renders legibly inline.
Cols: REAL | bright_silAB | glock_t1p5_train12. Pieces from DIFFERENT boards.
Usage: python3 present_singles.py <real_dir> <silAB_dir> <glock_dir> <ds_dir>
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
real_dir,aDir,bDir,ds_dir=sys.argv[1:5]
labels=json.load(open(os.path.join(ds_dir,"labels.json")))
P={"P":3,"N":4,"B":5,"R":6,"Q":7,"K":8,"p":9,"n":10,"b":11,"r":12,"q":13,"k":14}
ID2P={v:k for k,v in P.items()}
WIN={"P":120,"p":120,"N":140,"n":140,"B":170,"b":170,"R":140,"r":140,"Q":170,"q":170,"K":170,"k":170}
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
# choose distinct boards spread across the set; want a pawn from each + 1 knight + 1 queen
idxs=[int(x*(len(files)-1)) for x in (0.10,0.35,0.6,0.85)]
boards=[files[i] for i in idxs]
def find(board,wanted):
    g=fen_grid(labels[board[:-4]]["fen"],labels[board[:-4]]["viewpoint"])
    for r in range(8):
        for c in range(8):
            if ID2P.get(int(g[r,c]),"")==wanted: return (board,r,c,wanted)
    return None
picks=[]
picks.append(find(boards[0],"p"))
picks.append(find(boards[1],"p"))
picks.append(find(boards[2],"n") or find(boards[2],"p"))
picks.append(find(boards[3],"q") or find(boards[3],"b") or find(boards[3],"p"))
picks=[x for x in picks if x]
DISP=300; COLS=["REAL (target)","bright_silAB (realistic family)","glock_t1p5_train12 (clean family)"]; SRC=[real_dir,aDir,bDir]
def crop(d,f,r,c,pc):
    win=WIN[pc]; img=np.asarray(Image.open(os.path.join(d,f)).convert("RGB"))
    cell=img.shape[0]//8; h=win//2; cy=r*cell+cell//2; cx=c*cell+cell//2
    pad=np.pad(img,((h,h),(h,h),(0,0)),mode="edge")
    return Image.fromarray(pad[cy:cy+2*h,cx:cx+2*h]).resize((DISP,DISP),Image.LANCZOS)
for i,(f,r,c,pc) in enumerate(picks):
    W=DISP*3; img=Image.new("RGB",(W,DISP+24),(20,20,20)); d=ImageDraw.Draw(img)
    for k,l in enumerate(COLS): d.text((k*DISP+8,7),l,fill=(255,255,120))
    for k,dd in enumerate(SRC): img.paste(crop(dd,f,r,c,pc),(k*DISP,24))
    img.save(f"single_{i}.png")
print("wrote", [f"single_{i}.png" for i in range(len(picks))], "pieces:", [(pc,f[:-4]) for f,r,c,pc in picks])
