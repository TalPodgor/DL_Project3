"""Full-board triptychs [synthetic | generated | real], stacked.
Usage: python3 board_triptych.py <fake_dir> <real_dir> <ds_dir> <out.png> [names_csv]
If names omitted, auto-picks a dense, a mid, and a sparse board.
"""
import json, os, sys
import numpy as np
from PIL import Image

fake,real,ds,out=sys.argv[1:5]
labels=json.load(open(os.path.join(ds,"labels.json")))
files=set(os.listdir(fake))
have=[(n,labels[n]) for n in (f[:-4] for f in files if f.endswith(".png")) if n in labels]
if len(sys.argv)>5:
    names=sys.argv[5].split(",")
else:
    have.sort(key=lambda x:x[1].get("pieces",0))
    sparse=have[2][0]; mid=have[len(have)//2][0]; dense=have[-3][0]
    names=[dense,mid,sparse]
S=300
def load(d,n):
    return np.asarray(Image.open(os.path.join(d,n+".png")).convert("RGB").resize((S,S)))
rows=[]
for n in names:
    meta=labels[n]; abp=os.path.join(ds,meta["split"],n+".png")
    ab=np.asarray(Image.open(abp).convert("RGB")); w2=ab.shape[1]//2
    syn=np.asarray(Image.fromarray(ab[:,:w2]).resize((S,S)))
    gen=load(fake,n); rl=load(real,n)
    sep=np.full((S,4,3),255,np.uint8)
    row=np.concatenate([syn,sep,gen,sep,rl],axis=1)
    lab=np.full((16,row.shape[1],3),255,np.uint8)
    rows.append(np.concatenate([lab,row],axis=0))
    print(f"{n}  pieces={meta.get('pieces')}  [syn|gen|real]")
rowsep=np.full((6,rows[0].shape[1],3),0,np.uint8)
big=rows[0]
for r in rows[1:]: big=np.concatenate([big,rowsep,r],axis=0)
Image.fromarray(big).save(out)
print("saved ->",out,"  columns: SYNTHETIC | GENERATED(silABC) | REAL")
