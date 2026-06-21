import os,sys,json,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from build_paired_dataset_v3 import lookup_fen, RAW, BLEND, RENDER, ROOT
from rectify_and_pack import rectify
from PIL import Image, ImageDraw
reps={2:11140,4:29184,5:11884,6:16936,7:15144}
ELEVS=[25,30,35,40,45]; CAL=os.path.join(ROOT,"v3_pipeline","calib")
os.makedirs(CAL,exist_ok=True)
jobs=[]
for g,fr in reps.items():
    fen=lookup_fen(g,fr)
    for e in ELEVS:
        jobs.append(dict(fen=fen,viewpoint="white",out=os.path.join(CAL,f"g{g}_e{e}"),elev=e,dist=9,lens=35,res=768))
jf=os.path.join(CAL,"jobs.json"); json.dump(jobs,open(jf,"w"))
subprocess.run(["blender","-b",BLEND,"--python",RENDER,"--","--jobs",jf],check=True,
               stdout=subprocess.DEVNULL)
for j in jobs: rectify(j["out"],"white")
for g,fr in reps.items():
    real=Image.open(f"{RAW}/testB/game{g}_frame_{fr:06d}_white.jpg" if g==2 else
                    f"{RAW}/trainB/game{g}_frame_{fr:06d}_white.jpg").convert("RGB").resize((256,256))
    W=256*len(ELEVS)+40; H=256+30; c=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(c)
    for i,e in enumerate(ELEVS):
        syn=Image.open(os.path.join(CAL,f"g{g}_e{e}_rgbR.png")).convert("RGB").resize((256,256))
        ov=Image.blend(syn,real,0.5); x=10+i*(256+5)
        c.paste(ov,(x,22)); d.text((x,6),f"g{g} e{e}",fill=(255,0,0))
    c.save(f"/tmp/calib_g{g}.png"); print("saved",f"/tmp/calib_g{g}.png")
