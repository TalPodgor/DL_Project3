"""
render_aligned.py -- Blender headless renderer for FEN-conditioned chess scenes.

Opens ChessScene.blend (MIT low-poly set), and for each job places pieces per FEN
on the existing board, sets a parameterised oblique camera, and renders:
  <out>_rgb.png    simple lit render (synthetic conditioning input A)
  <out>_mask.png   flat per-class colours (semantic mask M)
  <out>_corners.json  playing-surface corner pixels (for board rectification)

Single job:
  blender -b assets/ChessScene.blend --python render_aligned.py -- \
     --fen "<FEN>" --viewpoint white --out renders/x --elev 35 --dist 9 --lens 35

Batch (one process, many renders -- much faster):
  blender -b assets/ChessScene.blend --python render_aligned.py -- --jobs jobs.json
where jobs.json = [{"fen":..., "viewpoint":"white", "out":"renders/x",
                    "elev":35,"dist":9,"lens":35,"yaw":0}, ...]
"""
import bpy, sys, math, argparse, json
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view

# ---------------- args ----------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--jobs")
ap.add_argument("--fen"); ap.add_argument("--viewpoint", choices=["white", "black"], default="white")
ap.add_argument("--out")
ap.add_argument("--elev", type=float, default=35.0); ap.add_argument("--dist", type=float, default=9.0)
ap.add_argument("--lens", type=float, default=35.0); ap.add_argument("--yaw", type=float, default=0.0)
ap.add_argument("--res", type=int, default=768)
A = ap.parse_args(argv)

REF = {"p": "White Pawn.000", "r": "LP Rook", "n": "Knight",
       "b": "LP Bishop 1", "q": "LP Queen", "k": "LP King"}
CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for i, t in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + t] = 3 + i
    CLASS_ID["b" + t] = 9 + i
def palette(cid):
    return ((cid * 17) % 256 / 255.0, (cid * 53) % 256 / 255.0, (cid * 97) % 256 / 255.0)

# ---------------- one-time scene setup ----------------
def world_bbox(obj):
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
    mx = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
    return mn, mx

board_surface = bpy.data.objects["Chess Board.001"]
bmn, bmx = world_bbox(board_surface)
BOARD_MIN = Vector((bmn.x, bmn.y)); BOARD_W = bmx.x - bmn.x
SQ = BOARD_W / 8.0; BOARD_TOP = bmx.z
CENTER = Vector(((bmn.x + bmx.x) / 2, (bmn.y + bmx.y) / 2, BOARD_TOP))
ORIG_BOARD_MAT = list(board_surface.data.materials)
print(f"[board] min={tuple(round(v,3) for v in bmn)} W={BOARD_W:.3f} SQ={SQ:.3f} top={BOARD_TOP:.3f}")

def flat_mat(name, rgb, emission=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    sh = nt.nodes.new("ShaderNodeEmission" if emission else "ShaderNodeBsdfDiffuse")
    sh.inputs[0].default_value = (*rgb, 1)
    nt.links.new(sh.outputs[0], out.inputs[0]); return m

WHITE_MAT = flat_mat("pc_white", (0.85, 0.78, 0.62))
BLACK_MAT = flat_mat("pc_black", (0.12, 0.09, 0.07))
MASK_BOARD = flat_mat("m_board", palette(CLASS_ID["empty_light"]), emission=True)
MASK_MATS = {cid: flat_mat(f"m_{cid}", palette(cid), emission=True)
             for cid in set(CLASS_ID.values())}

def make_template(ref_name):
    ref = bpy.data.objects[ref_name]
    dup = ref.copy(); dup.data = ref.data.copy()
    bpy.context.collection.objects.link(dup)
    mw = ref.matrix_world.copy(); dup.parent = None; dup.matrix_world = mw
    bpy.ops.object.select_all(action="DESELECT")
    dup.select_set(True); bpy.context.view_layer.objects.active = dup
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    vs = [v.co for v in dup.data.vertices]
    minx = min(v.x for v in vs); maxx = max(v.x for v in vs)
    miny = min(v.y for v in vs); maxy = max(v.y for v in vs); minz = min(v.z for v in vs)
    dup.data.transform(Matrix.Translation((-(minx+maxx)/2, -(miny+maxy)/2, -minz)))
    if len(dup.data.materials) == 0:
        dup.data.materials.append(WHITE_MAT)
    while len(dup.data.materials) > 1:
        dup.data.materials.pop()
    dup.location = (0, 0, -1000); dup.hide_render = True
    dup["is_template"] = 1
    return dup

TEMPLATES = {t: make_template(n) for t, n in REF.items()}
for o in list(bpy.data.objects):     # drop original placed pieces
    if o.type == "MESH" and "is_template" not in o.keys() and o is not board_surface:
        nm = o.name.lower()
        if any(k in nm for k in ["pawn", "rook", "knight", "bishop", "queen", "king"]):
            bpy.data.objects.remove(o, do_unlink=True)

# camera (created once, transform updated per job)
cam_data = bpy.data.cameras.new("Cam"); CAM = bpy.data.objects.new("Cam", cam_data)
bpy.context.collection.objects.link(CAM); bpy.context.scene.camera = CAM

scene = bpy.context.scene
try: scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError: scene.render.engine = "BLENDER_EEVEE"
scene.view_settings.view_transform = "Standard"
scene.render.film_transparent = False
WORLD = bpy.data.worlds[0]; WORLD.use_nodes = True
def set_world(rgb):
    for n in WORLD.node_tree.nodes:
        if n.type == "BACKGROUND": n.inputs[0].default_value = (*rgb, 1)

# ---------------- per-job helpers ----------------
def parse_fen(fen):
    grid = []
    for row in fen.split()[0].split("/"):
        cells = []
        for ch in row:
            cells += [None]*int(ch) if ch.isdigit() else [ch]
        grid.append(cells)
    return grid

def clear_pieces():
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.get("placed"):
            bpy.data.objects.remove(o, do_unlink=True)

def place(fen):
    grid = parse_fen(fen); placed = 0
    for r in range(8):
        rank_idx = 7 - r
        for f in range(8):
            ch = grid[r][f]
            if ch is None: continue
            t = ch.lower(); color = "w" if ch.isupper() else "b"
            obj = TEMPLATES[t].copy()              # shares mesh data
            bpy.context.collection.objects.link(obj)
            x = BOARD_MIN.x + (f + 0.5) * SQ; y = BOARD_MIN.y + (rank_idx + 0.5) * SQ
            obj.location = (x, y, BOARD_TOP); obj.rotation_euler = (0, 0, 0)
            obj.hide_render = False; obj["placed"] = 1; obj["cls"] = CLASS_ID[color + t]
            obj.material_slots[0].link = "OBJECT"
            obj.material_slots[0].material = WHITE_MAT if color == "w" else BLACK_MAT
            placed += 1
    return placed

def setup_camera(elev, dist, lens, yaw, viewpoint):
    CAM.data.lens = lens
    az = math.radians((0.0 if viewpoint == "white" else 180.0) + yaw); el = math.radians(elev)
    pos = CENTER + Vector((dist*math.cos(el)*math.sin(az), -dist*math.cos(el)*math.cos(az), dist*math.sin(el)))
    CAM.location = pos
    CAM.rotation_euler = (CENTER - pos).normalized().to_track_quat('-Z', 'Y').to_euler()

def save_corners(out, viewpoint, res, extra):
    bpy.context.view_layer.update()
    x0, y0 = BOARD_MIN.x, BOARD_MIN.y; x1, y1 = BOARD_MIN.x + BOARD_W, BOARD_MIN.y + BOARD_W
    wc = {"a1": (x0, y0), "h1": (x1, y0), "a8": (x0, y1), "h8": (x1, y1)}
    px = {}
    for name, (wx, wy) in wc.items():
        ndc = world_to_camera_view(scene, CAM, Vector((wx, wy, BOARD_TOP)))
        px[name] = [ndc.x * res, (1.0 - ndc.y) * res]
    json.dump({"res": res, "viewpoint": viewpoint, "corners": px, **extra}, open(out + "_corners.json", "w"))

def render_to(path):
    scene.render.filepath = path; bpy.ops.render.render(write_still=True)

def run_job(job):
    res = job.get("res", A.res)
    scene.render.resolution_x = res; scene.render.resolution_y = res
    clear_pieces()
    n = place(job["fen"])
    setup_camera(job.get("elev", 35.0), job.get("dist", 9.0), job.get("lens", 35.0),
                 job.get("yaw", 0.0), job["viewpoint"])
    save_corners(job["out"], job["viewpoint"], res,
                 {k: job.get(k) for k in ("elev", "dist", "lens", "yaw", "fen")})
    # RGB pass
    set_world((0.05, 0.05, 0.05))
    board_surface.data.materials.clear()
    for m in ORIG_BOARD_MAT: board_surface.data.materials.append(m)
    try: scene.eevee.taa_render_samples = 16
    except Exception: pass
    scene.render.filter_size = 1.5
    render_to(job["out"] + "_rgb.png")
    # MASK pass
    set_world((0, 0, 0))
    board_surface.data.materials.clear(); board_surface.data.materials.append(MASK_BOARD)
    for o in bpy.data.objects:
        if o.get("placed"):
            o.material_slots[0].link = "OBJECT"
            o.material_slots[0].material = MASK_MATS[o["cls"]]
    try: scene.eevee.taa_render_samples = 1
    except Exception: pass
    scene.render.filter_size = 0.01
    render_to(job["out"] + "_mask.png")
    print(f"[job] {job['out']} pieces={n}")

# ---------------- dispatch ----------------
if A.jobs:
    jobs = json.load(open(A.jobs))
    for i, job in enumerate(jobs):
        run_job(job)
        if (i + 1) % 25 == 0: print(f"[progress] {i+1}/{len(jobs)}")
else:
    run_job({"fen": A.fen, "viewpoint": A.viewpoint, "out": A.out,
             "elev": A.elev, "dist": A.dist, "lens": A.lens, "yaw": A.yaw, "res": A.res})
print("[done]")
