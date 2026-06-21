"""
render_realset_aligned.py -- v4 Blender renderer using the original chess-set.blend.

For each FEN/viewpoint job, renders:
  <out>_rgb.png
  <out>_mask.png
  <out>_corners.json

The board corners are labelled by chess square names so rectify_and_pack.py can
map the oblique render into the same canonical 512x512 board used by the real
photos. This renderer intentionally uses the more detailed course asset instead
of the low-poly v3 asset.
"""
import argparse
import json
import math
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--jobs")
ap.add_argument("--fen")
ap.add_argument("--viewpoint", choices=["white", "black"], default="white")
ap.add_argument("--out")
ap.add_argument("--elev", type=float, default=35.0)
ap.add_argument("--dist", type=float, default=9.0)
ap.add_argument("--lens", type=float, default=35.0)
ap.add_argument("--yaw", type=float, default=0.0)
ap.add_argument("--res", type=int, default=768)
A = ap.parse_args(argv)


CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for i, t in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + t] = 3 + i
    CLASS_ID["b" + t] = 9 + i


REF = {
    "P": "A(texture)",
    "N": "White knight",
    "B": "White bitshop",
    "R": "White rook",
    "Q": "White queen",
    "K": "White king",
    "p": "A(textures)",
    "n": "Black knight",
    "b": "Black bitshop",
    "r": "Black rook",
    "q": "Black queen",
    "k": "Black king",
}


def palette(cid):
    return ((cid * 17) % 256 / 255.0, (cid * 53) % 256 / 255.0, (cid * 97) % 256 / 255.0)


def world_bbox(obj):
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
    mx = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
    return mn, mx


def flat_mat(name, rgb, emission=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    shader = nt.nodes.new("ShaderNodeEmission" if emission else "ShaderNodeBsdfDiffuse")
    shader.inputs[0].default_value = (*rgb, 1.0)
    nt.links.new(shader.outputs[0], out.inputs[0])
    return mat


def classify_original_piece_name(name):
    if name in {"A(texture)", "B", "C", "D", "E", "F", "G", "H"}:
        return "P"
    if name in {"A(textures)", "B.001", "C.001", "D.001", "E.001", "F.001", "G.001", "H.001"}:
        return "p"
    low = name.lower()
    if "rook" in low:
        return "R" if "white" in low else "r"
    if "knight" in low:
        return "N" if "white" in low else "n"
    if "bitshop" in low or "bishop" in low:
        return "B" if "white" in low else "b"
    if "queen" in low:
        return "Q" if "white" in low else "q"
    if "king" in low:
        return "K" if "white" in low else "k"
    return None


board_surface = bpy.data.objects["Black & white"]
frame_surface = bpy.data.objects.get("Outer frame")
bmn, bmx = world_bbox(board_surface)
BOARD_MIN = Vector((bmn.x, bmn.y))
BOARD_MAX = Vector((bmx.x, bmx.y))
BOARD_W = max(bmx.x - bmn.x, bmx.y - bmn.y)
SQ = BOARD_W / 8.0
BOARD_TOP = bmx.z
CENTER = Vector(((bmn.x + bmx.x) / 2.0, (bmn.y + bmx.y) / 2.0, BOARD_TOP))
DIST_SCALE = BOARD_W / 4.8
ORIG_BOARD_MATS = list(board_surface.data.materials)
ORIG_FRAME_MATS = list(frame_surface.data.materials) if frame_surface else []
print(f"[board] min={tuple(round(v, 3) for v in bmn)} max={tuple(round(v, 3) for v in bmx)} "
      f"W={BOARD_W:.3f} SQ={SQ:.3f} top={BOARD_TOP:.3f} dist_scale={DIST_SCALE:.3f}")

MASK_BOARD = flat_mat("v4_mask_board", palette(CLASS_ID["empty_light"]), emission=True)
MASK_MATS = {cid: flat_mat(f"v4_mask_{cid}", palette(cid), emission=True)
             for cid in set(CLASS_ID.values())}


def make_template(piece_char, ref_name):
    ref = bpy.data.objects.get(ref_name)
    if ref is None:
        raise RuntimeError(f"Missing reference object {ref_name!r} for {piece_char}")
    dup = ref.copy()
    dup.data = ref.data.copy()
    bpy.context.collection.objects.link(dup)
    dup.name = f"TEMPLATE_{piece_char}"
    dup.parent = None
    dup.matrix_world = ref.matrix_world.copy()
    bpy.ops.object.select_all(action="DESELECT")
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    verts = [v.co for v in dup.data.vertices]
    minx, maxx = min(v.x for v in verts), max(v.x for v in verts)
    miny, maxy = min(v.y for v in verts), max(v.y for v in verts)
    minz = min(v.z for v in verts)
    dup.data.transform(Matrix.Translation((-(minx + maxx) / 2.0, -(miny + maxy) / 2.0, -minz)))
    dup.location = (0, 0, -10000)
    dup.hide_render = True
    dup.hide_viewport = True
    dup["is_template"] = 1
    dup["piece_char"] = piece_char
    return dup


TEMPLATES = {piece: make_template(piece, name) for piece, name in REF.items()}

for obj in list(bpy.data.objects):
    if obj.type != "MESH" or obj.get("is_template"):
        continue
    if classify_original_piece_name(obj.name):
        bpy.data.objects.remove(obj, do_unlink=True)

cam_data = bpy.data.cameras.new("V4Cam")
CAM = bpy.data.objects.new("V4Cam", cam_data)
bpy.context.collection.objects.link(CAM)
bpy.context.scene.camera = CAM

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "Medium High Contrast"
scene.render.resolution_percentage = 100

WORLD = bpy.data.worlds[0]
WORLD.use_nodes = True


def set_world(rgb):
    for node in WORLD.node_tree.nodes:
        if node.type == "BACKGROUND":
            node.inputs[0].default_value = (*rgb, 1.0)
            node.inputs[1].default_value = 0.8


def parse_fen(fen):
    grid = []
    for row in fen.split()[0].split("/"):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend([None] * int(ch))
            else:
                cells.append(ch)
        grid.append(cells)
    return grid


def square_center(file_idx, rank):
    # In chess-set.blend, file a is at high X and rank 1 is at high Y.
    x = BOARD_MIN.x + (7 - file_idx + 0.5) * SQ
    y = BOARD_MAX.y - (rank - 0.5) * SQ
    return x, y


def clear_pieces():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.get("placed"):
            bpy.data.objects.remove(obj, do_unlink=True)


def place(fen):
    grid = parse_fen(fen)
    placed = 0
    for fen_rank_idx, row in enumerate(grid):
        rank = 8 - fen_rank_idx
        for file_idx, ch in enumerate(row):
            if ch is None:
                continue
            tpl = TEMPLATES[ch]
            obj = tpl.copy()
            obj.data = tpl.data.copy()
            bpy.context.collection.objects.link(obj)
            x, y = square_center(file_idx, rank)
            obj.location = (x, y, BOARD_TOP)
            obj.hide_render = False
            obj.hide_viewport = False
            obj["placed"] = 1
            color = "w" if ch.isupper() else "b"
            obj["cls"] = CLASS_ID[color + ch.lower()]
            obj["piece_char"] = ch
            placed += 1
    return placed


def setup_camera(elev, dist, lens, yaw, viewpoint):
    CAM.data.lens = lens
    CAM.data.clip_end = 10000
    base = 0.0 if viewpoint == "white" else math.pi
    az = base + math.radians(yaw)
    el = math.radians(elev)
    d = dist * DIST_SCALE
    pos = CENTER + Vector((d * math.cos(el) * math.sin(az),
                           d * math.cos(el) * math.cos(az),
                           d * math.sin(el)))
    CAM.location = pos
    CAM.rotation_euler = (CENTER - pos).normalized().to_track_quat("-Z", "Y").to_euler()


def chess_corner_world():
    return {
        "a1": (BOARD_MAX.x, BOARD_MAX.y),
        "h1": (BOARD_MIN.x, BOARD_MAX.y),
        "a8": (BOARD_MAX.x, BOARD_MIN.y),
        "h8": (BOARD_MIN.x, BOARD_MIN.y),
    }


def save_corners(out, viewpoint, res, extra):
    bpy.context.view_layer.update()
    corners = {}
    for name, (wx, wy) in chess_corner_world().items():
        ndc = world_to_camera_view(scene, CAM, Vector((wx, wy, BOARD_TOP)))
        corners[name] = [ndc.x * res, (1.0 - ndc.y) * res]
    with open(out + "_corners.json", "w") as f:
        json.dump({"res": res, "viewpoint": viewpoint, "asset": "chess-set.blend",
                   "corners": corners, **extra}, f)


def render_to(path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def restore_board_materials():
    board_surface.data.materials.clear()
    for mat in ORIG_BOARD_MATS:
        board_surface.data.materials.append(mat)
    if frame_surface:
        frame_surface.data.materials.clear()
        for mat in ORIG_FRAME_MATS:
            frame_surface.data.materials.append(mat)


def set_mask_materials():
    board_surface.data.materials.clear()
    board_surface.data.materials.append(MASK_BOARD)
    if frame_surface:
        frame_surface.data.materials.clear()
        frame_surface.data.materials.append(MASK_BOARD)
    for obj in bpy.data.objects:
        if obj.get("placed"):
            obj.data.materials.clear()
            obj.data.materials.append(MASK_MATS[obj["cls"]])


def run_job(job):
    res = job.get("res", A.res)
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    clear_pieces()
    n = place(job["fen"])
    setup_camera(job.get("elev", 35.0), job.get("dist", 9.0), job.get("lens", 35.0),
                 job.get("yaw", 0.0), job["viewpoint"])
    save_corners(job["out"], job["viewpoint"], res,
                 {k: job.get(k) for k in ("elev", "dist", "lens", "yaw", "fen")})

    restore_board_materials()
    set_world((0.05, 0.05, 0.05))
    try:
        scene.eevee.taa_render_samples = 16
    except Exception:
        pass
    scene.render.filter_size = 1.5
    render_to(job["out"] + "_rgb.png")

    set_mask_materials()
    set_world((0.0, 0.0, 0.0))
    try:
        scene.eevee.taa_render_samples = 1
    except Exception:
        pass
    scene.render.filter_size = 0.01
    render_to(job["out"] + "_mask.png")
    print(f"[job] {job['out']} pieces={n}")


if A.jobs:
    jobs = json.load(open(A.jobs))
    for i, job in enumerate(jobs):
        run_job(job)
        if (i + 1) % 25 == 0:
            print(f"[progress] {i + 1}/{len(jobs)}")
else:
    run_job({"fen": A.fen, "viewpoint": A.viewpoint, "out": A.out,
             "elev": A.elev, "dist": A.dist, "lens": A.lens, "yaw": A.yaw, "res": A.res})
print("[done]")
