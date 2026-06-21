import bpy
print("=== SCENE OBJECTS ===")
for o in sorted(bpy.data.objects, key=lambda x: x.name):
    loc = tuple(round(v,3) for v in o.location)
    dim = tuple(round(v,3) for v in o.dimensions)
    print(f"{o.type:8s} | {o.name:28s} | loc={loc} | dim={dim} | parent={o.parent.name if o.parent else None}")
print("=== COLLECTIONS/GROUPS ===")
for c in bpy.data.collections:
    print("coll:", c.name, [o.name for o in c.objects][:12])
