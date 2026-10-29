#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_blender_compare.py — (din Turntable Studio) deschide în Blender (fără interfață) OBJ-ul texturat și GLB-ul,
listează materialele / imaginile pe care le vede Blender și le randează din aceleași camere
(Workbench, lumină plată, culoarea din textură), ca să se vadă direct dacă GLB-ul arată ca OBJ-ul.

Rulare:
  blender -b --factory-startup --python tools/debug_blender_compare.py -- \
      --obj output/obj/high/high_model.obj --glb output/glb/high_model.glb \
      --out output/debug/blender
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--glb_scale", type=float, default=1.0, help="scara GLB-ului față de OBJ (1: în pipeline-ul facial au aceeași unitate)")
    ap.add_argument("--obj_scale", type=float, default=1.0, help="scara OBJ-ului (1 = unitățile exportului)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=900)
    ap.add_argument("--azimuths", default="0,90,180,270")
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(path):
    before = set(bpy.data.objects)
    if path.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def describe(objs, label):
    meshes = [o for o in objs if o.type == "MESH"]
    faces = sum(len(o.data.polygons) for o in meshes)
    mats, imgs = [], set()
    for o in meshes:
        for slot in o.material_slots:
            m = slot.material
            if m is None:
                continue
            node_imgs = [n.image.name for n in (m.node_tree.nodes if m.use_nodes else []) if n.type == "TEX_IMAGE" and n.image]
            mats.append((m.name, node_imgs))
            imgs.update(node_imgs)
    print(f"[blender] {label}: {len(meshes)} obiecte mesh, {faces:,d} fețe, {len(mats)} sloturi de material, "
          f"{len(imgs)} imagini distincte")
    for name, node_imgs in mats[:40]:
        print(f"[blender]    material {name:24s} -> {', '.join(node_imgs) or '(fără imagine)'}")
    return {"meshes": len(meshes), "faces": faces, "materials": len(mats), "images": len(imgs)}


def bbox(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def render(objs, out_png, azimuth_deg, res, center, radius):
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_WORKBENCH"
    scn.display.shading.light = "FLAT"
    scn.display.shading.color_type = "TEXTURE"
    scn.display.shading.background_type = "VIEWPORT"
    scn.display.shading.background_color = (0.0, 0.0, 0.0)
    scn.render.resolution_x = int(res * 0.6)
    scn.render.resolution_y = res
    scn.render.film_transparent = False
    cam_data = bpy.data.cameras.get("dbg_cam") or bpy.data.cameras.new("dbg_cam")
    cam = bpy.data.objects.get("dbg_cam") or bpy.data.objects.new("dbg_cam", cam_data)
    if cam.name not in scn.collection.objects:
        scn.collection.objects.link(cam)
    scn.camera = cam
    cam_data.lens = 50.0
    a = math.radians(azimuth_deg)
    dist = radius * 3.2
    cam.location = center + Vector((dist * math.sin(a), -dist * math.cos(a), radius * 0.35))
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_data.clip_start = radius * 0.01
    cam_data.clip_end = radius * 100
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = o not in objs
    scn.render.filepath = out_png
    bpy.ops.render.render(write_still=True)


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    clear_scene()
    obj_objs = import_model(os.path.abspath(args.obj))
    for o in obj_objs:
        if o.parent is None:
            o.scale = tuple(s / args.obj_scale for s in o.scale)
    glb_objs = import_model(os.path.abspath(args.glb))
    for o in glb_objs:
        if o.parent is None:
            o.scale = tuple(s / args.glb_scale for s in o.scale)
    bpy.context.view_layer.update()
    info_obj = describe(obj_objs, "OBJ " + os.path.basename(args.obj))
    info_glb = describe(glb_objs, "GLB " + os.path.basename(args.glb))
    lo, hi = bbox(obj_objs)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-6)
    lo2, hi2 = bbox(glb_objs)
    print(f"[blender] bbox OBJ {tuple(round(v, 4) for v in (hi - lo))}  bbox GLB (după /{args.glb_scale}) "
          f"{tuple(round(v, 4) for v in (hi2 - lo2))}")
    for az in [float(a) for a in args.azimuths.split(",")]:
        render(obj_objs, os.path.join(args.out, f"obj_az{int(az):03d}.png"), az, args.res, center, radius)
        render(glb_objs, os.path.join(args.out, f"glb_az{int(az):03d}.png"), az, args.res, center, radius)
    print("[blender] RESULT", {"obj": info_obj, "glb": info_glb})


main()
