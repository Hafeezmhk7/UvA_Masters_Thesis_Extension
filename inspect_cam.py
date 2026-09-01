import json, os, zipfile

CAM = "/scratch-shared/mkhan4/gaussian_world/camera_data"
GS_FULL = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train"
GS_CHUNK = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train_grid1.0cm_chunk8x8_stride6x6"

scene = "0201_840151"   # a scene we know exists in all three

print("="*70)
print("1. CAMERA SCENE FOLDER CONTENTS")
print("="*70)
sdir = os.path.join(CAM, "scenes", scene)
for fn in sorted(os.listdir(sdir)):
    p = os.path.join(sdir, fn)
    print(f"  {fn:35s} {os.path.getsize(p)/1e6:.2f} MB")

print("\n" + "="*70)
print("2. POSE FILE STRUCTURE (lang_feat_selected_imgs.json)")
print("="*70)
pj = os.path.join(sdir, "lang_feat_selected_imgs.json")
d = json.load(open(pj))
print("  top-level type:", type(d).__name__)
if isinstance(d, dict):
    print("  top-level keys:", list(d.keys())[:10])
    k0 = list(d.keys())[0]
    print(f"  first key: {k0!r}")
    v0 = d[k0]
    print(f"  first value type: {type(v0).__name__}")
    if isinstance(v0, dict):
        print("  first value keys:", list(v0.keys()))
        for kk in v0:
            vv = v0[kk]
            if isinstance(vv, list):
                print(f"    {kk}: list len {len(vv)}, sample {vv[:1]}")
            else:
                print(f"    {kk}: {vv!r}")
elif isinstance(d, list):
    print("  list length:", len(d))
    print("  first element:", json.dumps(d[0], indent=2)[:500])

print("\n" + "="*70)
print("3. IMAGES ZIP CONTENTS")
print("="*70)
zp = os.path.join(sdir, "images.zip")
if os.path.exists(zp):
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        print(f"  {len(names)} entries, first 5:")
        for n in names[:5]:
            print("   ", n)

print("\n" + "="*70)
print("4. METADATA: chunks for this scene")
print("="*70)
meta = os.path.join(CAM, "metadata", "chunk_descriptions_simplified.jsonl")
rows = []
with open(meta) as f:
    for line in f:
        r = json.loads(line)
        if r.get("scene_id") == scene:
            rows.append(r)
print(f"  {len(rows)} chunks for scene {scene}")
if rows:
    r = rows[0]
    print("  chunk keys:", list(r.keys()))
    print("  first chunk_id:", r.get("chunk_id"))
    si = r.get("selected_images")
    print("  selected_images type:", type(si).__name__, "len", len(si) if isinstance(si,list) else "-")
    if isinstance(si, list) and si:
        print("  selected_images[0]:", si[0])

print("\n" + "="*70)
print("5. DO METADATA chunk_ids MATCH THE GRID-CHUNK FOLDER?")
print("="*70)
meta_chunk_ids = set(r["chunk_id"] for r in rows)
grid = set(os.listdir(GS_CHUNK))
this_scene_grid = sorted(g for g in grid if g.startswith(scene))
print(f"  metadata chunk_ids for scene: {sorted(meta_chunk_ids)[:6]}")
print(f"  grid folder chunks for scene: {this_scene_grid[:6]}")
print(f"  match: {meta_chunk_ids & grid == meta_chunk_ids and len(meta_chunk_ids)>0}")

print("\n" + "="*70)
print("6. FULL-SCENE SPLAT for this scene (train folder)")
print("="*70)
fs = os.path.join(GS_FULL, scene)
if os.path.isdir(fs):
    print("  files:", sorted(os.listdir(fs)))