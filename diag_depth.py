import json, os, numpy as np, torch
from utils.render import render_core
from scenesplat_npy import load_scenesplat_scene

FT = "/scratch-shared/mkhan4/gaussian_world/finetune_data/0201_840151"
GS = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train/0201_840151"

d = json.load(open(os.path.join(FT, "transforms_train.json")))
W, H = int(d["w"]), int(d["h"])
fx, fy, cx, cy = d["fl_x"], d["fl_y"], d["cx"], d["cy"]
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device="cuda")

fr = d["frames"][0]
c2w = np.array(fr["transform_matrix"])
w2c = np.linalg.inv(c2w)
view = torch.tensor(w2c, dtype=torch.float32, device="cuda").unsqueeze(0)

g = load_scenesplat_scene(GS)
means = g["coords"].cuda()
print("scene coord range:", [round(x,2) for x in means.min(0).values.tolist()],
      [round(x,2) for x in means.max(0).values.tolist()])
print("camera position:", [round(x,2) for x in c2w[:3, 3].tolist()])

colors, aux = render_core(
    means, g["quats"].cuda(), g["scales"].cuda(), g["opacities"].cuda(),
    g["sh0"].cuda(), None, view, K.unsqueeze(0), render_size=(W, H),
    background_color="white", render_mode="RGB+ED")

print("colors shape:", tuple(colors.shape))
print("aux keys:", list(aux.keys()))
acc = colors[:, 3:4]
al = aux["alphas"]
print("acc_depth  range:", round(float(acc.min()), 4), round(float(acc.max()), 4))
print("alpha      range:", round(float(al.min()), 4), round(float(al.max()), 4),
      "mean", round(float(al.mean()), 4))
norm = torch.where(al > 1e-6, acc / al.clamp_min(1e-6), torch.zeros_like(acc))
print("normalized range:", round(float(norm.min()), 4), round(float(norm.max()), 4))

cent = means.mean(0).cpu().numpy()
cam = c2w[:3, 3]
print("cam-to-centroid distance:", round(float(np.linalg.norm(cent - cam)), 2))