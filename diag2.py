import json, os, numpy as np, torch
from utils.render import render_core
from scenesplat_npy import load_scenesplat_scene

FT = "/scratch-shared/mkhan4/gaussian_world/finetune_data/0201_840151"
GS = "/scratch-shared/mkhan4/gaussian_world/preprocessed/interior_gs/train/0201_840151"
d = json.load(open(os.path.join(FT, "transforms_train.json")))
W, H = int(d["w"]), int(d["h"])
fx, fy, cx, cy = d["fl_x"], d["fl_y"], d["cx"], d["cy"]
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device="cuda").unsqueeze(0)
fr = d["frames"][0]
c2w = np.array(fr["transform_matrix"])
w2c = np.linalg.inv(c2w)
view = torch.tensor(w2c, dtype=torch.float32, device="cuda").unsqueeze(0)
g = load_scenesplat_scene(GS)

means = g["coords"].cuda()
w2c_t = torch.tensor(w2c, dtype=torch.float32, device="cuda")
ones = torch.ones(means.shape[0], 1, device="cuda")
cam_pts = (w2c_t @ torch.cat([means, ones], 1).T).T[:, :3]
print("gaussian camera-space z: min", round(float(cam_pts[:, 2].min()), 3),
      "max", round(float(cam_pts[:, 2].max()), 3),
      "mean", round(float(cam_pts[:, 2].mean()), 3))

sc = g["scales"].cuda()
print("log-scale range:", round(float(sc.min()), 3), round(float(sc.max()), 3))
print("actual size (exp) range:", round(float(sc.exp().min()), 5), round(float(sc.exp().max()), 3))

colors, aux = render_core(means, g["quats"].cuda(), sc, g["opacities"].cuda(),
    g["sh0"].cuda(), None, view, K, render_size=(W, H),
    background_color="white", render_mode="RGB+ED")
depth = colors[0, 3]
al = aux["alphas"][0, 0]
hit = al > 0.5
print("rendered depth over hit pixels: min", round(float(depth[hit].min()), 3),
      "max", round(float(depth[hit].max()), 3))
