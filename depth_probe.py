import numpy as np, torch
from utils.render import render_core

device = "cuda"
means = torch.tensor([[0.0, 0.0, 5.0]], device=device)
quats = torch.tensor([[1.0, 0, 0, 0]], device=device)
scales = torch.tensor([[np.log(0.3)]*3], device=device)
opac = torch.tensor([10.0], device=device)
sh0 = torch.tensor([[0.5, 0.5, 0.5]], device=device)

W = H = 256
fx = fy = 200.0
K = torch.tensor([[[fx,0,W/2],[0,fy,H/2],[0,0,1]]], device=device)
view = torch.eye(4, device=device).unsqueeze(0)

colors, aux = render_core(means, quats, scales, opac, sh0, None,
                          view, K, render_size=(W,H),
                          background_color="black", render_mode="RGB+ED")
depth = colors[0,3]
al = aux["alphas"][0,0]
hit = al > 0.5
print("gaussian true z-distance = 5.0")
print("depth at center pixel:", float(depth[H//2, W//2]))
if hit.any():
    print("depth over hit region: min", float(depth[hit].min()), "max", float(depth[hit].max()))
print("alpha center:", float(al[H//2, W//2]))
