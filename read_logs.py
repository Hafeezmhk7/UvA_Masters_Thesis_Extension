from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

vers = sorted(glob.glob("logs/scenesplat_finetune/lightning_logs/version_*"))
d = vers[-1]
print("reading", d)
ea = EventAccumulator(d)
ea.Reload()
tags = ea.Tags()["scalars"]
print("available metrics:", tags, "\n")
for tag in tags:
    if any(k in tag.lower() for k in ["loss", "psnr", "ssim", "lpips"]):
        events = ea.Scalars(tag)
        vals = [(e.step, round(e.value, 4)) for e in events]
        print(f"{tag}:  first {vals[0]}")
        for v in vals[-4:]:
            print("    ", v)
