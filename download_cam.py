from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="clapfor/temp_data",
    repo_type="dataset",
    local_dir="/scratch-shared/mkhan4/gaussian_world/camera_data",
    allow_patterns=["scenes/*", "metadata/*"],
    max_workers=8,
)
print("ALL DONE")