from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "save_pt"


def download(repo_id: str, local_name: str, allow_patterns: list[str]) -> None:
    target = MODEL_ROOT / local_name
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
    )
    print(f"{repo_id} -> {target}")


if __name__ == "__main__":
    download(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "stable-diffusion-v1-5__stable-diffusion-v1-5",
        [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    )
    download(
        "lllyasviel/control_v11p_sd15_seg",
        "lllyasviel__control_v11p_sd15_seg",
        ["config.json", "diffusion_pytorch_model.safetensors"],
    )
    palette_path = MODEL_ROOT / "panoptic_coco_categories.json"
    if not palette_path.exists():
        urlretrieve(
            "https://raw.githubusercontent.com/cocodataset/panopticapi/master/"
            "panoptic_coco_categories.json",
            palette_path,
        )
    print(f"COCO palette -> {palette_path}")
