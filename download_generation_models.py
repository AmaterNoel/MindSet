from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "save_pt"


MODELS = [
    {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_dir": SAVE_DIR / "stabilityai__stable-diffusion-xl-base-1.0",
        "allow_patterns": [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ],
    },
    {
        "repo_id": "xinsir/controlnet-tile-sdxl-1.0",
        "local_dir": SAVE_DIR / "xinsir__controlnet-tile-sdxl-1.0",
        "allow_patterns": [
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
        ],
    },
    {
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "local_dir": SAVE_DIR / "madebyollin__sdxl-vae-fp16-fix",
        "allow_patterns": [
            "config.json",
            "diffusion_pytorch_model.safetensors",
        ],
    },
    {
        "repo_id": "h94/IP-Adapter",
        "local_dir": SAVE_DIR / "h94__IP-Adapter",
        "allow_patterns": [
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/image_encoder/config.json",
            "sdxl_models/image_encoder/model.safetensors",
        ],
    },
    {
        "repo_id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "local_dir": SAVE_DIR / "laion__CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "allow_patterns": [
            "README.md",
            "config.json",
            "open_clip_config.json",
            "open_clip_model.safetensors",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    },
]


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        repo_id = spec["repo_id"]
        local_dir = spec["local_dir"]
        print(f"\nDownloading {repo_id}")
        print(f"  -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=spec["allow_patterns"],
            max_workers=8,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
