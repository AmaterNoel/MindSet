from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

root = Path(__file__).resolve().parents[1] / "save_pt" / "openai__clip-vit-large-patch14"
root.mkdir(parents=True, exist_ok=True)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.save_pretrained(root, safe_serialization=True)
processor.save_pretrained(root)
print(root)
