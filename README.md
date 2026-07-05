# MindSet

MindSet is an NSD fMRI-to-semantics training pipeline. It uses 3D brain responses to predict a global CLIP semantic embedding and a variable-size set of concept embeddings.

## Files

- `process.py`: builds trial-level concept labels and CLIP feature targets from NSD annotations.
- `dataloader.py`: loads NSD 3D beta volumes and processed semantic labels.
- `model.py`: 3D stem projector, spatial Transformer, global semantic head, and query-based concept head.
- `loss.py`: positive-only semantic, concept, consistency, and sparsity losses.
- `train.py`: training, validation, testing, logging, and flatmap saliency visualization.

## Local Assets

Large files are intentionally not tracked by git. Keep datasets under `D:\datasets\NSD` and local pretrained models under `save_pt\`.

Typical required local model path:

```text
save_pt\openai__clip-vit-base-patch32
```

## Run

```powershell
python process.py
python train.py
```
