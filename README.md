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

## Experiment dashboard

Training entry points write one JSON object per split/epoch to
`metrics.jsonl`. Build a local, dependency-free dashboard with:

```powershell
python tools/experiment_dashboard.py --output-root output build
python -m http.server 8000 --directory .
```

See `REMOTE_WORKFLOW.md` for the remote execution, artifact fetch, and safe
cleanup workflow.

Build the single-file beta cache once:

```powershell
python process\prepare_betas_npy.py
```

On a server, the beta cache can live outside the copied NSD annotation/surface folder:

```bash
python train.py \
  --nsd-root /path/to/NSD \
  --betas-path /server/existing/nsd/subj01/betas_float16.npy
```
