# Remote experiment workflow

## Access prerequisites

The training server must be reachable from this machine, normally through the
same LAN, a VPN, or an SSH jump host. Configure a key-based alias locally:

```sshconfig
Host mindset-train
    HostName 172.18.116.140
    User longnuoer
    IdentityFile ~/.ssh/id_ed25519
    # ProxyJump user@public-jump-host
```

Test it with `ssh -o BatchMode=yes mindset-train "hostname; nvidia-smi"`.
Do not commit passwords or private keys. A `SHA256:...` value is a public-key
fingerprint, not an SSH private key.

## Recommended loop

1. Develop and test locally on a feature branch.
2. Push a reviewed commit to GitHub.
3. On the server, pull that exact commit into a dedicated worktree.
4. Run training in `tmux` or through a scheduler, writing each run to a unique
   directory under `output/`.
5. Fetch only `metrics.jsonl`, `config.json`, `test_metrics.json`, logs, and
   visualization images. Keep large checkpoints and datasets on the server.
6. Build the local dashboard:

```powershell
python tools/experiment_dashboard.py --output-root output build
python -m http.server 8000 --directory .
```

Open `http://localhost:8000/dashboard/`. Rebuild after fetching a new run.

## Cleanup policy

Cleanup is intentionally preview-only by default:

```powershell
python tools/experiment_dashboard.py --output-root output clean --keep-latest 5
```

Review the printed paths, then add `--apply`. Best checkpoints, metrics,
configuration, logs, and visualizations are preserved. Server cleanup should
follow the same two-stage review and must never touch datasets or pretrained
models.
