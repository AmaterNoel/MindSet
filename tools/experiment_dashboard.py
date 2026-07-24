from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PROTECTED_NAMES = {"best_model.pt", "best_base_model.pt", "best_low_model.pt"}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def discover_runs(output_root: Path) -> list[dict[str, Any]]:
    runs = []
    if not output_root.exists():
        return runs
    for metrics_path in output_root.rglob("metrics.jsonl"):
        run_dir = metrics_path.parent
        rows = read_metrics(metrics_path)
        config = read_json(run_dir / "config.json")
        images = [
            str(path.relative_to(output_root)).replace("\\", "/")
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        test = read_json(run_dir / "test_metrics.json")
        runs.append(
            {
                "name": str(run_dir.relative_to(output_root)).replace("\\", "/"),
                "mtime": metrics_path.stat().st_mtime,
                "metrics": rows,
                "config": config,
                "test": test,
                "images": images,
            }
        )
    return sorted(runs, key=lambda item: item["mtime"], reverse=True)


def render_dashboard(output_root: Path, destination: Path) -> None:
    runs = discover_runs(output_root)
    payload = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    asset_root = os.path.relpath(output_root.resolve(), destination.resolve().parent).replace("\\", "/") + "/"
    title = "MindSet Experiments"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title>
<style>
:root{{--bg:#09111f;--card:#111d31;--ink:#eaf1ff;--muted:#91a3be;--cyan:#51d5ff;--lime:#b5f36b}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07101d,#10182a);color:var(--ink);
font:14px/1.5 system-ui,sans-serif}} main{{max-width:1400px;margin:auto;padding:28px}}
h1{{font-size:32px;margin:0}} .sub{{color:var(--muted);margin:4px 0 24px}} select{{background:#101c30;color:var(--ink);
border:1px solid #30405c;border-radius:8px;padding:9px}} .grid{{display:grid;grid-template-columns:1.5fr 1fr;gap:18px}}
.card{{background:rgba(17,29,49,.92);border:1px solid #253652;border-radius:14px;padding:18px;overflow:hidden}}
.wide{{grid-column:1/-1}} svg{{width:100%;height:340px}} .legend{{color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}} pre{{white-space:pre-wrap;color:#bed0e9}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.gallery img{{width:100%;border-radius:8px;background:#08101d}} table{{width:100%;border-collapse:collapse}}
td,th{{padding:7px;border-bottom:1px solid #263752;text-align:right}} td:first-child,th:first-child{{text-align:left}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{title}</h1>
<p class="sub">训练曲线、最终指标与可视化结果 · 共 <span id="count"></span> 次实验</p>
<select id="run"></select><div class="grid" style="margin-top:18px">
<section class="card wide"><h2>指标曲线</h2><div id="legend" class="legend"></div><svg id="chart" viewBox="0 0 1000 340"></svg></section>
<section class="card"><h2>最终指标</h2><div id="summary"></div></section>
<section class="card"><h2>配置</h2><pre id="config"></pre></section>
<section class="card wide"><h2>可视化结果</h2><div id="gallery" class="gallery"></div></section>
</div></main><script>
const runs={payload}, root={json.dumps(asset_root)};
const colors=["#51d5ff","#b5f36b","#ffb45c","#d989ff","#ff718c","#91a3be"];
const sel=document.querySelector("#run"); document.querySelector("#count").textContent=runs.length;
runs.forEach((r,i)=>sel.add(new Option(r.name,i)));
function numericKeys(rows){{const s=new Set;rows.forEach(r=>Object.entries(r).forEach(([k,v])=>typeof v==="number"&&!["epoch"].includes(k)&&s.add(k)));return [...s]}}
function draw(run){{const rows=run.metrics.filter(r=>r.split!=="test"&&Number.isFinite(r.epoch));const keys=numericKeys(rows).slice(0,6);
 const svg=document.querySelector("#chart"), W=1000,H=340,p=38;svg.innerHTML="";
 let vals=rows.flatMap(r=>keys.map(k=>r[k]).filter(Number.isFinite));let lo=Math.min(...vals),hi=Math.max(...vals);
 if(!vals.length){{svg.innerHTML='<text x="40" y="80" fill="#91a3be">暂无可绘制指标</text>';return}}
 if(lo===hi)hi=lo+1;const epochs=rows.map(r=>r.epoch),emin=Math.min(...epochs),emax=Math.max(...epochs);
 const x=e=>p+(e-emin)/(Math.max(1,emax-emin))*(W-2*p),y=v=>H-p-(v-lo)/(hi-lo)*(H-2*p);
 svg.innerHTML=`<line x1="${{p}}" y1="${{H-p}}" x2="${{W-p}}" y2="${{H-p}}" stroke="#40506c"/><line x1="${{p}}" y1="${{p}}" x2="${{p}}" y2="${{H-p}}" stroke="#40506c"/>`;
 keys.forEach((k,i)=>{{const pts=rows.filter(r=>Number.isFinite(r[k])).map(r=>`${{x(r.epoch)}},${{y(r[k])}}`).join(" ");
 svg.insertAdjacentHTML("beforeend",`<polyline points="${{pts}}" fill="none" stroke="${{colors[i]}}" stroke-width="2"/>`)}});
 document.querySelector("#legend").innerHTML=keys.map((k,i)=>`<span><i class="dot" style="background:${{colors[i]}}"></i>${{k}}</span>`).join("");
 const last=run.test&&Object.keys(run.test).length?run.test:run.metrics.at(-1)||{{}};
 document.querySelector("#summary").innerHTML='<table>'+Object.entries(last).filter(([,v])=>typeof v==="number").map(([k,v])=>`<tr><td>${{k}}</td><td>${{v.toFixed(5)}}</td></tr>`).join("")+'</table>';
 document.querySelector("#config").textContent=JSON.stringify(run.config,null,2);
 document.querySelector("#gallery").innerHTML=run.images.map(p=>`<a href="${{root+p}}"><img loading="lazy" src="${{root+p}}" alt="${{p}}"></a>`).join("")||"暂无图片";
}}
sel.onchange=()=>draw(runs[sel.value]);if(runs.length)draw(runs[0]);
</script></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    print(f"Wrote {destination} with {len(runs)} run(s)")


def cleanup(output_root: Path, keep_latest: int, apply: bool) -> None:
    run_dirs = sorted(
        {path.parent for path in output_root.rglob("metrics.jsonl")},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates: list[Path] = []
    for run_dir in run_dirs[keep_latest:]:
        candidates.extend(
            path for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"} and path.name not in PROTECTED_NAMES
        )
    for path in candidates:
        print(("DELETE " if apply else "WOULD DELETE ") + str(path))
        if apply:
            path.unlink()
    print(f"{len(candidates)} candidate(s); {'deleted' if apply else 'dry run only'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MindSet experiment dashboard or safely prune checkpoints.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--destination", type=Path, default=Path("dashboard/index.html"))
    clean = sub.add_parser("clean")
    clean.add_argument("--keep-latest", type=int, default=5)
    clean.add_argument("--apply", action="store_true", help="Actually delete candidates; default is a dry run.")
    args = parser.parse_args()
    if args.command == "build":
        render_dashboard(args.output_root, args.destination)
    else:
        cleanup(args.output_root, max(0, args.keep_latest), args.apply)


if __name__ == "__main__":
    main()
