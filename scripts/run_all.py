"""Reproduce every experiment with one command.

    python scripts/run_all.py                 # synthetic suite (configs/experiments.yaml)
    python scripts/run_all.py --groups main   # a subset of groups
    python scripts/run_all.py --natural       # also the TinyStories runs (needs data/tinystories_sentences.txt)

Each experiment runs in its own subprocess (``workers`` in parallel); results
are collected into ``outputs/summary.json``, ``outputs/summary.md`` and the
project-level ``report.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from abc_induction.report_all import build_global_report  # noqa: E402


def run_one(exp: dict, out_root: str, extra_args: list[str]) -> dict:
    name = exp["name"]
    out_dir = os.path.join(out_root, name)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "run_experiment.py"), exp["config"], "--output-dir", out_dir, "--quiet"]
    overrides = exp.get("overrides") or {}
    merged = dict(flatten(exp.get("suite_overrides") or {}))
    merged.update(flatten(overrides))
    sets = [f"name={json.dumps(name)}"] + [f"{k}={json.dumps(v)}" for k, v in merged.items()] + extra_args
    cmd += ["--set", *sets]
    t = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = proc.returncode == 0
    rec = {"name": name, "ok": ok, "seconds": round(time.time() - t, 1), "output_dir": out_dir}
    if not ok:
        rec["error"] = (proc.stderr or "")[-3000:]
        print(f"[FAIL] {name}\n{rec['error']}", flush=True)
    else:
        print(f"[ok] {name} ({rec['seconds']}s)", flush=True)
    return rec


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> {"a.b.c": leaf} (lists stay leaves)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=os.path.join(ROOT, "configs", "experiments.yaml"))
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--natural", action="store_true", help="also run the TinyStories study (scripts/run_natural.py)")
    ap.add_argument("--natural-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--analysis", default=os.path.join(ROOT, "docs", "analysis.md"))
    ap.add_argument("--collect", action="store_true", help="do not run anything; rebuild summary/report from existing outputs")
    ap.add_argument("--report", default=None, help="where to write the project-level report (default: report.md)")
    args = ap.parse_args()
    with open(args.suite, encoding="utf-8") as fh:
        suite = yaml.safe_load(fh)
    groups = suite["groups"]
    if args.groups:
        groups = {g: groups[g] for g in args.groups}
    workers = args.workers or int(suite.get("workers", 4))
    jobs = [(g, exp) for g, grp in groups.items() for exp in grp["experiments"]]
    for _, exp in jobs:  # suite-level overrides (e.g. the v0.1 suite keeps the v0.2 stage off)
        exp["suite_overrides"] = suite.get("overrides") or {}
    print(f"running {len(jobs)} experiments with {workers} workers")
    results = {}
    if args.collect:
        for g, exp in jobs:
            out_dir = os.path.join(args.out, exp["name"])
            results[exp["name"]] = {"name": exp["name"], "ok": os.path.exists(os.path.join(out_dir, "metrics.json")), "seconds": None, "output_dir": out_dir}
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_one, exp, args.out, []): (g, exp["name"]) for g, exp in jobs}
            for fut, (g, name) in futs.items():
                results[name] = fut.result()
    natural_md = ""
    if args.natural and not args.collect:
        sizes = [str(x) for x in (args.natural_sizes or [10000, 50000])]
        cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "run_natural.py"), "--sizes", *sizes, "--punct-ablation"]
        print("running natural-language study:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT)
        nat_report = os.path.join(args.out, "natural", "natural_report.md")
        if os.path.exists(nat_report):
            with open(nat_report, encoding="utf-8") as fh:
                natural_md = fh.read()
    # ---- collect
    summary = {"groups": {}, "runs": {}}
    for g, grp in groups.items():
        summary["groups"][g] = {"title": grp["title"], "experiments": [e["name"] for e in grp["experiments"]]}
    for name, rec in results.items():
        mpath = os.path.join(rec["output_dir"], "metrics.json")
        if rec["ok"] and os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as fh:
                data = json.load(fh)
            rec["metrics"] = data.get("metrics", {})
            rec["diagnostics"] = data.get("diagnostics", {})
            rec["diagnostics_v02"] = data.get("diagnostics_v02", {})
            cpath = os.path.join(rec["output_dir"], "config.yaml")
            with open(cpath, encoding="utf-8") as fh:
                rec["config"] = yaml.safe_load(fh)
        summary["runs"][name] = rec
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    analysis = ""
    if os.path.exists(args.analysis):
        with open(args.analysis, encoding="utf-8") as fh:
            analysis = fh.read()
    if not natural_md:
        nat_report = os.path.join(args.out, "natural", "natural_report.md")
        if os.path.exists(nat_report):
            with open(nat_report, encoding="utf-8") as fh:
                natural_md = fh.read()
    text = build_global_report(summary, analysis, natural_md)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    report_path = args.report or os.path.join(ROOT, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    n_ok = sum(1 for r in results.values() if r["ok"])
    print(f"done: {n_ok}/{len(results)} runs ok; report at {report_path}")


if __name__ == "__main__":
    main()
