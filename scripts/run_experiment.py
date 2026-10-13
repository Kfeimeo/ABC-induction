"""Run one experiment from a YAML config.

    python scripts/run_experiment.py configs/synthetic1_clean.yaml [--set key.sub=value ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abc_induction.experiment import load_config, run_experiment  # noqa: E402


def parse_overrides(items: list[str]) -> dict:
    out: dict = {}
    for it in items:
        key, _, val = it.partition("=")
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=json_value")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, parse_overrides(args.set))
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    res = run_experiment(cfg)
    m = res.metrics
    summary = {k: m[k] for k in ("class_recovery_type_level", "constituents_derivation_level", "composition_recovery", "ambiguity_detection") if k in m}
    print(json.dumps({"name": cfg["name"], "output_dir": cfg["output_dir"], "efficiency": m.get("efficiency"), "summary": {
        "class_pair_f1": summary.get("class_recovery_type_level", {}).get("f1"),
        "class_ari": summary.get("class_recovery_type_level", {}).get("ari"),
        "constituent_f1": summary.get("constituents_derivation_level", {}).get("all", {}).get("f1"),
        "constituent_f1_len2": summary.get("constituents_derivation_level", {}).get("len_ge_2", {}).get("f1"),
        "composition_f1": summary.get("composition_recovery", {}).get("f1"),
        "ambiguity_f1": summary.get("ambiguity_detection", {}).get("f1"),
        "occ_assignment_acc": summary.get("ambiguity_detection", {}).get("occurrence_assignment_accuracy"),
    }, "summary_v02": {
        "class_pair_f1": m.get("v02", {}).get("class_recovery_type_level", {}).get("f1"),
        "class_ari": m.get("v02", {}).get("class_recovery_type_level", {}).get("ari"),
        "constituent_f1": m.get("v02", {}).get("constituents_derivation_level", {}).get("all", {}).get("f1"),
        "composition_f1": m.get("v02", {}).get("composition_recovery", {}).get("f1"),
        "ambiguity_f1": m.get("v02", {}).get("ambiguity_detection", {}).get("f1"),
        "occ_assignment_acc": m.get("v02", {}).get("ambiguity_detection", {}).get("occurrence_assignment_accuracy"),
        "objective": m.get("v02", {}).get("objective", {}).get("objective"),
        "moves": m.get("v02", {}).get("accepted_moves"),
        "runtime_sec": m.get("v02", {}).get("runtime_sec"),
    }}, indent=1))


if __name__ == "__main__":
    main()
