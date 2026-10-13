"""Profile one induction run with cProfile.

    python scripts/profile_run.py configs/tinystories.yaml --set dataset.max_sentences=4000 --out outputs/profile/ts4k

Writes ``profile.prof`` (pstats), ``profile.txt`` (top functions by total and
cumulative time, per-stage timings) and prints a short summary.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import logging
import os
import pstats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abc_induction.experiment import load_config, run_experiment  # noqa: E402
from run_experiment import parse_overrides  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/profile/run")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(args.config, parse_overrides(args.set))
    cfg["output_dir"] = args.out
    os.makedirs(args.out, exist_ok=True)
    prof = cProfile.Profile()
    prof.enable()
    res = run_experiment(cfg)
    prof.disable()
    prof.dump_stats(os.path.join(args.out, "profile.prof"))
    buf = io.StringIO()
    eff = res.metrics["efficiency"]
    buf.write(f"# profile: {cfg['name']} ({eff['n_sentences']} sentences, {eff['n_unique_span_types']} span types, "
              f"{eff['n_provisional_classes']} provisional / {eff['n_accepted_classes']} accepted classes)\n\n")
    buf.write("## per-stage wall time (s)\n")
    for k, v in eff["runtime_sec"].items():
        buf.write(f"  {k:14s} {v:8.2f}\n")
    st = pstats.Stats(prof, stream=buf)
    st.strip_dirs()
    buf.write("\n## top functions by own time (tottime)\n")
    st.sort_stats("tottime").print_stats(args.top)
    buf.write("\n## top functions by cumulative time\n")
    st.sort_stats("cumulative").print_stats(args.top)
    text = buf.getvalue()
    with open(os.path.join(args.out, "profile.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
