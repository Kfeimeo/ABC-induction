"""Root exact-string (modifier) evidence at corpus scale, without the grammar.

    python scripts/root_string_scan.py --sizes 10000 100000 500000 --out outputs/v03/root_scan

The `A.`/`AB.`, `A.`/`XA.` and `AB.`/`AXB.` tests of ``modifier_evidence.ModifierAnalyzer`` need no
posterior, so they can be run on subsets far larger than the grammar induction can handle.  For
every size N (the same shuffled prefix as ``run_natural.py``) this writes the number of pairs and
candidates per direction, the top candidates by productivity (distinct host sentences) and
support, and the agreement of the candidate sets of the two disjoint halves of N.  Candidates are
strings: no class, no linguistic claim.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from abc_induction.corpus import tokenize  # noqa: E402
from abc_induction.modifier_evidence import PUNCT  # noqa: E402


def load(path: str, min_len: int = 2, max_len: int = 15, seed: int = 42) -> list[tuple[str, ...]]:
    sents = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = tokenize(line.strip(), "separate_tokens", False)
            if min_len <= len(t) <= max_len:
                sents.append(tuple(t))
    random.Random(seed).shuffle(sents)
    return sents


def strip(s: tuple[str, ...]) -> tuple[str, ...]:
    n = len(s)
    while n > 0 and s[n - 1] in PUNCT:
        n -= 1
    return s[:n]


def scan(sents: list[tuple[str, ...]], max_mod: int = 3, max_ins: int = 4) -> dict:
    idx: dict[tuple[str, ...], int] = defaultdict(int)
    for s in sents:
        c = strip(s)
        if c:
            idx[c] += 1
    support = {d: Counter() for d in ("RIGHT", "LEFT", "INTERNAL")}
    hosts = {d: defaultdict(set) for d in ("RIGHT", "LEFT", "INTERNAL")}
    examples = {d: defaultdict(list) for d in ("RIGHT", "LEFT", "INTERNAL")}
    pairs = Counter()

    def add(d, cand, host, long, w):
        if all(t in PUNCT for t in cand):
            return
        support[d][cand] += w
        hosts[d][cand].add(host)
        pairs[d] += 1
        if len(examples[d][cand]) < 3:
            examples[d][cand].append(f"{' '.join(host)}  ->  {' '.join(long)}")

    for s, w in idx.items():
        n = len(s)
        if n < 2:
            continue
        for k in range(1, min(max_mod, n - 1) + 1):
            if s[:-k] in idx:
                add("RIGHT", s[-k:], s[:-k], s, w)
            if s[k:] in idx:
                add("LEFT", s[:k], s[k:], s, w)
        for i in range(1, n - 1):
            for l in range(1, max_ins + 1):
                j = i + l
                if j >= n:
                    break
                red = s[:i] + s[j:]
                if red in idx:
                    add("INTERNAL", s[i:j], red, s, w)
    out = {"n_sentences": len(sents), "n_distinct": len(idx), "pairs": dict(pairs), "directions": {}}
    for d in ("RIGHT", "LEFT", "INTERNAL"):
        rows = [{"candidate": " ".join(c), "support": v, "productivity": len(hosts[d][c]), "examples": examples[d][c]} for c, v in support[d].items()]
        rows.sort(key=lambda r: (-r["productivity"], -r["support"]))
        out["directions"][d] = {
            "n_candidates": len(rows),
            "n_support_ge_3": sum(1 for r in rows if r["support"] >= 3),
            "n_productivity_ge_3": sum(1 for r in rows if r["productivity"] >= 3),
            "n_single_token_productivity_ge_3": sum(1 for r in rows if r["productivity"] >= 3 and " " not in r["candidate"]),
            "top_by_productivity": rows[:25],
        }
    return out


def halves_agreement(sents: list[tuple[str, ...]], min_prod: int = 3) -> dict:
    n = len(sents) // 2
    a, b = scan_sets(sents[:n], min_prod), scan_sets(sents[n:], min_prod)
    out = {}
    for d in ("RIGHT", "LEFT", "INTERNAL"):
        sa, sb = a[d], b[d]
        out[d] = {"n_a": len(sa), "n_b": len(sb), "n_shared": len(sa & sb), "jaccard": round(len(sa & sb) / max(1, len(sa | sb)), 4), "shared": sorted(sa & sb)[:20]}
    return out


def scan_sets(sents: list[tuple[str, ...]], min_prod: int) -> dict[str, set[str]]:
    """Candidate sets (productivity >= min_prod) per direction."""
    idx: dict[tuple[str, ...], int] = defaultdict(int)
    for s in sents:
        c = strip(s)
        if c:
            idx[c] += 1
    hosts = {d: defaultdict(set) for d in ("RIGHT", "LEFT", "INTERNAL")}
    for s in idx:
        n = len(s)
        if n < 2:
            continue
        for k in range(1, min(3, n - 1) + 1):
            if s[:-k] in idx and not all(t in PUNCT for t in s[-k:]):
                hosts["RIGHT"][s[-k:]].add(s[:-k])
            if s[k:] in idx and not all(t in PUNCT for t in s[:k]):
                hosts["LEFT"][s[:k]].add(s[k:])
        for i in range(1, n - 1):
            for l in range(1, 5):
                j = i + l
                if j >= n:
                    break
                if s[:i] + s[j:] in idx and not all(t in PUNCT for t in s[i:j]):
                    hosts["INTERNAL"][s[i:j]].add(s[:i] + s[j:])
    return {d: {" ".join(c) for c, h in hosts[d].items() if len(h) >= min_prod} for d in hosts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "tinystories_sentences.txt"))
    ap.add_argument("--sizes", nargs="*", type=int, default=[10000, 50000, 100000, 200000, 500000])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "v03", "root_scan"))
    ap.add_argument("--min-productivity", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sents = load(args.data)
    print(f"{len(sents)} usable sentences")
    summary = {}
    md = ["# Root exact-string evidence vs corpus size (TinyStories, no grammar)\n",
          "Pairs are distinct sentence strings; candidates are the removed spans; productivity = distinct host sentences; the halves agreement is the Jaccard of the candidate sets (productivity >= %d) of the two disjoint halves of each subset.\n" % args.min_productivity]
    md.append("| N | distinct sentences | right pairs | left pairs | internal pairs | RIGHT cand. (prod>=3 / single-token) | LEFT cand. | INTERNAL cand. | halves Jaccard R / L / I |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for n in args.sizes:
        if n > len(sents):
            n = len(sents)
        res = scan(sents[:n])
        res["halves_agreement"] = halves_agreement(sents[:n], args.min_productivity)
        summary[n] = res
        d = res["directions"]
        h = res["halves_agreement"]
        md.append(f"| {n} | {res['n_distinct']} | {res['pairs'].get('RIGHT', 0)} | {res['pairs'].get('LEFT', 0)} | {res['pairs'].get('INTERNAL', 0)} | {d['RIGHT']['n_productivity_ge_3']} / {d['RIGHT']['n_single_token_productivity_ge_3']} | {d['LEFT']['n_productivity_ge_3']} / {d['LEFT']['n_single_token_productivity_ge_3']} | {d['INTERNAL']['n_productivity_ge_3']} / {d['INTERNAL']['n_single_token_productivity_ge_3']} | {h['RIGHT']['jaccard']} / {h['LEFT']['jaccard']} / {h['INTERNAL']['jaccard']} |")
        print(md[-1])
    for n, res in summary.items():
        md.append(f"\n## N = {n}\n")
        for dname in ("RIGHT", "LEFT", "INTERNAL"):
            md.append(f"- **{dname}** top by productivity: " + "; ".join(f"`{r['candidate']}` (prod {r['productivity']}, support {r['support']}; e.g. {r['examples'][0]})" for r in res["directions"][dname]["top_by_productivity"][:15]))
        md.append(f"- halves agreement: {res['halves_agreement']}")
    with open(os.path.join(args.out, "root_scan.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(args.out, "root_scan.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
