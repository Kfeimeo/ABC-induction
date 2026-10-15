"""Natural-language study on TinyStories (raw text only, intrinsic evaluation).

    python scripts/prepare_tinystories.py
    python scripts/run_natural.py --sizes 10000 50000 [--punct-ablation]

For every corpus size N: a base run, a seed-variant run, two disjoint halves
(subset stability), a held-out context-prediction test, productivity / MDL
statistics, and (optionally) a punctuation-removed ablation.  Writes
``outputs/natural/natural_summary.json`` and ``outputs/natural/natural_report.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from abc_induction.corpus import tokenize  # noqa: E402
from abc_induction.experiment import load_config, run_experiment  # noqa: E402
from abc_induction.nl_evaluation import (  # noqa: E402
    compare_modifiers_v02,
    modifier_inspection,
    compare_runs,
    compare_runs_v02,
    coverage_v02,
    heldout_context_prediction,
    productivity_summary,
    split_summary,
    top_classes_for_inspection,
    top_classes_v02,
)


def load_sentences(path: str, cfg: dict, punct: str) -> list[list[str]]:
    p = cfg["preprocess"]
    seen = set()
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            toks = tokenize(line.strip(), punct, bool(cfg["dataset"].get("lowercase", False)))
            if not (int(p.get("min_sentence_length", 1)) <= len(toks) <= int(p["max_sentence_length"])):
                continue
            key = tuple(toks)
            if key in seen:
                continue
            seen.add(key)
            out.append(toks)
    return out


def write_lines(path: str, sents: list[list[str]]) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        for s in sents:
            fh.write(" ".join(s) + "\n")
    return path


def run(cfg: dict, path: str, name: str, out_root: str, seed: int = 42, overrides: dict | None = None) -> tuple:
    from abc_induction.experiment import deep_update

    c = json.loads(json.dumps(cfg))
    if overrides:
        c = deep_update(c, overrides)
    c["name"] = name
    c["seed"] = seed
    c["dataset"]["path"] = path
    c["dataset"]["max_sentences"] = None
    c["dataset"]["shuffle"] = False
    c["dataset"]["punctuation_mode"] = "keep"
    c["dataset"]["remove_duplicate_sentences"] = False
    c["output_dir"] = os.path.join(out_root, name)
    t = time.time()
    res = run_experiment(c)
    return res, round(time.time() - t, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "tinystories.yaml"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--sizes", nargs="*", type=int, default=[10000, 50000])
    ap.add_argument("--heldout", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "natural"))
    ap.add_argument("--punct-ablation", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    ap.add_argument("--no-seed-run", action="store_true", help="skip the seed-variant run (v0.1 showed it is uninformative)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    data = args.data or os.path.join(ROOT, cfg["dataset"]["path"])
    os.makedirs(os.path.join(args.out, "_subsets"), exist_ok=True)
    rng = random.Random(int(cfg.get("seed", 42)))
    all_sents = load_sentences(data, cfg, cfg["dataset"].get("punctuation_mode", "separate_tokens"))
    rng.shuffle(all_sents)
    print(f"{len(all_sents)} usable sentences in {data}")
    summary: dict = {"sizes": {}}
    report: list[str] = ["# Natural-language study (TinyStories, raw text only)\n",
                         "No gold syntax exists for this corpus; the numbers below are intrinsic (stability, compression, held-out prediction, productivity, coverage) and are **not** syntax-recovery accuracies. v0.1 numbers come from the v0.1 stage of the same run, v0.2 numbers from the posterior of the v0.2 stage.\n"]
    for n in args.sizes:
        if n + args.heldout > len(all_sents):
            print(f"skipping size {n}: only {len(all_sents)} sentences available")
            continue
        train = all_sents[:n]
        held = all_sents[n:n + args.heldout]
        sub = os.path.join(args.out, "_subsets")
        p_train = write_lines(os.path.join(sub, f"train_{n}.txt"), train)
        p_a = write_lines(os.path.join(sub, f"half_a_{n}.txt"), train[: n // 2])
        p_b = write_lines(os.path.join(sub, f"half_b_{n}.txt"), train[n // 2:])
        rec: dict = {"n_sentences": n}
        base, secs = run(cfg, p_train, f"tinystories_{n}", args.out)
        rec["base_seconds"] = secs
        rec["efficiency"] = base.metrics.get("efficiency")
        rec["mdl"] = base.metrics.get("mdl")
        rec["productivity"] = productivity_summary(base)
        rec["heldout_context_prediction"] = heldout_context_prediction(base, held)
        rec["top_classes"] = top_classes_for_inspection(base)
        rec["top_splits"] = split_summary(base)
        rec["structural_ambiguity"] = base.metrics.get("structural_ambiguity")
        rec["v02"] = coverage_v02(base)
        rec["v02_top_classes"] = top_classes_v02(base)
        rec["modifier_candidates"] = modifier_inspection(base)
        if not args.no_stability:
            if not args.no_seed_run:
                seed_run, _ = run(cfg, p_train, f"tinystories_{n}_seed7", args.out, seed=7)
                rec["seed_stability"] = compare_runs(base, seed_run)
                rec["seed_stability_v02"] = compare_runs_v02(base, seed_run)
            ra, _ = run(cfg, p_a, f"tinystories_{n}_halfA", args.out)
            rb, _ = run(cfg, p_b, f"tinystories_{n}_halfB", args.out)
            rec["subset_stability"] = compare_runs(ra, rb)
            rec["subset_stability_v02"] = compare_runs_v02(ra, rb)
            rec["subset_stability_modifier"] = compare_modifiers_v02(ra, rb)
            rec["halves_v02"] = {"A": coverage_v02(ra), "B": coverage_v02(rb)}
        if args.punct_ablation:
            nopunct = [tokenize(" ".join(s), "remove") for s in train]
            nopunct = [s for s in nopunct if len(s) >= 2]
            p_np = write_lines(os.path.join(sub, f"train_{n}_nopunct.txt"), nopunct)
            rnp, _ = run(cfg, p_np, f"tinystories_{n}_nopunct", args.out)
            rec["punctuation_removed"] = {"efficiency": rnp.metrics.get("efficiency"), "mdl": rnp.metrics.get("mdl"), "productivity": productivity_summary(rnp), "top_classes": top_classes_for_inspection(rnp, 8, 10), "v02": coverage_v02(rnp)}
            rsp, _ = run(cfg, p_train, f"tinystories_{n}_punctprior", args.out, overrides={"punctuation": {"mode": "structural_prior"}})
            rec["punctuation_structural_prior"] = {"v02": coverage_v02(rsp), "v02_top_classes": top_classes_v02(rsp, 8)}
            if not args.no_stability:
                rsa, _ = run(cfg, p_a, f"tinystories_{n}_punctprior_halfA", args.out, overrides={"punctuation": {"mode": "structural_prior"}})
                rsb, _ = run(cfg, p_b, f"tinystories_{n}_punctprior_halfB", args.out, overrides={"punctuation": {"mode": "structural_prior"}})
                rec["punctuation_structural_prior"]["subset_stability_v02"] = compare_runs_v02(rsa, rsb)
        summary["sizes"][str(n)] = rec
        # ---- report section
        eff = rec["efficiency"] or {}
        report.append(f"## {n} sentences\n")
        report.append(f"- spans: {eff.get('n_unique_span_types')} types / {eff.get('n_span_occurrences')} occurrences; contexts {eff.get('n_unique_contexts')}; edges {eff.get('n_span_context_edges')}; merges {eff.get('n_merge_operations')}; accepted classes {eff.get('n_accepted_classes')}; relations {eff.get('n_relations')}; splits {eff.get('n_split_operations')}; runtime {eff.get('runtime_sec', {}).get('total')} s; peak memory {eff.get('peak_memory_mb')} MB")
        if rec["mdl"]:
            report.append(f"- MDL: {rec['mdl'].get('bits_per_sentence')} bits/sentence (grammar {rec['mdl'].get('grammar_bits')} bits), fully derived sentences {rec['mdl'].get('fully_derived_sentences')}/{rec['mdl'].get('n_sentences')} (MDL sample)")
        pr = rec["productivity"]
        report.append(f"- productivity: {pr.get('n_relations')} relations, lexical support mean {pr.get('lexical_support_mean')} / median {pr.get('lexical_support_median')} / max {pr.get('lexical_support_max')}; {pr.get('relations_with_lexical_support_ge_10')} relations with >= 10 distinct lexical realisations")
        ho = rec["heldout_context_prediction"]
        report.append(f"- held-out context prediction (mean log2 P of k=1 context over {ho.get('n_heldout_occurrences')} held-out occurrences): class model {ho.get('mean_log2_likelihood_class')}, type model {ho.get('mean_log2_likelihood_type')}, background {ho.get('mean_log2_likelihood_background')}; rare types (freq<10, n={ho.get('n_rare_type_occurrences')}): class {ho.get('mean_log2_likelihood_class_rare_types')} vs type {ho.get('mean_log2_likelihood_type_rare_types')}; contexts unseen for the type but seen for its class: {ho.get('fraction_context_unseen_for_type_but_seen_for_class')}")
        if "seed_stability" in rec:
            ss, sb = rec["seed_stability"], rec["subset_stability"]
            report.append(f"- stability across seeds: class pairwise F1 {ss.get('class_pairwise_f1')} / ARI {ss.get('class_ari')} on {ss.get('n_types_classified_in_both')} shared classified types; constituent-set agreement (Jaccard) {ss.get('constituent_agreement_jaccard')}; split agreement {ss.get('split_agreement_jaccard')}")
            report.append(f"- stability across disjoint halves: class pairwise F1 {sb.get('class_pairwise_f1')} / ARI {sb.get('class_ari')} on {sb.get('n_types_classified_in_both')} shared classified types; constituent-set agreement {sb.get('constituent_agreement_jaccard')}; split agreement {sb.get('split_agreement_jaccard')} (A: {sb.get('splits_a')[:8]}, B: {sb.get('splits_b')[:8]})")
        if "punctuation_removed" in rec:
            pn = rec["punctuation_removed"]
            report.append(f"- punctuation removed: {pn['efficiency'].get('n_accepted_classes')} classes, {pn['efficiency'].get('n_relations')} relations, {pn['mdl'].get('bits_per_sentence') if pn['mdl'] else None} bits/sentence")
        v = rec.get("v02") or {}
        if v:
            report.append("\n### v0.2 posterior (same corpus)\n")
            report.append(f"- coverage: fully parsed (single root) {v.get('fully_parsed_fraction')}, mean chunks/sentence {v.get('mean_chunks_per_sentence')}, unknown-token mass {v.get('unknown_token_mass')}")
            report.append(f"- complexity: {v.get('active_classes')} active classes, {v.get('active_rules')} active rules, {v.get('lexical_entries')} lexical entries, {v.get('elbo_bits_per_sentence')} bits/sentence (-ELBO/N); pruning {v.get('pruning')}")
            report.append(f"- structure moves: {v.get('n_accepted_moves')}/{v.get('n_proposed_moves')} accepted ({', '.join(v.get('accepted_moves', [])[:8])}); runtime {v.get('runtime_sec')}")
            s2 = rec.get("subset_stability_v02") or {}
            if s2:
                report.append(f"- stability across disjoint halves (v0.2): class pairwise F1 {s2.get('class_pairwise_f1')} / ARI {s2.get('class_ari')} on {s2.get('n_types_classified_in_both')} shared classified types; constituent-set agreement {s2.get('constituent_agreement_jaccard')}; rule agreement (Jaccard after class alignment) {s2.get('rule_agreement_jaccard')} ({s2.get('n_rules_a')} vs {s2.get('n_rules_b')} rules); ambiguity agreement {s2.get('ambiguity_agreement_jaccard')}")
            ss2 = rec.get("seed_stability_v02") or {}
            if ss2:
                report.append(f"- stability across seeds (v0.2): class pairwise F1 {ss2.get('class_pairwise_f1')} / ARI {ss2.get('class_ari')}; constituent agreement {ss2.get('constituent_agreement_jaccard')}; rule agreement {ss2.get('rule_agreement_jaccard')}")
            report.append(f"- occurrence ambiguity: {v.get('n_ambiguous_types')} span types with a second class above the mass threshold; highest posterior entropy (not a linguistic claim): " + "; ".join(f"`{e['span_type']}` ({e['n_occurrences']} occ, H={e['entropy_nats']}, {e['mixture']})" for e in (v.get('highest_entropy_span_types') or [])[:8]))
            sm = rec.get("subset_stability_modifier")
            if sm:
                report.append("- identity-preserving (modifier) candidate agreement across disjoint halves vs general rule agreement (rule Jaccard " + str(s2.get("rule_agreement_jaccard")) + "): " + "; ".join(f"{k}: candidates {v['candidate_jaccard']} ({v['n_shared']} shared of {v['n_a']}/{v['n_b']}), direction {v['direction_agreement']}, host {v['host_agreement']}" for k, v in sm.items()))
            mc = rec.get("modifier_candidates") or {}
            if mc:
                report.append("\nTop identity-preserving candidates (no gold; strings, not linguistic claims):\n")
                for k, rows in mc.items():
                    report.append(f"- **{k}**: " + "; ".join(f"`{r['candidate']}` {r['direction']} of {r['host_class']} (cand. {r['candidate_class']}, support {r['support']}, {r['distinct_hosts']} hosts, mod {r['mod']}; e.g. {r['examples'][0] if r['examples'] else ''})" for r in rows[:8]))
            pr = rec.get("punctuation_structural_prior") or {}
            pn2 = (rec.get("punctuation_removed") or {}).get("v02") or {}
            if pr:
                pv = pr.get("v02", {})
                report.append("- punctuation ablation (v0.2): " + "; ".join([
                    f"as token: parsed {v.get('fully_parsed_fraction')}, {v.get('active_classes')} classes / {v.get('active_rules')} rules, {v.get('elbo_bits_per_sentence')} bits/sent",
                    f"removed: parsed {pn2.get('fully_parsed_fraction')}, {pn2.get('active_classes')} classes / {pn2.get('active_rules')} rules, {pn2.get('elbo_bits_per_sentence')} bits/sent",
                    f"structural prior: parsed {pv.get('fully_parsed_fraction')}, {pv.get('active_classes')} classes / {pv.get('active_rules')} rules, {pv.get('elbo_bits_per_sentence')} bits/sent",
                ]))
                sp2 = pr.get("subset_stability_v02") or {}
                if sp2:
                    report.append(f"- structural-prior stability across halves: class F1 {sp2.get('class_pairwise_f1')} / ARI {sp2.get('class_ari')}; constituent agreement {sp2.get('constituent_agreement_jaccard')}; rule agreement {sp2.get('rule_agreement_jaccard')}")
            report.append("\nTop v0.2 classes (posterior view):\n")
            for c in rec.get("v02_top_classes", [])[:12]:
                report.append(f"- {c['name']} (mass {c['mass']}): lexical {c['lexical'][:10]}; spans {c['spans'][:4]}; contexts {c['contexts'][:2]}; composition in {c['in'][:2]}")
        report.append("\nTop inferred classes (human inspection):\n")
        for c in rec["top_classes"][:12]:
            report.append(f"- C{c['class']} ({c['n_members']} members, {c['n_occurrences']} occ): lexical {c['lexical_members'][:12]}; multi-token {c['multi_token_members'][:6]}; contexts {c['top_contexts'][:3]}")
        report.append("\nTop relations A + B -> C:\n")
        for lab in pr.get("top_relations", [])[:12]:
            report.append(f"- {lab}")
        if rec["top_splits"]:
            report.append("\nTop ambiguity splits:\n")
            for s in rec["top_splits"][:10]:
                report.append(f"- `{s['span']}` ({s['n']} occ): {s['a'][0]} x{s['a'][1]} {s['a'][2][:2]} vs {s['b'][0]} x{s['b'][1]} {s['b'][2][:2]}; split gain {s['split_gain']} bits")
        report.append("")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "natural_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(args.out, "natural_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
