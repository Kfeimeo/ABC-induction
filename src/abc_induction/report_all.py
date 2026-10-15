"""Project-level ``report.md`` aggregating every run of the suite."""
from __future__ import annotations

from typing import Any


def _g(d: dict, *path: str, default: Any = None) -> Any:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _tbl(rows: list[list[Any]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(out)


COLS = [
    ("class pair F1", ("class_recovery_type_level", "f1")),
    ("class ARI", ("class_recovery_type_level", "ari")),
    ("occ. class F1", ("class_recovery_occurrence_level", "f1")),
    ("#classes", ("class_recovery_type_level", "n_pred_classes")),
    ("const F1", ("constituents_derivation_level", "all", "f1")),
    ("const F1 len>=2", ("constituents_derivation_level", "len_ge_2", "f1")),
    ("const P len>=2", ("constituents_derivation_level", "len_ge_2", "precision")),
    ("comp P", ("composition_recovery", "precision")),
    ("comp R", ("composition_recovery", "recall")),
    ("comp F1", ("composition_recovery", "f1")),
    ("amb F1", ("ambiguity_detection", "f1")),
    ("amb occ acc", ("ambiguity_detection", "occurrence_assignment_accuracy")),
    ("bits/sent", ("mdl", "bits_per_sentence")),
    ("runtime s", ("efficiency", "runtime_sec", "total")),
]


def _row(name: str, m: dict) -> list[Any]:
    return [name] + [_g(m, *path) for _, path in COLS]


COLS_V02 = [
    ("class pair F1", ("class_recovery_type_level", "f1")),
    ("class ARI", ("class_recovery_type_level", "ari")),
    ("occ. class F1", ("class_recovery_occurrence_level", "f1")),
    ("#classes", ("class_recovery_type_level", "n_pred_classes")),
    ("const F1", ("constituents_derivation_level", "all", "f1")),
    ("const F1 len>=2", ("constituents_derivation_level", "len_ge_2", "f1")),
    ("comp P", ("composition_recovery", "precision")),
    ("comp R", ("composition_recovery", "recall")),
    ("comp F1", ("composition_recovery", "f1")),
    ("amb F1", ("ambiguity_detection", "f1")),
    ("amb occ acc", ("ambiguity_detection", "occurrence_assignment_accuracy")),
    ("bits/sent (ELBO)", ("posterior", "bits_per_sentence")),
    ("#rules", ("posterior", "n_rules")),
    ("moves acc/prop", ("moves",)),
    ("v0.2 runtime s", ("runtime_sec", "total")),
]


COLS_MOD = ["ROOT P/R", "CONST P/R", "CLASS P/R", "RULE P/R", "COMBINED P/R", "root LEFT/RIGHT P", "dir acc", "host acc (class)", "ins P/R (root)", "lex_add acc/prop", "false lex", "true amb lex", "rejected by penalty"]


def _pr(x: dict, occ: bool = True) -> str:
    c = x.get("constituent_candidates_only", {})
    op = (x.get("occurrence_precision", {}).get("ALL") or {}).get("precision")
    return f"{_fmt(c.get('precision'))}/{_fmt(c.get('recall'))}" + (f" [{_fmt(op)}]" if occ and op is not None else "")


def _row_modifier(name: str, m2: dict) -> list[Any]:
    me = m2["modifier"]["by_source"]
    g = lambda src: me.get(src, {}).get("support>=2", {})
    root = g("ROOT_STRING")
    le = m2.get("lexicon_expansion", {})
    ld = m2.get("lexical_distinguishability", {})
    cls_ = g("CLASS_GENERALIZED")
    return [name, _pr(root), _pr(g("CONSTITUENT_STRING")), _pr(cls_), _pr(g("POSTERIOR_RULE")), _pr(g("COMBINED")),
            f"{_fmt(root.get('LEFT', {}).get('precision'))}/{_fmt(root.get('RIGHT', {}).get('precision'))}",
            g("COMBINED").get("direction_accuracy"), cls_.get("host_accuracy"),
            f"{_fmt(root.get('insertion', {}).get('precision'))}/{_fmt(root.get('insertion', {}).get('recall'))}",
            f"{ld.get('n_accepted', 0)}/{ld.get('n_lex_add_proposals', 0)}", le.get("n_false"), le.get("n_true"), ld.get("n_rejected_by_duplicate_penalty")]


def _row_v02(name: str, m2: dict, n_sent: int) -> list[Any]:
    m2 = dict(m2)
    post = dict(m2.get("posterior", {}))
    elbo = (m2.get("objective") or {}).get("elbo")
    post["bits_per_sentence"] = (-elbo / max(1, n_sent) / 0.6931) if elbo is not None else None
    m2["posterior"] = post
    m2["moves"] = f"{m2.get('n_accepted_moves')}/{m2.get('n_structure_moves')}"
    rt = dict(m2.get("runtime_sec") or {})
    rt["total"] = sum(v for v in rt.values() if isinstance(v, (int, float)))
    m2["runtime_sec"] = rt
    return [name] + [_g(m2, *path) for _, path in COLS_V02]


def build_global_report(summary: dict[str, Any], analysis_md: str = "", natural_md: str = "") -> str:
    runs = summary["runs"]
    L: list[str] = []
    title = summary.get("title") or "ABC induction v0.2 — experiment report"
    L.append(f"# {title}\n")
    L.append("Pipeline v0.1: raw corpus -> candidate spans -> multi-scale sparse contexts -> provisional classes (agglomerative, sparsity-corrected similarity) -> constituency score + MDL elimination with a predictive chart -> occurrence-level (legacy) ambiguity splits -> (E, R).\n")
    L.append("Pipeline v0.2 (on top of the v0.1 result): finite Bayesian latent grammar (sparse PCFG, Dirichlet posteriors) -> sparse Inside-Outside (semiring chart, numba) -> Dirichlet CAVI with a computable ELBO -> objective F = ELBO + log P(G) -> greedy class merge / class split / rule and lexicon proposals with checkpoint + rollback; lexical contexts as leaf potentials, punctuation as span potentials, adaptive constituent contexts as proposal features only. Every run reports both stages on the same gold (`v0.1` rows = v0.1 metrics, `v0.2` rows = metrics of the v0.2 posterior).\n")
    L.append("Metric columns: class pair F1 / ARI are computed over gold-constituent span types (type level), occurrence-level F1 over gold-constituent occurrences; constituent F1 is span-level over all gold spans (derivation level, i.e. the nodes of the induced grammar's Viterbi derivations); composition P/R/F1 compare R(A,B,C) after Hungarian class alignment; ambiguity F1 is over gold-ambiguous span types and occurrence accuracy is measured on their occurrences.\n")
    failed = [n for n, r in runs.items() if not r.get("ok")]
    if failed:
        L.append(f"**Failed runs:** {', '.join(failed)}\n")
    for g, grp in summary["groups"].items():
        L.append(f"## {grp['title']}\n")
        rows = []
        for name in grp["experiments"]:
            r = runs.get(name)
            if not r or not r.get("ok"):
                rows.append([name] + ["FAILED"] + [None] * (len(COLS) - 1))
                continue
            rows.append(_row(name, r.get("metrics", {})))
        L.append(_tbl(rows, ["run"] + [c for c, _ in COLS]))
        L.append("")
        rows2 = []
        for name in grp["experiments"]:
            r = runs.get(name)
            if not r or not r.get("ok"):
                continue
            m2 = r.get("metrics", {}).get("v02")
            if m2:
                rows2.append(_row_v02(name, m2, _g(r.get("metrics", {}), "efficiency", "n_sentences", default=1)))
        if rows2:
            L.append("v0.2 posterior on the same runs:\n")
            L.append(_tbl(rows2, ["run"] + [c for c, _ in COLS_V02]))
            L.append("")
        rows3 = []
        for name in grp["experiments"]:
            r = runs.get(name)
            m2 = (r or {}).get("metrics", {}).get("v02") if r and r.get("ok") else None
            if m2 and m2.get("modifier"):
                rows3.append(_row_modifier(name, m2))
        if rows3:
            L.append("Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:\n")
            L.append(_tbl(rows3, ["run"] + COLS_MOD))
            L.append("")
        # per-run links and extra details
        for name in grp["experiments"]:
            r = runs.get(name)
            if not r or not r.get("ok"):
                continue
            m = r["metrics"]
            eff = m.get("efficiency", {})
            cr = m.get("composition_recovery", {})
            am = m.get("ambiguity_detection", {})
            bits = []
            bits.append(f"[{name}/report.md]({r['output_dir'].replace(chr(92), '/')}/report.md)")
            bits.append(f"{eff.get('n_sentences')} sentences, {eff.get('n_unique_span_types')} span types, {eff.get('n_unique_contexts')} contexts, {eff.get('n_span_context_edges')} edges, peak {eff.get('peak_memory_mb')} MB, {eff.get('n_merge_operations')} merges, {eff.get('n_split_operations')} splits")
            if cr:
                bits.append(f"relations recovered: {', '.join(cr.get('recovered', []))}" + (f"; missed: {', '.join(cr.get('missed', []))}" if cr.get("missed") else ""))
            if am and am.get("n_gold"):
                bits.append(f"ambiguity: missed {am.get('missed_splits')}, false {am.get('false_splits')}")
            m2 = m.get("v02")
            if m2:
                cr2 = m2.get("composition_recovery", {})
                bits.append(f"v0.2: {m2.get('n_accepted_moves')}/{m2.get('n_structure_moves')} moves accepted ({', '.join(m2.get('accepted_moves', [])[:8])}); relations {', '.join(cr2.get('recovered', []))}" + (f"; missed {', '.join(cr2.get('missed', []))}" if cr2.get("missed") else "") + (f"; false {[x['mapped'] for x in cr2.get('false_relations', [])[:4]]}" if cr2.get("false_relations") else ""))
                am2 = m2.get("ambiguity_detection", {})
                if am2.get("n_gold"):
                    bits.append(f"v0.2 ambiguity: missed {am2.get('missed_splits')}, false {am2.get('false_splits')}, occurrence accuracy {am2.get('occurrence_assignment_accuracy')}")
            L.append("- " + "; ".join(bits))
        L.append("")
    # ---- failure examples from the main runs
    L.append("## Key failure cases (from the main runs)\n")
    for name in ("synthetic1_clean", "synthetic2_ambiguity", "synthetic3_noisy"):
        r = runs.get(name)
        if not r or not r.get("ok"):
            continue
        dg = r.get("diagnostics", {})
        m = r["metrics"]
        L.append(f"### {name}\n")
        fm = dg.get("false_merges", [])
        L.append(f"- false merges: {dg.get('n_false_merges', 0)}" + (" — e.g. " + "; ".join(f"{f['gold_a']}{f['members_a'][:2]} + {f['gold_b']}{f['members_b'][:2]} (score {f['score']}, why: {f['why_merged'][:2]})" for f in fm[:3]) if fm else ""))
        mm = dg.get("missed_merges", [])
        if mm:
            L.append("- fragmented gold classes: " + "; ".join(f"{x['gold_class']} -> " + ", ".join(f"{fr['inferred_class']}({fr['n_types']})" for fr in x['fragments'][:4]) for x in mm[:4]))
        fc = dg.get("false_constituents", [])
        L.append(f"- false constituent types: {dg.get('n_false_constituent_types', 0)}" + (" — e.g. " + ", ".join(f"`{x['span']}` (freq {x['freq']}, in class ~{x['class_gold']})" for x in fc[:5]) if fc else ""))
        mc = dg.get("missed_constituents", [])
        L.append(f"- missed constituent types: {dg.get('n_missed_constituent_types', 0)}" + (" — e.g. " + ", ".join(f"`{x['span']}` ({x['gold_class']})" for x in mc[:5]) if mc else ""))
        fr_ = _g(m, "composition_recovery", "false_relations", default=[])
        if fr_:
            L.append("- false relations: " + "; ".join(f"{x['mapped']} (support {x['support']}, lexical {x['lexical']})" for x in fr_[:5]))
        am = m.get("ambiguity_detection", {})
        if am.get("per_word"):
            L.append("- ambiguity per word: " + "; ".join(f"`{w}` acc {v['accuracy']} (split {'accepted' if v['split_accepted'] else 'rejected'}, gold {v['gold_split']})" for w, v in am["per_word"].items()))
        dg2 = r.get("diagnostics_v02") or {}
        if dg2:
            L.append("- v0.2 failure modes: fragmentation " + "; ".join(f"{g}: {d['n_fragments_ge_5pct']}" for g, d in dg2.get("fragmentation", {}).items()) + f"; extra root classes {dg2.get('extra_root_classes')}; private crutch rules {len(dg2.get('private_crutch_rules', []))}; filler-induced rules {len(dg2.get('filler_induced_rules', []))}; filler classes {dg2.get('filler_token_classes')}; ambiguity pollution {dg2.get('ambiguity_pollution_class_purity')}")
            cont = dg2.get("contamination", {})
            low = [(c, d["purity"], d["second"]) for c, d in cont.items() if d["purity"] < 0.95]
            if low:
                L.append(f"- v0.2 contaminated classes (purity < 0.95): {low[:6]}")
        L.append("")
    if natural_md:
        L.append(natural_md)
        L.append("")
    if analysis_md:
        L.append(analysis_md)
    return "\n".join(L)
