"""End-to-end experiment runner.

    v0.1  raw corpus -> spans -> contexts -> provisional classes -> constituency/MDL
                     -> legacy ambiguity splits -> (E, R) -> evaluation
    v0.2  -> GrammarModel (warm start) -> CAVI -> class merge/split, rule & lexicon
             proposals -> posterior decoding -> evaluation (``metrics["v02"]``)

Every run writes ``config.yaml, metrics.json, classes.json, relations.json,
ambiguities.json, merge_history.jsonl, structural_ambiguity.json, run.log``
(v0.1) plus ``objective_trace.csv, structure_moves.jsonl, posterior_stats.json,
classes_v02.json, relations_v02.json, ambiguity_v02.json`` (v0.2) and a
human-readable ``report.md`` into ``output_dir``.  The v0.1 stage can be cached
(``v01_cache``) because every v0.2 ablation shares it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import pickle
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil
import yaml

from .ambiguity import AmbiguityDetector, SplitReport
from .clustering import ClassState, agglomerate
from .composition import CompositionRelation, collect_composition, structural_ambiguity_report
from .constituency import ConstituencyResult, select_constituents
from .contexts import ContextData, extract_contexts
from .corpus import Corpus
from .evaluation import GoldInfo, build_gold, diagnostics, evaluate
from .spans import SpanInventory, enumerate_spans
from .synthetic import SyntheticDataset, load_grammar

log = logging.getLogger("abc_induction")

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "run",
    "seed": 42,
    "dataset": {"kind": "synthetic", "grammar": "grammars/synthetic1_clean.yaml", "n_sentences": 3000},
    "preprocess": {"max_sentence_length": 15, "min_sentence_length": 1, "max_span_length": 8, "min_span_frequency": 2},
    "contexts": {"scales": [1, 2, 3, 4], "one_sided": False, "min_context_count": 1},
    "clustering": {
        "similarity": "bhattacharyya",
        "merge_threshold": 0.7,
        "min_shared_contexts": 2,
        "max_context_df": 500,
        "scale_weights": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        "sparsity_correction": True,
        "self_floor": 0.05,
        "min_raw_similarity": 0.0,
        "sided_weight": 1.0,
        "max_rounds": 200,
        "evidence_contexts": 8,
        "attach": {"enabled": True, "threshold": 0.8, "margin": 0.2, "min_occurrences": 2, "iterations": 2},
    },
    "constituency": {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "lambda_complexity": 1.0,
        "min_score": 0.2,
        "min_relation_support": 2,
        "min_side_realisations": 1,
        "singleton_penalty": 0.5,
        "fixed_phrase_penalty": 0.5,
        "productivity_scale": 10.0,
    },
    "mdl": {"enabled": True, "em_iters": 2, "em_iters_after": 1, "max_passes": 6, "margin_bits": 0.0, "max_sentences": None, "max_sentences_per_eval": 400, "costs": {}},
    "ambiguity": {"enabled": True, "max_scale": 2, "lambda_split": 20.0, "min_occurrences": 8, "min_cluster_occurrences": 3, "min_cluster_fraction": 0.1, "min_other_members": 2, "min_alignment": 0.5, "min_ll_gain": 0.0, "max_span_length": 1, "proposal": "guided", "reattach_singletons": True},
    "relations": {"eta": 0.5, "min_support": 2, "min_lexical_support": 2, "min_side_realisations": 2},
    # ------------------------------------------------------------------ v0.2
    "probabilistic": {
        "enabled": True,
        "init": "v01",  # v01 | random (robustness ablation)
        "random_k": 8,
        "k_max": 0,  # 0 = no cap on the number of classes carried over from v0.1
        "alpha_rule": 0.5,
        "gamma_type": 0.5,  # = alpha_lex: Dirichlet pseudo-count of lexical expansions
        "alpha_unk": 0.1,
        "alpha_root": 0.5,
        "unk_enabled": True,
        "chunks_enabled": True,  # chunk-sequence fallback for sentences the grammar cannot derive as one root (p_cont learned)
        "max_cavi_iterations": 50,
        "cavi_tolerance": 1e-5,
        "constituent_threshold": 0.5,
        "ambiguity_mass": 0.2,
        "ambiguity_min_occurrences": 5,
        "ambiguity_max_span_length": 1,
        "posterior_output_threshold": 1e-3,
    },
    "structure_prior": {"lambda_class": 8.0, "lambda_rule": 7.0, "lambda_lex": 2.0, "lambda_private": 20.0, "private_tau": 0.6, "private_min_count": 5},
    "structure_learning": {
        "enabled": True,
        "merge_enabled": True,
        "split_enabled": True,
        "rule_proposals_enabled": True,
        "merge_threshold": 0.0,
        "split_threshold": 5.0,
        "rule_threshold": 0.0,
        "proposal_vb_iterations": 5,
        "proposal_eval": "full",  # full | affected
        "affected_threshold": 1e-3,
        "max_rounds": 6,
        "max_accepts_per_stage": 2,
        "max_proposals_per_stage": 12,
        "max_merge_candidates": 8,
        "merge_min_similarity": 0.3,
        "max_split_candidates": 5,
        "split_min_occurrences": 20,
        "split_min_separation": 0.05,
        "split_feature_weights": {"context": 1.0, "composition": 3.0, "abstract": 2.0},
        "rule_prune_count": 0.5,
        "max_rule_add_pairs": 5,
        "rule_add_min_count": 5.0,
        "max_lex_add_candidates": 6,
        "lex_add_min_share": 0.15,
        "lex_add_min_occurrences": 8,
        "post_move_cavi_iterations": 10,
        "objective_tolerance": 1e-6,
        "rejection_memory": False,  # opt-in: skip proposals rejected by > rejection_margin until a structural move touches their classes (changes which moves get accepted; see docs/profiling.md)
        "rejection_margin": 5.0,
    },
    "lexical_context": {"enabled": True, "proposals": True, "max_scale": 2, "temperature": 1.0, "top_k": 5, "floor": -6.0, "backoff_eta": 1.0},
    "adaptive_context": {"enabled": True, "constituent_posterior_threshold": 0.9, "centre_posterior_threshold": 0.5, "max_refinement_depth": 2, "top_features": 20},
    "punctuation": {"mode": "punctuation_as_token", "sentence_final_weight": 2.0, "paired_weight": 2.0, "comma_weight": 1.0},
    "chart": {"backend": "numba", "exact": True, "max_classes_per_cell": 32, "posterior_prune_threshold": 1e-6},
    "modifier": {  # identity-preserving constituent evidence (docs/modifier_study.md)
        "mode": "off",  # off | string_only | string_plus_class | posterior_rule_only : evidence sources that feed rule proposals
        "diagnostics": True,
        "max_modifier_length": 3,
        "max_insert_span_length": 4,
        "constituent_threshold": 0.8,
        "min_support": 2,
        "min_mod": 0.5,
        "max_rule_proposals": 6,
        "proposal_min_support": 3,
        # duplicate-entry penalty on lexicon additions: lambda_dup * exp(-D_struct / tau);
        # combine = sum (D = l_c D_comp + l_m D_mod), gated (D = D_cover * mean profile distance), cover (D = D_cover)
        "distinguishability": {"enabled": False, "combine": "gated", "lambda_comp": 1.0, "lambda_modifier": 1.0, "lambda_dup": 10.0, "tau": 0.1, "min_mass": 1.0},
    },
    "problog_reference": {"enabled": False},
    "v01_cache": True,
    "output_dir": "outputs/run",
}

V01_KEYS = ("seed", "dataset", "preprocess", "contexts", "clustering", "constituency", "mdl", "ambiguity", "relations")


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = deep_update(DEFAULT_CONFIG, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    if "legacy_ambiguity_split" in cfg:  # alias of the v0.1 "ambiguity" block
        cfg["ambiguity"] = deep_update(cfg.get("ambiguity", {}), cfg["legacy_ambiguity_split"])
    return cfg


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def v01_cache_key(cfg: dict[str, Any]) -> str:
    sub = {k: cfg.get(k) for k in V01_KEYS}
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


@dataclass
class RunResult:
    cfg: dict[str, Any]
    corpus: Corpus
    inv: SpanInventory
    cd: ContextData
    state: ClassState
    const: ConstituencyResult
    occ_class: np.ndarray
    type_classes: dict[int, list[int]]
    relations: list[CompositionRelation]
    splits: list[SplitReport]
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    gold: GoldInfo | None
    dataset: SyntheticDataset | None
    timings: dict[str, float] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)
    v02: Any | None = None  # v02_stage.ProbabilisticResult
    metrics_v02: dict[str, Any] = field(default_factory=dict)
    diagnostics_v02: dict[str, Any] = field(default_factory=dict)


def _setup_logging(out_dir: str) -> logging.FileHandler:
    os.makedirs(out_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"), mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(fh)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    return fh


def load_dataset(cfg: dict[str, Any]) -> tuple[Corpus, SyntheticDataset | None]:
    d = cfg["dataset"]
    p = cfg["preprocess"]
    if d.get("kind", "synthetic") == "synthetic":
        grammar = load_grammar(d["grammar"])
        ds = grammar.generate(int(d.get("n_sentences", 3000)), seed=int(cfg.get("seed", 42)))
        corpus = Corpus.from_token_lists(ds.token_lists(), min_sentence_length=1, max_sentence_length=10**9)
        return corpus, ds
    corpus = Corpus.from_text_file(
        d["path"],
        punctuation_mode=d.get("punctuation_mode", "separate_tokens"),
        lowercase=bool(d.get("lowercase", False)),
        min_sentence_length=int(p.get("min_sentence_length", 1)),
        max_sentence_length=int(p.get("max_sentence_length", 15)),
        remove_duplicate_sentences=bool(d.get("remove_duplicate_sentences", True)),
        max_sentences=d.get("max_sentences"),
    )
    if d.get("shuffle", False):
        rng = random.Random(int(cfg.get("seed", 42)))
        toks = [list(s.tokens) for s in corpus.sentences]
        rng.shuffle(toks)
        corpus = Corpus.from_token_lists(toks, max_sentence_length=10**9)
    return corpus, None


def run_experiment(cfg: dict[str, Any], write_outputs: bool = True) -> RunResult:
    """Run the whole pipeline for one configuration."""
    out_dir = cfg.get("output_dir", "outputs/run")
    fh = _setup_logging(out_dir) if write_outputs else None
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    timings: dict[str, float] = {}
    proc = psutil.Process(os.getpid())
    t_all = time.time()
    try:
        log.info("=== experiment %s ===", cfg.get("name"))
        cache_path = None
        if cfg.get("v01_cache", False):
            cache_dir = os.path.join(os.path.dirname(out_dir.rstrip("/\\")) or ".", "_cache_v01")
            cache_path = os.path.join(cache_dir, v01_cache_key(cfg) + ".pkl")
        stage = None
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    stage = pickle.load(fh)
                log.info("v0.1 stage loaded from cache %s", cache_path)
            except Exception as exc:  # pragma: no cover
                log.warning("could not load v0.1 cache (%s); recomputing", exc)
                stage = None
        if stage is None:
            stage = run_v01_stage(cfg, timings)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as fh:
                    pickle.dump(stage, fh, protocol=pickle.HIGHEST_PROTOCOL)
        corpus, ds, inv, cd, state, const, occ_class, splits, type_classes = stage[:9]
        timings.update(stage_timings(stage, timings))
        p = cfg["preprocess"]
        # ---- final relations (v0.1)
        t = time.time()
        evidence = collect_composition(inv, occ_class)
        r = cfg["relations"]
        rels = evidence.relations(float(r.get("eta", 0.5)), max(1, len(const.accepted)), int(r.get("min_support", 2)), int(r.get("min_lexical_support", 2)), int(r.get("min_side_realisations", 2)))
        rels = [x for x in rels if x.left_class in const.accepted and x.right_class in const.accepted and x.result_class in const.accepted]
        if const.mdl is not None:
            live = const.mdl.grammar.rules
            rels = [x for x in rels if (x.result_class, x.left_class, x.right_class) in live]
        timings["relations"] = time.time() - t
        struct = structural_ambiguity_report(inv, evidence)
        # ---- evaluation
        gold = None
        metrics: dict[str, Any] = {}
        diag: dict[str, Any] = {}
        split_tids = {s.tid: s.to_json() for s in splits if s.accepted and s.kind == "split"}
        if ds is not None:
            gold = build_gold(ds, corpus, inv, int(r.get("min_support", 2)))
            metrics = evaluate(gold, inv, state, type_classes, occ_class, const.constituent_mask, rels, split_tids, int(p["min_span_frequency"]), const.constituent_positions)
            mapping = {int(k): v for k, v in metrics["class_alignment"].items()}
            diag = diagnostics(gold, inv, state, type_classes, const.reports, const.accepted, state.history, mapping)
        else:
            mapping = {}
        # ---- v0.2 probabilistic stage
        v02 = None
        metrics_v02: dict[str, Any] = {}
        diag_v02: dict[str, Any] = {}
        pcfg = cfg.get("probabilistic") or {}
        if pcfg.get("enabled", False):
            from .v02_stage import run_probabilistic_stage
            from .evaluation_v02 import evaluate_v02, failure_modes_v02

            t = time.time()
            v02 = run_probabilistic_stage(cfg, corpus, inv, cd, const.accepted, const.reports, const.mdl, type_classes, occ_class)
            timings["v02_total"] = time.time() - t
            timings.update({f"v02_{k}": v for k, v in v02.timings.items()})
            if gold is not None:
                metrics_v02 = evaluate_v02(gold, inv, state, v02, int(r.get("min_support", 2)), int(p["min_span_frequency"]))
                diag_v02 = failure_modes_v02(gold, inv, v02, metrics_v02, ds)
            metrics_v02["posterior"] = {k: v for k, v in v02.posterior_stats.items() if k not in ("classes", "highest_entropy_span_types", "occurrence_ambiguity", "private_rules")}
            metrics_v02["objective"] = v02.objective
            metrics_v02["initial_objective"] = v02.initial_objective
            metrics_v02["init_summary"] = v02.init_summary
            metrics_v02["potentials"] = v02.potentials_info
            metrics_v02["n_structure_moves"] = len(v02.moves)
            metrics_v02["n_accepted_moves"] = sum(1 for m in v02.moves if m.get("accepted"))
            metrics_v02["accepted_moves"] = [f"{m['kind']}({m.get('a_name') or m.get('rule') or m.get('token') or m.get('n_rules') or ''}{(' + ' + str(m.get('b_name'))) if m.get('b_name') else ''})" for m in v02.moves if m.get("accepted")]
            metrics_v02["runtime_sec"] = {k: round(v, 2) for k, v in v02.timings.items()}
            metrics_v02["monotonicity_violations"] = v02.learner.monotonicity_violations if v02.learner else 0
            metrics_v02["n_memo_skipped_proposals"] = v02.learner.n_memo_skipped if v02.learner else 0
            if v02.modifier_table is not None:
                metrics_v02["modifier_stats"] = v02.modifier_table.stats
                if ds is not None:
                    from .modifier_eval import evaluate_modifiers, gold_modifiers

                    gm = gold_modifiers(ds)
                    m2map = {int(k): v for k, v in metrics_v02.get("class_alignment", {}).items()}
                    metrics_v02["modifier"] = evaluate_modifiers(v02.modifier_table, gm, m2map, float((cfg.get("modifier") or {}).get("min_mod", 0.5)))
            metrics_v02["lexical_distinguishability"] = {
                "n_lex_add_proposals": len(v02.lexical_distinguishability),
                "n_accepted": sum(1 for r in v02.lexical_distinguishability if r.get("accepted")),
                "n_rejected_by_duplicate_penalty": sum(1 for r in v02.lexical_distinguishability if r.get("rejected_by_penalty")),
            }
            if ds is not None:
                metrics_v02["lexicon_expansion"] = lexicon_expansion_vs_gold(v02, ds, {int(k): v for k, v in metrics_v02.get("class_alignment", {}).items()})
        timings["total"] = time.time() - t_all
        mem = proc.memory_info()
        peak = getattr(mem, "peak_wset", mem.rss)
        metrics["git_commit"] = git_commit()
        metrics["seed"] = seed
        metrics["efficiency"] = {
            "n_sentences": len(corpus),
            "n_tokens": corpus.n_tokens,
            "vocab": len(corpus.itos),
            "n_candidate_spans": inv.n_candidate_spans_total,
            "n_candidate_span_types": inv.n_candidate_types_total,
            "n_span_occurrences": inv.n_occurrences,
            "n_unique_span_types": inv.n_types,
            "n_unique_contexts": cd.n_contexts,
            "n_span_context_edges": cd.n_edges,
            "peak_memory_mb": round(peak / 1e6, 1),
            "runtime_sec": {k: round(v, 2) for k, v in timings.items()},
            "n_merge_operations": len(state.history),
            "n_provisional_classes": len(state.members),
            "n_accepted_classes": len(const.accepted),
            "n_relations": len(rels),
            "n_split_operations": len(split_tids),
            "n_split_tests": len(splits),
            "n_mdl_removed_classes": sum(1 for rp in const.reports.values() if rp.status == "rejected_mdl"),
            "n_mdl_removed_relations": len(const.removed_relations),
        }
        if const.mdl is not None:
            metrics["mdl"] = const.mdl.summary()
        metrics["structural_ambiguity"] = {k: v for k, v in struct.items() if k != "examples"}
        class_names = {c: f"C{c}" + (f"~{mapping[c]}" if c in mapping else "") for c in const.accepted}
        if v02 is not None:
            metrics["v02"] = metrics_v02
            m2 = {int(k): v for k, v in metrics_v02.get("class_alignment", {}).items()}
            v02.class_names = {c: f"{v02.model.names.get(c, 'C%d' % c)}" + (f"~{m2[c]}" if c in m2 else "") for c in v02.model.classes}
        result = RunResult(cfg, corpus, inv, cd, state, const, occ_class, type_classes, rels, splits, metrics, diag, gold, ds, timings, class_names, v02, metrics_v02, diag_v02)
        if write_outputs:
            write_run_outputs(result, struct, out_dir)
        return result
    finally:
        if fh is not None:
            logging.getLogger().removeHandler(fh)
            fh.close()


def lexicon_expansion_vs_gold(v02: Any, ds: SyntheticDataset, mapping: dict[int, str]) -> dict[str, Any]:
    """Accepted ``lex_add`` moves judged against the gold lexicon: an added entry ``(B, w)`` is *true*
    when ``w`` belongs to the gold class aligned with ``B`` (an ambiguous word's second category), and
    *false* otherwise (a duplicate analysis of a word that has one gold category)."""
    gold_lex: dict[str, set[str]] = {}
    for cname, cspec in (ds.config.get("classes") or {}).items():
        for w in ((cspec or {}).get("lexicon") or {}):
            gold_lex.setdefault(str(w), set()).add(cname)
    for w, dist in (ds.ambiguous_words or {}).items():
        gold_lex.setdefault(w, set()).update(dist.keys())
    model_classes = {v02.model.names.get(c, f"C{c}"): c for c in v02.model.classes}
    true_, false_, unaligned = [], [], []
    for r in v02.lexical_distinguishability:
        if not r.get("accepted"):
            continue
        w = r["word"]
        cid = model_classes.get(r["new_class"])
        g = mapping.get(cid) if cid is not None else None
        if g is None:
            unaligned.append(f"{w}->{r['new_class']}")
        elif g in gold_lex.get(w, set()):
            true_.append(f"{w}->{g}")
        else:
            false_.append(f"{w}->{g}")
    return {"n_accepted": len(true_) + len(false_) + len(unaligned), "n_true": len(true_), "n_false": len(false_), "n_unaligned": len(unaligned), "true": true_, "false": false_, "unaligned": unaligned}


def run_v01_stage(cfg: dict[str, Any], timings: dict[str, float]) -> tuple:
    """The v0.1 pipeline up to the legacy ambiguity splits (cacheable)."""
    t = time.time()
    corpus, ds = load_dataset(cfg)
    timings["load"] = time.time() - t
    log.info("corpus: %d sentences, %d tokens, vocab %d", len(corpus), corpus.n_tokens, len(corpus.itos))
    p = cfg["preprocess"]
    t = time.time()
    inv = enumerate_spans(corpus, int(p["max_span_length"]), int(p["min_span_frequency"]))
    timings["spans"] = time.time() - t
    log.info("spans: %d types, %d occurrences (of %d candidate spans / %d candidate types)", inv.n_types, inv.n_occurrences, inv.n_candidate_spans_total, inv.n_candidate_types_total)
    c = cfg["contexts"]
    t = time.time()
    cd = extract_contexts(corpus, inv, tuple(int(k) for k in c["scales"]), bool(c.get("one_sided", False)), int(c.get("min_context_count", 1)))
    timings["contexts"] = time.time() - t
    log.info("contexts: %d unique, %d span-context edges", cd.n_contexts, cd.n_edges)
    t = time.time()
    state = agglomerate(cd, inv, cfg["clustering"])
    timings["clustering"] = time.time() - t
    log.info("clustering: %d provisional classes, %d merge events", len(state.members), len(state.history))
    t = time.time()
    const = select_constituents(state, inv, cd, cfg["constituency"], cfg.get("mdl"))
    timings["constituency"] = time.time() - t
    log.info("constituency: %d accepted classes", len(const.accepted))
    occ_class = const.occ_class.copy()
    splits: list[SplitReport] = []
    type_classes: dict[int, list[int]] = {}
    for cid in const.accepted:
        for tid in const.reports[cid].members:
            type_classes.setdefault(tid, []).append(cid)
    amb_cfg = cfg.get("ambiguity") or {}
    t = time.time()
    if amb_cfg.get("enabled", False):
        det = AmbiguityDetector(inv, cd, state, const.reports, const.accepted, occ_class, const.mdl, amb_cfg)
        splits = det.run()
        type_classes = det.type_classes
        occ_class = det.occ_class
    timings["ambiguity"] = time.time() - t
    stage = (corpus, ds, inv, cd, state, const, occ_class, splits, type_classes)
    return stage + (dict(timings),)


def stage_timings(stage: tuple, timings: dict[str, float]) -> dict[str, float]:
    return stage[9] if len(stage) > 9 else {}


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_run_outputs(res: RunResult, struct: dict[str, Any], out_dir: str) -> None:
    from .report import write_report

    os.makedirs(out_dir, exist_ok=True)
    inv = res.inv
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(res.cfg, fh, sort_keys=False, allow_unicode=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"metrics": res.metrics, "diagnostics": res.diagnostics}, fh, indent=1, default=_json_default)
    classes = []
    rel_by_class: dict[int, list[str]] = {}
    for r in res.relations:
        lab = f"{res.class_names.get(r.left_class, r.left_class)} + {res.class_names.get(r.right_class, r.right_class)} -> {res.class_names.get(r.result_class, r.result_class)}"
        for c in (r.left_class, r.right_class, r.result_class):
            rel_by_class.setdefault(c, []).append(lab)
    for cid, rep in sorted(res.const.reports.items(), key=lambda kv: (kv[1].status != "accepted", -kv[1].n_occurrences)):
        d = rep.to_json(inv)
        d["name"] = res.class_names.get(cid, f"C{cid}")
        d["composition_participation"] = sorted(set(rel_by_class.get(cid, [])))[:20]
        classes.append(d)
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as fh:
        json.dump(classes, fh, indent=1, default=_json_default)
    with open(os.path.join(out_dir, "relations.json"), "w", encoding="utf-8") as fh:
        json.dump({"accepted": [r.to_json(res.class_names) for r in res.relations], "removed_by_mdl": res.const.removed_relations}, fh, indent=1, default=_json_default)
    with open(os.path.join(out_dir, "ambiguities.json"), "w", encoding="utf-8") as fh:
        json.dump([s.to_json() for s in res.splits], fh, indent=1, default=_json_default)
    with open(os.path.join(out_dir, "structural_ambiguity.json"), "w", encoding="utf-8") as fh:
        json.dump(struct, fh, indent=1, default=_json_default)
    with open(os.path.join(out_dir, "merge_history.jsonl"), "w", encoding="utf-8") as fh:
        for ev in res.state.history:
            fh.write(json.dumps(ev.to_json(), default=_json_default) + "\n")
    with open(os.path.join(out_dir, "corpus.txt"), "w", encoding="utf-8") as fh:
        for s in res.corpus.sentences:
            fh.write(" ".join(s.tokens) + "\n")
    if res.dataset is not None:
        with open(os.path.join(out_dir, "gold.json"), "w", encoding="utf-8") as fh:
            json.dump(res.dataset.gold_json(), fh, default=_json_default)
    if res.v02 is not None:
        from .probabilistic import write_moves, write_trace

        v02 = res.v02
        write_trace(v02.trace, os.path.join(out_dir, "objective_trace.csv"))
        write_moves(v02.moves, os.path.join(out_dir, "structure_moves.jsonl"))
        with open(os.path.join(out_dir, "posterior_stats.json"), "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in v02.posterior_stats.items() if k != "classes"}, fh, indent=1, default=_json_default)
        with open(os.path.join(out_dir, "classes_v02.json"), "w", encoding="utf-8") as fh:
            json.dump(v02.posterior_stats.get("classes", []), fh, indent=1, default=_json_default)
        with open(os.path.join(out_dir, "relations_v02.json"), "w", encoding="utf-8") as fh:
            json.dump([r.to_json(v02.class_names) for r in v02.decoding.relations], fh, indent=1, default=_json_default)
        with open(os.path.join(out_dir, "ambiguity_v02.json"), "w", encoding="utf-8") as fh:
            json.dump({"occurrence_ambiguity": list(v02.decoding.ambiguous_types.values()), "highest_entropy_span_types": v02.posterior_stats.get("highest_entropy_span_types", [])}, fh, indent=1, default=_json_default)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({"metrics": res.metrics, "diagnostics": res.diagnostics, "diagnostics_v02": res.diagnostics_v02}, fh, indent=1, default=_json_default)
        if v02.modifier_table is not None:
            from .modifier_evidence import table_to_json

            with open(os.path.join(out_dir, "modifier_evidence.json"), "w", encoding="utf-8") as fh:
                json.dump(table_to_json(v02.modifier_table), fh, indent=1, default=_json_default)
            with open(os.path.join(out_dir, "modifier_candidates.json"), "w", encoding="utf-8") as fh:
                json.dump(list(v02.modifier_table.by_candidate.values()), fh, indent=1, default=_json_default)
            with open(os.path.join(out_dir, "modifier_string_pairs.json"), "w", encoding="utf-8") as fh:
                json.dump(v02.modifier_table.pairs, fh, indent=1, default=_json_default)
        with open(os.path.join(out_dir, "lexical_distinguishability.json"), "w", encoding="utf-8") as fh:
            json.dump(v02.lexical_distinguishability, fh, indent=1, default=_json_default)
    write_report(res, out_dir)
