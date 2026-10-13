"""Debug helper: run a config and print compact diagnostics."""
import sys, logging, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
logging.basicConfig(level=logging.INFO, format="%(message)s")
from abc_induction.experiment import load_config, run_experiment
from collections import defaultdict
cfg = load_config(sys.argv[1])
cfg["output_dir"] = sys.argv[2] if len(sys.argv) > 2 else "outputs/_debug"
res = run_experiment(cfg)
inv, gold, state = res.inv, res.gold, res.state
m = res.metrics
print("TYPE-LEVEL", {k: m["class_recovery_type_level"][k] for k in ("ari","precision","recall","f1","purity","n_pred_classes","n_pruned_types")})
print("OCC-LEVEL", {k: m["class_recovery_occurrence_level"][k] for k in ("ari","precision","recall","f1","purity")}, "occ acc", m["occurrence_category_accuracy"])
print("CONST deriv all", m["constituents_derivation_level"]["all"], "\n      len>=2", m["constituents_derivation_level"]["len_ge_2"])
print("CONST class all", m["constituents_class_level"]["all"], "\n      len>=2", m["constituents_class_level"]["len_ge_2"])
print("COMP", {k: m["composition_recovery"][k] for k in ("precision","recall","f1","recovered","missed")})
print("false rels", [(x["inferred"], x["mapped"], x["support"]) for x in m["composition_recovery"]["false_relations"][:8]])
print("AMB", {k: m["ambiguity_detection"][k] for k in ("precision","recall","f1","false_splits","missed_splits")}, m["ambiguity_detection"].get("occurrence_assignment_accuracy"), m["ambiguity_detection"].get("per_word"))
print("alignment", m["class_alignment"])
by_gold = defaultdict(list)
for t in range(inv.n_types):
    if inv.type_length(t) == 1 and t in gold.type_gold_class:
        c0 = int(state.type_class[t]); rep = res.const.reports[c0]
        by_gold[gold.type_gold_class[t]].append((inv.type_str(t), c0, rep.n_members, res.type_classes.get(t)))
for g, items in by_gold.items():
    print(g, items[:10])
print("accepted classes:")
for c in sorted(res.const.accepted, key=lambda c: -res.const.reports[c].n_occurrences)[:16]:
    r = res.const.reports[c]
    print(f"  {res.class_names[c]} mem={r.n_members} occ={r.n_occurrences} usage={r.usage} score={r.score:.2f} pruned={len(r.pruned_members)}", [inv.type_str(t) for t in r.members[:5]], "| pruned", [inv.type_str(t) for t in r.pruned_members[:3]])
print("relations:")
for r in res.relations[:15]: print("  ", r.to_json(res.class_names)["label"], r.support, r.distinct_lexical_support)
print("removed relations", len(res.const.removed_relations), [(d["label"], d["usage"], d["delta_bits"]) for d in sorted(res.const.removed_relations, key=lambda d: -d["usage"])[:6]])
print("diag false merges", res.diagnostics["n_false_merges"], [(f["gold_a"], f["gold_b"], f["size_a"], f["size_b"], f["score"]) for f in res.diagnostics["false_merges"][:5]])
print("missed const", [(d["span"], d["freq"], d["gold_class"], (d["why"] or "")[:60]) for d in res.diagnostics["missed_constituents"][:5]])
print("false const", [(d["span"], d["freq"], d["class_gold"]) for d in res.diagnostics["false_constituents"][:8]])
print("mdl", m.get("mdl"))
print("eff", m["efficiency"]["runtime_sec"], m["efficiency"]["peak_memory_mb"])
