"""Dirichlet variational inference (CAVI / variational EM) for :class:`GrammarModel`.

One iteration:

    E-step   run Inside-Outside with the expected log parameters
             E_q[log theta_{C,e}] = psi(alpha'_{C,e}) - psi(sum_e' alpha'_{C,e'})
             and collect expected counts E[N_{C,e}], E[N_root,C], E[#chunks].
    M-step   alpha'_{C,e} = alpha_{C,e} + E[N_{C,e}]   (stored as counts in the model)

ELBO (Kurihara & Sato 2006, PCFG VB) evaluated right after an E-step:

    ELBO = sum_s log Z~_s  -  sum_d KL( Dir(alpha'_d) || Dir(alpha_d) )

where ``Z~_s`` is the inside sum under the expected log parameters (the
optimal q(Z) given q(Theta) is the chart posterior, so
``E_q[log p(D,Z|Theta)] - E_q[log q(Z)] = sum_s log Z~_s``).  The sequence
of ELBO values across iterations is non-decreasing (exact chart; with beam
pruning the bound is approximate and the monotonicity check is advisory).

The structure objective is ``F(G) = ELBO(G) + log P(G)``.

Backends: ``python`` (reference, :mod:`abc_induction.chart`) and ``numba``
(:mod:`abc_induction.chart_numba`); both return identical expectations.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .chart import NEG_INF, CompiledGrammar, LogSemiring, expectations, inside_chart, viterbi
from .grammar_model import UNK, ExpectedCounts, GrammarModel

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- potentials
class Potentials:
    """Fixed per-sentence virtual evidence.

    ``leaf[sid]`` : list (per position) of ``{model class id: log lambda}``
    (missing classes get ``floor``), or None.
    ``span[sid]`` : ``(n+1, n+1)`` array of ``log psi(i, j)`` or None.
    """

    def __init__(self, leaf: dict[int, list[dict[int, float]]] | None = None, span: dict[int, np.ndarray] | None = None, floor: float = -6.0) -> None:
        self.leaf = leaf or {}
        self.span = span or {}
        self.floor = float(floor)

    def has_leaf(self) -> bool:
        return bool(self.leaf)

    def has_span(self) -> bool:
        return bool(self.span)

    def leaf_array(self, sid: int, n: int, class_ids: list[int], class_map: dict[int, list[int]] | None = None) -> np.ndarray | None:
        """``(n, K)`` log potentials aligned with the compiled class order.

        ``class_map`` maps a *model* class id to the potential class ids it
        inherits from (after merges: both parents, max; after splits: the
        parent).  Model classes without an entry use their own id.
        """
        entries = self.leaf.get(sid)
        if entries is None:
            return None
        K = len(class_ids)
        out = np.full((n, K), self.floor)
        sources = [(class_map.get(c, [c]) if class_map else [c]) for c in class_ids]
        for i, d in enumerate(entries[:n]):
            if not d:
                continue
            for k in range(K):
                best = self.floor
                for src in sources[k]:
                    v = d.get(src)
                    if v is not None and v > best:
                        best = v
                if best > self.floor:
                    out[i, k] = best
        return out

    def span_array(self, sid: int) -> np.ndarray | None:
        return self.span.get(sid)



# --------------------------------------------------------------------------- E-step outputs
@dataclass
class EStepOutput:
    """Side outputs of an E-step (optional, for structure proposals / reports)."""

    log_z: dict[int, float] = field(default_factory=dict)  # per sentence
    node_posteriors: dict[int, list[tuple[int, int, int, float]]] = field(default_factory=dict)  # sid -> [(i, j, class id, p)]
    rule_apps: dict[int, list[tuple[tuple[int, int, int], int, int, int, float]]] = field(default_factory=dict)  # sid -> [(rule key, i, k, j, p)]
    dropped_mass: list[float] = field(default_factory=list)
    pruned_cells: int = 0
    n_cells: int = 0


# --------------------------------------------------------------------------- engine
class VariationalInference:
    """CAVI over a corpus for one :class:`GrammarModel`."""

    def __init__(
        self,
        model: GrammarModel,
        sentences: list[np.ndarray],
        cfg: dict[str, Any] | None = None,
        potentials: Potentials | None = None,
        span_types: list[np.ndarray] | None = None,
        backend: str = "python",
        chart_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.sentences = sentences
        self.cfg = cfg or {}
        self.potentials = potentials or Potentials()
        self.span_types = span_types  # per sentence (n+1, n+1) int32 span type ids (-1 none) for the private-rule prior
        self.backend = backend
        chart_cfg = chart_cfg or {}
        self.exact = bool(chart_cfg.get("exact", True))
        self.max_classes_per_cell = 0 if self.exact else int(chart_cfg.get("max_classes_per_cell", 0))
        self.prune_threshold = 0.0 if self.exact else float(chart_cfg.get("posterior_prune_threshold", 0.0))
        self.posterior_threshold = float(chart_cfg.get("posterior_output_threshold", 1e-3))
        self.app_threshold = float(chart_cfg.get("rule_app_output_threshold", 1e-2))
        self.class_map: dict[int, list[int]] = {}  # model class id -> potential class ids it inherits
        self.trace: list[dict[str, Any]] = []
        self.last_ec: ExpectedCounts | None = None
        self.last_log_z: dict[int, float] = {}
        self._numba = None
        if backend == "numba":
            from . import chart_numba

            self._numba = chart_numba

    # ------------------------------------------------------------------ E-step
    def e_step(self, sids: Iterable[int] | None = None, collect: bool = False) -> tuple[ExpectedCounts, EStepOutput]:
        g = self.model.compile()
        sids = list(range(len(self.sentences))) if sids is None else list(sids)
        if self._numba is not None:
            return self._numba.e_step(self, g, sids, collect)
        return self._e_step_python(g, sids, collect)

    def _e_step_python(self, g: CompiledGrammar, sids: list[int], collect: bool) -> tuple[ExpectedCounts, EStepOutput]:
        ec = ExpectedCounts()
        out = EStepOutput()
        class_ids = g.class_ids
        rule_keys = g.rule_keys  # type: ignore[attr-defined]
        chunk_logp = g.chunk_logp  # type: ignore[attr-defined]
        for sid in sids:
            toks = self.sentences[sid].tolist()
            n = len(toks)
            lp = self.potentials.leaf_array(sid, n, class_ids, self.class_map)
            sp = self.potentials.span_array(sid)
            res = inside_chart(toks, g, LogSemiring, lp, sp, self.max_classes_per_cell, self.prune_threshold, chunk_logp=chunk_logp)
            out.pruned_cells += res.pruned_cells
            out.dropped_mass.extend(res.dropped_mass)
            out.n_cells += n * (n + 1) // 2
            if res.log_z == NEG_INF:
                ec.n_underivable += 1
                out.log_z[sid] = NEG_INF
                continue
            ex = expectations(toks, g, res, sp, self.app_threshold)
            ec.n_sentences += 1
            ec.n_chunks += ex.n_chunks
            ec.log_z += ex.log_z
            out.log_z[sid] = ex.log_z
            for r, p in ex.rule_counts.items():
                key = rule_keys[r]
                ec.rule[key] = ec.rule.get(key, 0.0) + p
            for (c, w), p in ex.lex_counts.items():
                key = (class_ids[c], w)
                ec.lex[key] = ec.lex.get(key, 0.0) + p
            for c, p in ex.root_counts.items():
                key = class_ids[c]
                ec.root[key] = ec.root.get(key, 0.0) + p
            st = self.span_types[sid] if self.span_types is not None else None
            if st is not None:
                for r, i, k, j, p in ex.rule_apps:
                    key = rule_keys[r]
                    d = ec.rule_realisations.setdefault(key, {})
                    tl, tr = int(st[i, k]), int(st[k, j])
                    if tl >= 0:
                        d[tl] = d.get(tl, 0.0) + p
                    if tr >= 0 and tr != tl:
                        d[tr] = d.get(tr, 0.0) + p
            if collect:
                out.node_posteriors[sid] = [(i, j, class_ids[c], p) for (i, j), d in ex.node_posteriors.items() for c, p in d.items() if p >= self.posterior_threshold]
                out.rule_apps[sid] = [(rule_keys[r], i, k, j, p) for r, i, k, j, p in ex.rule_apps]
        ec.pruning = self._pruning_stats(out)
        return ec, out

    @staticmethod
    def _pruning_stats(out: EStepOutput) -> dict[str, float]:
        if not out.dropped_mass:
            return {"cells_pruned": out.pruned_cells, "mean_retained_mass": 1.0, "max_dropped_mass": 0.0, "n_cells": out.n_cells}
        d = np.asarray(out.dropped_mass)
        return {"cells_pruned": out.pruned_cells, "mean_retained_mass": float(1.0 - d.mean()), "max_dropped_mass": float(d.max()), "n_cells": out.n_cells}

    # ------------------------------------------------------------------ M-step / objective
    def m_step(self, ec: ExpectedCounts) -> None:
        self.model.set_counts(ec)

    def elbo_components(self, ec: ExpectedCounts) -> dict[str, float]:
        """ELBO pieces for the model state whose parameters produced ``ec`` (call before the M-step)."""
        kl = self.model.kl_to_prior()
        lp, comp = self.model.structure_log_prior()
        elbo = ec.log_z - kl
        return {
            "log_z": ec.log_z,
            "kl": kl,
            "elbo": elbo,
            "structure_prior": lp,
            "objective": elbo + lp,
            **{f"prior_{k}": v for k, v in comp.items()},
        }

    def iterate(self, sids: Iterable[int] | None = None) -> dict[str, float]:
        """One E-step (scored) followed by one M-step; returns the ELBO components."""
        ec, out = self.e_step(sids)
        comps = self.elbo_components(ec)
        self.last_ec = ec
        self.last_log_z.update(out.log_z)
        self.m_step(ec)
        comps["n_underivable"] = ec.n_underivable
        comps.update({f"pruning_{k}": v for k, v in ec.pruning.items()})
        return comps

    def run(self, max_iterations: int = 50, tolerance: float = 1e-4, min_iterations: int = 2, label: str = "cavi", sids: Iterable[int] | None = None, elbo_tolerance: float = 1e-6) -> list[dict[str, Any]]:
        """CAVI until ``|delta ELBO| / |ELBO| < tolerance``.  Records a trace row per iteration."""
        rows: list[dict[str, Any]] = []
        prev = None
        sids = None if sids is None else list(sids)
        for it in range(max_iterations):
            t = time.time()
            comps = self.iterate(sids)
            elbo = comps["elbo"]
            delta = None if prev is None else elbo - prev
            row = {
                "phase": label,
                "iteration": it,
                "ELBO": elbo,
                "delta_ELBO": delta,
                "log_z": comps["log_z"],
                "kl": comps["kl"],
                "structure_prior": comps["structure_prior"],
                "objective": comps["objective"],
                "expected_rule_count": sum(self.last_ec.rule.values()) if self.last_ec else 0.0,
                "active_classes": self.model.n_classes,
                "active_rules": len(self.model.active_rules()),
                "active_lex": self.model.n_lex,
                "posterior_entropy": self.mean_type_entropy(),
                "n_underivable": comps["n_underivable"],
                "seconds": round(time.time() - t, 3),
                "mean_retained_mass": comps.get("pruning_mean_retained_mass", 1.0),
            }
            rows.append(row)
            self.trace.append(row)
            if delta is not None and delta < -abs(elbo_tolerance) * max(1.0, abs(elbo)) and self.exact:
                log.warning("%s: ELBO decreased by %.6g at iteration %d (exact chart)", label, -delta, it)
                row["elbo_decreased"] = True
            if it + 1 >= min_iterations and prev is not None and abs(delta) < tolerance * max(1.0, abs(elbo)):
                break
            prev = elbo
        return rows

    # ------------------------------------------------------------------ diagnostics
    def mean_type_entropy(self) -> float:
        ent = self.model.posterior_entropy_by_type()
        return float(np.mean(list(ent.values()))) if ent else 0.0

    def score(self, sids: Iterable[int] | None = None) -> dict[str, float]:
        """Objective of the current model state (one E-step, no M-step)."""
        ec, _ = self.e_step(sids)
        return self.elbo_components(ec)

    def viterbi_parse(self, sid: int, g: CompiledGrammar | None = None):
        g = g or self.model.compile()
        toks = self.sentences[sid].tolist()
        lp = self.potentials.leaf_array(sid, len(toks), g.class_ids, self.class_map)
        sp = self.potentials.span_array(sid)
        return viterbi(toks, g, lp, sp, g.chunk_logp)  # type: ignore[attr-defined]
