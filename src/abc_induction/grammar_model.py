"""Finite Bayesian latent grammar ``GrammarModel`` (v0.2).

    GrammarModel = active classes E
                 + sparse rule inventory  (C, A, B)   with Dirichlet posteriors
                 + sparse lexical entries (C, w)      with Dirichlet posteriors
                 + root distribution                  with Dirichlet posterior
                 + chunk-sequence fallback            with Beta posterior (natural text only)
                 + structure prior log P(G)           (centralised, see StructurePrior)

Parametrisation
---------------
For every active class ``C`` the expansions ``e in Exp(C)`` are the rules
``C -> A B`` producing ``C``, the lexical entries ``C -> w`` and (optionally)
the unknown-token expansion ``C -> <unk>``:

    theta_C ~ Dir(alpha_C),   q(theta_C) = Dir(alpha'_C),   alpha'_{C,e} = alpha_{C,e} + E[N_{C,e}]

    E_q[log theta_{C,e}] = psi(alpha'_{C,e}) - psi(sum_e' alpha'_{C,e'})

The composition relation of v0.1, ``P(C | A, B)``, is *derived* from the
posterior expected counts for reporting (``rule_conditional``), and the
type-level class mixture ``pi_x(C) = P(C | x)`` is derived from the expected
lexical counts (``type_mixture``).  Those two bottom-up conditionals cannot
serve as the generative parameters themselves: with the string observed, a
bottom-up labelling model has no string likelihood (every complete grammar
gives sum_{T,Z} = 1, and an incomplete one is rewarded for making more
labellings legal), so nothing would drive structure learning.  The top-down
expansion distributions give the compression signal that v0.1's MDL data cost
already used (``N(C, e) * -log P(e | C)``), now with Dirichlet uncertainty.

Class ids are never reused (merges and splits create fresh ids) so that
structure moves can be logged and rolled back unambiguously.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.special import digamma, gammaln

from .chart import NEG_INF, CompiledGrammar

UNK = -1  # token id of the unknown-token expansion


# --------------------------------------------------------------------------- structure prior
@dataclass
class StructurePrior:
    """``log P(G) = -lambda_C |E| - lambda_R |R| - lambda_L |Lex| - sum_r Penalty_private(r)``.

    All structure costs live here (nats).  The private-rule penalty uses the
    posterior expected realisation counts of every rule:

        Penalty_private(r) = lambda_p * max(0, max_x N_{r,x} / N_r - tau_p)

    for rules with ``N_r >= private_min_count`` (``N_{r,x}`` = expected count of
    rule ``r`` realised with span type ``x`` on either side).
    """

    lambda_class: float = 8.0
    lambda_rule: float = 7.0
    lambda_lex: float = 2.0
    lambda_private: float = 20.0
    private_tau: float = 0.6
    private_min_count: float = 5.0

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> "StructurePrior":
        cfg = cfg or {}
        return cls(**{k: float(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})

    def private_penalty(self, n_rule: float, dominant: float) -> float:
        if n_rule < self.private_min_count or n_rule <= 0:
            return 0.0
        return self.lambda_private * max(0.0, dominant / n_rule - self.private_tau)


# --------------------------------------------------------------------------- expected counts
@dataclass
class ExpectedCounts:
    """Posterior expected counts of one E-step."""

    rule: dict[tuple[int, int, int], float] = field(default_factory=dict)
    lex: dict[tuple[int, int], float] = field(default_factory=dict)
    root: dict[int, float] = field(default_factory=dict)
    n_sentences: float = 0.0
    n_chunks: float = 0.0  # expected number of chunks (>= n_sentences)
    log_z: float = 0.0  # sum_s log Z_s
    n_underivable: int = 0
    rule_realisations: dict[tuple[int, int, int], dict[int, float]] = field(default_factory=dict)  # rule -> {span type: E[N]}
    pruning: dict[str, float] = field(default_factory=dict)

    def rule_total(self, key: tuple[int, int, int]) -> float:
        return self.rule.get(key, 0.0)

    def scaled(self, factor: float) -> "ExpectedCounts":
        return ExpectedCounts(
            {k: v * factor for k, v in self.rule.items()},
            {k: v * factor for k, v in self.lex.items()},
            {k: v * factor for k, v in self.root.items()},
            self.n_sentences * factor,
            self.n_chunks * factor,
            self.log_z * factor,
            self.n_underivable,
            {k: {t: v * factor for t, v in d.items()} for k, d in self.rule_realisations.items()},
            dict(self.pruning),
        )


# --------------------------------------------------------------------------- the model
class GrammarModel:
    """Active classes, sparse rules / lexicon and their Dirichlet posteriors."""

    def __init__(
        self,
        alpha_rule: float = 0.5,
        alpha_lex: float = 0.5,
        alpha_unk: float = 0.1,
        alpha_root: float = 0.5,
        unk_enabled: bool = True,
        chunks_enabled: bool = False,
        beta_chunk: tuple[float, float] = (1.0, 1.0),
        prior: StructurePrior | None = None,
    ) -> None:
        self.alpha_rule = float(alpha_rule)
        self.alpha_lex = float(alpha_lex)
        self.alpha_unk = float(alpha_unk)
        self.alpha_root = float(alpha_root)
        self.unk_enabled = bool(unk_enabled)
        self.chunks_enabled = bool(chunks_enabled)
        self.beta_chunk = (float(beta_chunk[0]), float(beta_chunk[1]))
        self.prior = prior or StructurePrior()
        self.classes: set[int] = set()
        self.rules: dict[tuple[int, int, int], float] = {}  # (C, A, B) -> expected count
        self.lex: dict[tuple[int, int], float] = {}  # (C, w) -> expected count (w = UNK allowed)
        self.root: dict[int, float] = {}  # C -> expected count
        self.chunk_counts = (0.0, 0.0)  # (expected continuations, expected sentences)
        self.next_class_id = 0
        self.names: dict[int, str] = {}
        self.parent: dict[int, tuple[str, tuple[int, ...]]] = {}  # lineage: new id -> (op, old ids)
        self.realisations: dict[tuple[int, int, int], dict[int, float]] = {}
        self.version = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], prior: StructurePrior | None = None) -> "GrammarModel":
        return cls(
            alpha_rule=float(cfg.get("alpha_rule", 0.5)),
            alpha_lex=float(cfg.get("alpha_lex", cfg.get("gamma_type", 0.5))),
            alpha_unk=float(cfg.get("alpha_unk", 0.1)),
            alpha_root=float(cfg.get("alpha_root", 0.5)),
            unk_enabled=bool(cfg.get("unk_enabled", True)),
            chunks_enabled=bool(cfg.get("chunks_enabled", False)),
            beta_chunk=tuple(cfg.get("beta_chunk", (1.0, 1.0))),
            prior=prior,
        )

    def new_class(self, name: str | None = None) -> int:
        c = self.next_class_id
        self.next_class_id += 1
        self.classes.add(c)
        self.names[c] = name or f"C{c}"
        self.version += 1
        return c

    def add_class(self, cid: int, name: str | None = None) -> None:
        self.classes.add(cid)
        self.names[cid] = name or f"C{cid}"
        self.next_class_id = max(self.next_class_id, cid + 1)
        self.version += 1

    def add_rule(self, c: int, a: int, b: int, count: float = 0.0) -> None:
        key = (c, a, b)
        self.rules[key] = self.rules.get(key, 0.0) + count
        self.version += 1

    def add_lex(self, c: int, w: int, count: float = 0.0) -> None:
        key = (c, w)
        self.lex[key] = self.lex.get(key, 0.0) + count
        self.version += 1

    def remove_rule(self, key: tuple[int, int, int]) -> None:
        self.rules.pop(key, None)
        self.realisations.pop(key, None)
        self.version += 1

    def remove_lex(self, key: tuple[int, int]) -> None:
        self.lex.pop(key, None)
        self.version += 1

    def remove_class(self, c: int) -> None:
        self.classes.discard(c)
        for k in [k for k in self.rules if c in k]:
            self.remove_rule(k)
        for k in [k for k in self.lex if k[0] == c]:
            self.lex.pop(k)
        self.root.pop(c, None)
        self.version += 1

    # ------------------------------------------------------------------ views
    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def n_lex(self) -> int:
        return sum(1 for (c, w) in self.lex if w != UNK)

    def active_rules(self) -> list[tuple[int, int, int]]:
        return [k for k in self.rules if k[0] in self.classes and k[1] in self.classes and k[2] in self.classes]

    def rules_of(self, c: int) -> list[tuple[int, int, int]]:
        return [k for k in self.rules if k[0] == c]

    def lex_of(self, c: int) -> list[tuple[int, int]]:
        return [k for k in self.lex if k[0] == c]

    def expansions(self, c: int) -> list[tuple[str, Any, float, float]]:
        """``(kind, key, alpha, count)`` for every expansion of class ``c``."""
        out: list[tuple[str, Any, float, float]] = []
        for key, n in self.rules.items():
            if key[0] == c and key[1] in self.classes and key[2] in self.classes:
                out.append(("r", key, self.alpha_rule, n))
        for key, n in self.lex.items():
            if key[0] == c:
                out.append(("l", key, self.alpha_unk if key[1] == UNK else self.alpha_lex, n))
        return out

    def checkpoint(self) -> dict[str, Any]:
        """Lightweight snapshot (dict copies) for rollback."""
        return {
            "classes": set(self.classes),
            "rules": dict(self.rules),
            "lex": dict(self.lex),
            "root": dict(self.root),
            "chunk_counts": self.chunk_counts,
            "next_class_id": self.next_class_id,
            "names": dict(self.names),
            "parent": dict(self.parent),
            "realisations": {k: dict(v) for k, v in self.realisations.items()},
            "version": self.version,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.classes = set(snap["classes"])
        self.rules = dict(snap["rules"])
        self.lex = dict(snap["lex"])
        self.root = dict(snap["root"])
        self.chunk_counts = snap["chunk_counts"]
        self.next_class_id = snap["next_class_id"]
        self.names = dict(snap["names"])
        self.parent = dict(snap["parent"])
        self.realisations = {k: dict(v) for k, v in snap["realisations"].items()}
        self.version = snap["version"] + 1

    def copy(self) -> "GrammarModel":
        m = GrammarModel(self.alpha_rule, self.alpha_lex, self.alpha_unk, self.alpha_root, self.unk_enabled, self.chunks_enabled, self.beta_chunk, self.prior)
        m.restore(self.checkpoint())
        return m

    # ------------------------------------------------------------------ variational parameters
    def _class_tables(self) -> dict[int, tuple[list[tuple[str, Any]], np.ndarray]]:
        """Per class: expansion keys and posterior alpha' (prior + expected count)."""
        out = {}
        for c in self.classes:
            exps = self.expansions(c)
            if self.unk_enabled and not any(k == "l" and key[1] == UNK for k, key, _, _ in exps):
                exps.append(("l", (c, UNK), self.alpha_unk, 0.0))
            keys = [(k, key) for k, key, _, _ in exps]
            alpha = np.array([a + n for _, _, a, n in exps], dtype=np.float64)
            out[c] = (keys, alpha)
        return out

    def expected_log_params(self) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int], float], dict[int, float], tuple[float, float]]:
        """``E_q[log theta]`` for rules, lexical entries, roots and the chunk Bernoulli."""
        rule_lp: dict[tuple[int, int, int], float] = {}
        lex_lp: dict[tuple[int, int], float] = {}
        for c, (keys, alpha) in self._class_tables().items():
            if len(alpha) == 0:
                continue
            lp = digamma(alpha) - digamma(alpha.sum())
            for (kind, key), v in zip(keys, lp.tolist()):
                if kind == "r":
                    rule_lp[key] = v
                else:
                    lex_lp[key] = v
        classes = sorted(self.classes)
        ar = np.array([self.alpha_root + self.root.get(c, 0.0) for c in classes], dtype=np.float64)
        root_lp = {}
        if len(ar):
            lr = digamma(ar) - digamma(ar.sum())
            root_lp = dict(zip(classes, lr.tolist()))
        a, b = self.beta_chunk
        cont, stop = self.chunk_counts
        tot = digamma(a + cont + b + stop)
        chunk_lp = (float(digamma(a + cont) - tot), float(digamma(b + stop) - tot))
        return rule_lp, lex_lp, root_lp, chunk_lp

    def compile(self) -> CompiledGrammar:
        """Compact tables with ``E_q[log theta]`` for the chart backends."""
        rule_lp, lex_lp, root_lp, chunk_lp = self.expected_log_params()
        classes = sorted(self.classes)
        idx = {c: i for i, c in enumerate(classes)}
        K = len(classes)
        rules = [(idx[c], idx[a], idx[b], lp) for (c, a, b), lp in rule_lp.items()]
        rules.sort(key=lambda t: (t[1], t[2], t[0]))
        lex: dict[int, list[tuple[int, float]]] = {}
        unk = np.full(K, NEG_INF)
        lex_entry_index: dict[tuple[int, int], int] = {}
        for (c, w), lp in lex_lp.items():
            if w == UNK:
                unk[idx[c]] = lp
            else:
                lex.setdefault(w, []).append((idx[c], lp))
        for w in lex:
            lex[w].sort()
        root = np.full(K, NEG_INF)
        for c, lp in root_lp.items():
            root[idx[c]] = lp
        g = CompiledGrammar(K, rules, lex, root, unk, class_ids=classes)
        g.chunk_logp = chunk_lp if self.chunks_enabled else None  # type: ignore[attr-defined]
        g.rule_keys = [(classes[c], classes[a], classes[b]) for c, a, b, _ in rules]  # type: ignore[attr-defined]
        return g

    # ------------------------------------------------------------------ M-step
    def set_counts(self, ec: ExpectedCounts) -> None:
        """``alpha' = alpha + E[N]``: store the expected counts (prior is added lazily)."""
        for key in self.rules:
            self.rules[key] = ec.rule.get(key, 0.0)
        for key in self.lex:
            self.lex[key] = 0.0
        for key, n in ec.lex.items():
            if key[0] in self.classes:
                self.lex[key] = n
        self.root = {c: n for c, n in ec.root.items() if c in self.classes}
        self.chunk_counts = (max(ec.n_chunks - ec.n_sentences, 0.0), ec.n_sentences)
        self.realisations = {k: dict(v) for k, v in ec.rule_realisations.items() if k in self.rules}
        self.version += 1

    # ------------------------------------------------------------------ ELBO pieces
    def kl_to_prior(self) -> float:
        """``sum_d KL(Dir(alpha'_d) || Dir(alpha_d))`` over all expansion / root / chunk distributions."""
        total = 0.0
        for c, (keys, alpha_post) in self._class_tables().items():
            if len(alpha_post) == 0:
                continue
            alpha_prior = np.array([self.alpha_rule if k == "r" else (self.alpha_unk if key[1] == UNK else self.alpha_lex) for k, key in keys])
            total += _dir_kl(alpha_post, alpha_prior)
        classes = sorted(self.classes)
        if classes:
            ap = np.array([self.alpha_root + self.root.get(c, 0.0) for c in classes])
            total += _dir_kl(ap, np.full(len(classes), self.alpha_root))
        if self.chunks_enabled:
            a, b = self.beta_chunk
            cont, stop = self.chunk_counts
            total += _dir_kl(np.array([a + cont, b + stop]), np.array([a, b]))
        return float(total)

    def structure_log_prior(self) -> tuple[float, dict[str, float]]:
        """``log P(G)`` and its components."""
        p = self.prior
        n_rules = len(self.active_rules())
        n_lex = self.n_lex
        private = 0.0
        for key in self.rules:
            n = self.rules[key]
            real = self.realisations.get(key)
            if not real:
                continue
            private += p.private_penalty(n, max(real.values()))
        comp = {
            "class": -p.lambda_class * self.n_classes,
            "rule": -p.lambda_rule * n_rules,
            "lex": -p.lambda_lex * n_lex,
            "private": -private,
        }
        return sum(comp.values()), comp

    # ------------------------------------------------------------------ derived (bottom-up) views
    def rule_conditional(self, key: tuple[int, int, int], eta: float = 0.5) -> float:
        """Posterior-mean ``P(C | A, B) = (E[N_ABC] + eta) / (sum_C' E[N_ABC'] + K eta)``."""
        c, a, b = key
        tot = sum(n for (cc, aa, bb), n in self.rules.items() if aa == a and bb == b)
        k = max(1, sum(1 for (cc, aa, bb) in self.rules if aa == a and bb == b))
        return (self.rules.get(key, 0.0) + eta) / (tot + k * eta)

    def type_mixture(self, w: int, gamma: float = 0.0) -> dict[int, float]:
        """Posterior type-level class mixture ``pi_w(C) proportional to gamma + E[N_{C,w}]``."""
        d = {c: n + gamma for (c, ww), n in self.lex.items() if ww == w}
        tot = sum(d.values())
        return {c: v / tot for c, v in d.items()} if tot > 0 else {}

    def posterior_entropy_by_type(self) -> dict[int, float]:
        """Entropy (nats) of ``pi_w`` for every token with >= 2 lexical entries."""
        by_w: dict[int, list[float]] = {}
        for (c, w), n in self.lex.items():
            if w != UNK:
                by_w.setdefault(w, []).append(n)
        out = {}
        for w, ns in by_w.items():
            tot = sum(ns)
            if len(ns) >= 2 and tot > 0:
                p = np.array(ns) / tot
                p = p[p > 0]
                out[w] = float(-(p * np.log(p)).sum())
        return out

    # ------------------------------------------------------------------ structure operations
    def merge_classes(self, a: int, b: int, name: str | None = None) -> int:
        """``A1 + A2 => A``: fresh class, rules / lexicon / roots summed (warm start)."""
        new = self.new_class(name or f"{self.names.get(a, a)}+{self.names.get(b, b)}")
        ren = {a: new, b: new}
        rules: dict[tuple[int, int, int], float] = {}
        real: dict[tuple[int, int, int], dict[int, float]] = {}
        for (c, l, r), n in self.rules.items():
            key = (ren.get(c, c), ren.get(l, l), ren.get(r, r))
            rules[key] = rules.get(key, 0.0) + n
            if (c, l, r) in self.realisations:
                d = real.setdefault(key, {})
                for t, v in self.realisations[(c, l, r)].items():
                    d[t] = d.get(t, 0.0) + v
        lex: dict[tuple[int, int], float] = {}
        for (c, w), n in self.lex.items():
            key = (ren.get(c, c), w)
            lex[key] = lex.get(key, 0.0) + n
        root: dict[int, float] = {}
        for c, n in self.root.items():
            key = ren.get(c, c)
            root[key] = root.get(key, 0.0) + n
        self.rules, self.lex, self.root, self.realisations = rules, lex, root, real
        self.classes.discard(a)
        self.classes.discard(b)
        self.parent[new] = ("merge", (a, b))
        self.version += 1
        return new

    def split_class(
        self,
        a: int,
        rule_shares: dict[tuple[int, int, int], float],
        lex_shares: dict[tuple[int, int], float],
        root_share: float,
        names: tuple[str, str] | None = None,
    ) -> tuple[int, int]:
        """``A => A1, A2``: every rule / lexical entry / root count involving ``A`` is
        copied to both halves; the expected count of each copy is divided
        according to ``*_shares`` (fraction going to A1, from the occurrence
        partition).  Rules where ``A`` appears several times are expanded to
        all combinations.  Unsupported copies are then removed by the rule
        proposals / posterior counts, not here.
        """
        n1 = self.new_class(names[0] if names else f"{self.names.get(a, a)}.1")
        n2 = self.new_class(names[1] if names else f"{self.names.get(a, a)}.2")
        halves = (n1, n2)
        rules: dict[tuple[int, int, int], float] = {}
        real: dict[tuple[int, int, int], dict[int, float]] = {}
        for key, n in self.rules.items():
            if a not in key:
                rules[key] = n
                if key in self.realisations:
                    real[key] = dict(self.realisations[key])
                continue
            s = rule_shares.get(key, 0.5)
            positions = [i for i, x in enumerate(key) if x == a]
            combos = [()]
            for _ in positions:
                combos = [c + (h,) for c in combos for h in halves]
            for combo in combos:
                k = list(key)
                for pos, h in zip(positions, combo):
                    k[pos] = h
                # share: product of per-position shares (s for A1, 1-s for A2)
                w = 1.0
                for h in combo:
                    w *= s if h == n1 else (1.0 - s)
                kk = (k[0], k[1], k[2])
                rules[kk] = rules.get(kk, 0.0) + n * w
                if key in self.realisations:
                    d = real.setdefault(kk, {})
                    for t, v in self.realisations[key].items():
                        d[t] = d.get(t, 0.0) + v * w
        lex: dict[tuple[int, int], float] = {}
        for (c, w), n in self.lex.items():
            if c != a:
                lex[(c, w)] = n
                continue
            s = lex_shares.get((c, w), 0.5)
            lex[(n1, w)] = n * s
            lex[(n2, w)] = n * (1.0 - s)
        root: dict[int, float] = {c: n for c, n in self.root.items() if c != a}
        if a in self.root:
            root[n1] = self.root[a] * root_share
            root[n2] = self.root[a] * (1.0 - root_share)
        self.rules, self.lex, self.root, self.realisations = rules, lex, root, real
        self.classes.discard(a)
        self.parent[n1] = ("split", (a,))
        self.parent[n2] = ("split", (a,))
        self.version += 1
        return n1, n2

    # ------------------------------------------------------------------ summaries
    def summary(self) -> dict[str, Any]:
        lp, comp = self.structure_log_prior()
        return {
            "n_classes": self.n_classes,
            "n_rules": len(self.active_rules()),
            "n_lexical_entries": self.n_lex,
            "structure_log_prior": round(lp, 3),
            "structure_prior_components": {k: round(v, 3) for k, v in comp.items()},
        }


def _dir_kl(a: np.ndarray, b: np.ndarray) -> float:
    """``KL(Dir(a) || Dir(b))``."""
    sa, sb = a.sum(), b.sum()
    return float(gammaln(sa) - gammaln(a).sum() - gammaln(sb) + gammaln(b).sum() + ((a - b) * (digamma(a) - digamma(sa))).sum())
