"""Adaptive abstract constituent-context signatures (proposal features only).

For a centre node ``u = (i, j)`` with posterior class ``C`` we look at the
nearest *constituents* to its left and right: the shortest spans ``[l, i)``
and ``[j, r)`` whose constituency posterior ``P(Q = 1 | s)`` exceeds
``constituent_posterior_threshold``.  Because class assignments are soft, the
abstract context ``(A, _, B)`` of ``u`` is counted with the soft weight

    w_{uAB} = q(Z_L = A) * q(Z_R = B) * q(Z_u = C)

(BOS / EOS when there is no neighbour).  Depth 1 uses the nearest
constituents ``(L1, R1)``; refinement expands along the existing constituent
hierarchy - the next larger high-posterior constituents ``L2 ⊃ L1``,
``R2 ⊃ R1`` - giving the keys ``(L2, R1)``, ``(L1, R2)``, ``(L2, R2)``.

Refinement is triggered only for *flagged* classes (raw-context-similar but
composition-different pairs, high-entropy classes, weak split proposals),
and refined features are kept only if they score well on information gain /
posterior class separation / support.  These signatures are never part of
the ELBO: they generate merge / split candidates and split partitions, which
the objective then accepts or rejects.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cavi import EStepOutput

BOS = -1
EOS = -2

Feature = tuple  # ("abs", depth_key, A, B)


@dataclass
class AbstractContexts:
    """Per-class and per-occurrence abstract context features."""

    class_features: dict[int, dict[Feature, float]] = field(default_factory=dict)
    occ_features: dict[tuple[int, int, int], dict[Feature, float]] = field(default_factory=dict)  # (sid, i, j) -> features
    depth_of_class: dict[int, int] = field(default_factory=dict)
    selected_refinements: dict[int, list[tuple[Feature, float]]] = field(default_factory=dict)
    n_centres: int = 0


def _neighbours(spans: dict[tuple[int, int], dict[int, float]], const: dict[tuple[int, int], float], i: int, j: int, n: int, tau: float, depth: int):
    """Nearest high-posterior constituents ending at ``i`` (left) and starting at ``j`` (right), up to ``depth`` levels."""
    left = sorted((l for (l, e) in const if e == i and const[(l, e)] >= tau), key=lambda l: -l)[:depth]
    right = sorted((r for (s, r) in const if s == j and const[(s, r)] >= tau), key=lambda r: r)[:depth]
    L = [spans[(l, i)] for l in left]
    R = [spans[(j, r)] for r in right]
    if not L:
        L = [{BOS: 1.0}] if i == 0 else [{}]
    if not R:
        R = [{EOS: 1.0}] if j == n else [{}]
    return L, R


def compute_abstract_contexts(
    out: EStepOutput,
    sentence_lengths: dict[int, int],
    cfg: dict[str, Any],
    flagged: set[int] | None = None,
) -> AbstractContexts:
    """Depth-1 signatures for all classes; depth-2 for ``flagged`` classes."""
    tau_q = float(cfg.get("constituent_posterior_threshold", 0.9))
    tau_u = float(cfg.get("centre_posterior_threshold", 0.5))
    max_depth = int(cfg.get("max_refinement_depth", 2))
    flagged = flagged or set()
    res = AbstractContexts()
    cf: dict[int, dict[Feature, float]] = defaultdict(lambda: defaultdict(float))
    for sid, nodes in out.node_posteriors.items():
        n = sentence_lengths[sid]
        spans: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
        for i, j, c, p in nodes:
            spans[(i, j)][c] = p
        const = {k: sum(d.values()) for k, d in spans.items()}
        for (i, j), d in spans.items():
            for c, p in d.items():
                if p < tau_u:
                    continue
                depth = max_depth if c in flagged else 1
                L, R = _neighbours(spans, const, i, j, n, tau_q, depth)
                feats: dict[Feature, float] = {}
                for li, Ld in enumerate(L):
                    for ri, Rd in enumerate(R):
                        if li > 0 and ri > 0 and depth < 2:
                            continue
                        key = f"{li + 1}{ri + 1}"
                        for a, qa in Ld.items():
                            for b, qb in Rd.items():
                                w = qa * qb * p
                                if w < 1e-4:
                                    continue
                                f = ("abs", key, a, b)
                                feats[f] = feats.get(f, 0.0) + w
                if feats:
                    res.occ_features[(sid, i, j)] = feats
                    for f, w in feats.items():
                        cf[c][f] += w
                    res.n_centres += 1
                res.depth_of_class[c] = max(res.depth_of_class.get(c, 1), depth)
    res.class_features = {c: dict(d) for c, d in cf.items()}
    return res


def select_refinements(
    ctx: AbstractContexts,
    pairs: list[tuple[int, int]],
    top_n: int = 20,
    min_support: float = 3.0,
) -> dict[int, list[tuple[Feature, float]]]:
    """Score depth-2 features by how well they separate the classes of each flagged pair.

    Score = information gain of the feature about the pair identity (bits),
    weighted by support; the ``top_n`` features per class are kept.
    """
    out: dict[int, list[tuple[Feature, float]]] = {}
    for a, b in pairs:
        fa = ctx.class_features.get(a, {})
        fb = ctx.class_features.get(b, {})
        na, nb = sum(fa.values()), sum(fb.values())
        if na <= 0 or nb <= 0:
            continue
        pa = na / (na + nb)
        h0 = -(pa * math.log2(pa) + (1 - pa) * math.log2(1 - pa)) if 0 < pa < 1 else 0.0
        scored = []
        for f in set(fa) | set(fb):
            if f[1] == "11":
                continue  # depth-1 features are always present
            wa, wb = fa.get(f, 0.0), fb.get(f, 0.0)
            s = wa + wb
            if s < min_support:
                continue
            # conditional entropy of the pair identity given feature presence
            p1 = wa / s
            h1 = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1)) if 0 < p1 < 1 else 0.0
            rest_a, rest_b = na - wa, nb - wb
            r = rest_a + rest_b
            if r > 0:
                p2 = rest_a / r
                h2 = -(p2 * math.log2(p2) + (1 - p2) * math.log2(1 - p2)) if 0 < p2 < 1 else 0.0
            else:
                h2 = 0.0
            gain = h0 - (s / (s + r)) * h1 - (r / (s + r)) * h2
            scored.append((f, gain * min(1.0, s / (min_support * 3))))
        scored.sort(key=lambda t: -t[1])
        for c in (a, b):
            cur = out.setdefault(c, [])
            cur.extend(scored[:top_n])
    for c in out:
        seen = {}
        for f, s in out[c]:
            if f not in seen or s > seen[f]:
                seen[f] = s
        out[c] = sorted(seen.items(), key=lambda t: -t[1])[:top_n]
    ctx.selected_refinements = out
    return out


def render_feature(f: Feature, names: dict[int, str]) -> str:
    _, key, a, b = f
    na = "<BOS>" if a == BOS else ("?" if a == -3 else names.get(a, f"C{a}"))
    nb = "<EOS>" if b == EOS else names.get(b, f"C{b}")
    depth = {"11": "(L1,_,R1)", "21": "(L2,_,R1)", "12": "(L1,_,R2)", "22": "(L2,_,R2)"}.get(key, key)
    return f"{depth}: {na} _ {nb}"
