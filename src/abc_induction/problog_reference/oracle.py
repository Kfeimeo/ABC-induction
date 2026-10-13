"""ProbLog encoding of a tiny :class:`CompiledGrammar` and exact queries."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..chart import NEG_INF, CompiledGrammar


def problog_available() -> bool:
    try:
        import problog  # noqa: F401

        return True
    except Exception:
        return False


def _p(logp: float) -> float:
    return 0.0 if logp == NEG_INF else min(1.0, math.exp(logp))


def encode(
    tokens: list[int],
    g: CompiledGrammar,
    leaf_potential: np.ndarray | None = None,
    span_potential: np.ndarray | None = None,
    max_depth: int | None = None,
) -> str:
    """ProbLog program for ``P(tokens)`` under ``g`` (one root, no chunk layer).

    The expansion probabilities are ``exp(logp)`` of the compiled tables
    (expected log parameters), so the oracle checks *inference*, not the
    normalisation of the variational parameters (they need not sum to one).
    """
    n = len(tokens)
    K = g.n_classes
    max_depth = n if max_depth is None else max_depth
    L: list[str] = []
    L.append("unktok(-1).")  # every predicate needs a clause
    for i, w in enumerate(tokens):
        L.append(f"tok({i}, w{w}).")
        if not g.lex.get(w):
            L.append(f"unktok({i}).")
    # expansions of every class: AD per (class, path) -> we instantiate paths lazily through a
    # generic clause; ProbLog needs ground ADs, so enumerate all paths up to the depth bound.
    paths = _paths(max_depth)
    exps: dict[int, list[tuple[float, str]]] = {c: [] for c in range(K)}
    for r, (c, a, b, lp) in enumerate(g.rules):
        exps[c].append((_p(lp), f"rule({a}, {b}, {r})"))
    lex_words = set(tokens)
    seen: set[tuple[int, str]] = set()
    for w in lex_words:
        for c, lp in g.leaf_entries(w):
            key = f"lex(w{w})" if g.lex.get(w) else "lex(unk)"
            if (c, key) in seen:
                continue  # one UNK disjunct per class, however many unknown tokens
            seen.add((c, key))
            exps[c].append((_p(lp), key))
    for c in range(K):
        items = [(p, e) for p, e in exps[c] if p > 0]
        if not items:
            continue
        for path in paths:
            heads = "; ".join(f"{p}::expand({c}, {path}, {e})" for p, e in items)
            L.append(heads + ".")
    # root: one annotated disjunction (derivations with different roots are exclusive worlds)
    roots = [(c, _p(float(g.root[c]))) for c in range(K) if _p(float(g.root[c])) > 0]
    if roots:
        L.append("; ".join(f"{rp}::root({c})" for c, rp in roots) + ".")
    # virtual evidence
    if leaf_potential is not None:
        for i in range(n):
            for c in range(K):
                p = _p(float(leaf_potential[i, c]))
                if p > 0:
                    L.append(f"{p}::leafpot({i}, {c}).")
    if span_potential is not None:
        for i in range(n):
            for j in range(i + 1, n + 1):
                p = _p(float(span_potential[i, j]))
                if p > 0:
                    L.append(f"{p}::spanpot({i}, {j}).")
    leaf_guard = "leafpot(I, C)" if leaf_potential is not None else "true"
    span_guard = "spanpot(I, J)" if span_potential is not None else "true"
    L.append(
        f"""
% d(C, Path, I, J): class C at Gorn address Path derives exactly tokens [I, J).
d(C, P, I, J) :- J is I + 1, tok(I, W), expand(C, P, lex(W)), {leaf_guard}, {span_guard}.
d(C, P, I, J) :- J is I + 1, unktok(I), expand(C, P, lex(unk)), {leaf_guard}, {span_guard}.
d(C, P, I, J) :- J > I + 1, expand(C, P, rule(A, B, _)), K1 is I + 1, K2 is J - 1, between(K1, K2, K),
                 d(A, [0|P], I, K), d(B, [1|P], K, J), {span_guard}.
sentence :- root(C), d(C, [], 0, {n}).
% on(C, P, I, J): node (C, P, I, J) lies on the root derivation.
on(C, [], 0, {n}) :- root(C), d(C, [], 0, {n}).
on(A, [0|P], I, K) :- on(C, P, I, J), expand(C, P, rule(A, B, _)), K1 is I + 1, K2 is J - 1, between(K1, K2, K),
                      d(A, [0|P], I, K), d(B, [1|P], K, J).
on(B, [1|P], K, J) :- on(C, P, I, J), expand(C, P, rule(A, B, _)), K1 is I + 1, K2 is J - 1, between(K1, K2, K),
                      d(A, [0|P], I, K), d(B, [1|P], K, J).
node(I, J, C) :- on(C, _, I, J).
const(I, J) :- node(I, J, _).
"""
    )
    return "\n".join(L)


def _paths(max_depth: int) -> list[str]:
    out = []
    frontier = ["[]"]
    inner = [[]]
    out.append("[]")
    cur: list[list[int]] = [[]]
    for _ in range(max_depth - 1):
        nxt = []
        for p in cur:
            for b in (0, 1):
                q = [b] + p
                nxt.append(q)
                out.append("[" + ",".join(str(x) for x in q) + "]")
        cur = nxt
    return out


class ProbLogOracle:
    """Exact marginal / posterior queries through ProbLog."""

    def __init__(self, tokens: list[int], g: CompiledGrammar, leaf_potential: np.ndarray | None = None, span_potential: np.ndarray | None = None) -> None:
        if not problog_available():
            raise RuntimeError("problog is not installed")
        self.tokens = tokens
        self.g = g
        self.program = encode(tokens, g, leaf_potential, span_potential)
        self.n = len(tokens)

    def _run(self, extra: str) -> dict[str, float]:
        from problog import get_evaluatable
        from problog.program import PrologString

        prog = PrologString(self.program + "\n" + extra)
        res = get_evaluatable().create_from(prog).evaluate()
        return {str(k): float(v) for k, v in res.items()}

    def marginal(self) -> float:
        """``P(sentence)`` = inside mass."""
        return self._run("query(sentence).").get("sentence", 0.0)

    def posteriors(self) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int], float]]:
        """``P(Z_ij = C | s)`` and ``P(Q_ij = 1 | s)`` (conditioned on the sentence being derived)."""
        n, K = self.n, self.g.n_classes
        q = ["evidence(sentence, true)."]
        for i in range(n):
            for j in range(i + 1, n + 1):
                q.append(f"query(const({i}, {j})).")
                for c in range(K):
                    q.append(f"query(node({i}, {j}, {c})).")
        res = self._run("\n".join(q))
        nodes: dict[tuple[int, int, int], float] = {}
        const: dict[tuple[int, int], float] = {}
        for k, v in res.items():
            if k.startswith("node("):
                i, j, c = (int(x) for x in k[5:-1].split(","))
                nodes[(i, j, c)] = v
            elif k.startswith("const("):
                i, j = (int(x) for x in k[6:-1].split(","))
                const[(i, j)] = v
        return nodes, const
