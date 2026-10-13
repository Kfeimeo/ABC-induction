"""Accelerated Inside-Outside backend (numba) with the same semantics as :mod:`chart`.

Dense ``(n+1, n+1, K)`` log tables per sentence plus per-cell *active class
lists*, so every split iterates only over the classes that are actually
derivable in the left cell and, for each, over the rules indexed by that left
class (``rule_ptr``).  Expected counts are accumulated in-kernel; sparse node
posteriors and rule applications above a threshold are returned for the
structure proposals.

The kernel is validated against the reference backend in the test-suite
(``tests/test_chart_exact.py``); both must agree to 1e-9.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # pragma: no cover - import guard
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f

        return deco if not (a and callable(a[0])) else a[0]


from .chart import NEG_INF, CompiledGrammar
from .grammar_model import UNK, ExpectedCounts

NEG = -1e300  # finite stand-in for -inf inside the kernel (avoids inf - inf)


@njit(cache=True)
def _lse(a: float, b: float) -> float:
    if a <= NEG:
        return b
    if b <= NEG:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(cache=True)
def _kernel(
    tokens, K, lex_ptr, lex_cls, lex_lp, lex_eid, unk_lp,
    rule_ptr, rule_C, rule_B, rule_lp, root_lp,
    leaf_pot, has_leaf, span_pot, has_span,
    beam, log_thr, mode, use_chunks, lc, ls,
    span_tid, has_tid, post_thr, app_thr,
    rule_counts, lex_counts, unk_counts, root_counts,
    node_buf, app_buf, real_buf,
):
    """Inside (+ outside + expectations in mode 0) of one sentence.

    Returns ``(log_z, n_chunks, n_nodes, n_apps, n_real, n_pruned, dropped_sum, dropped_max, bp_k, bp_r, inside, active, n_active)``.
    """
    n = tokens.shape[0]
    V = lex_ptr.shape[0] - 1
    inside = np.full((n + 1, n + 1, K), NEG)
    active = np.zeros((n + 1, n + 1, K), dtype=np.int32)
    n_active = np.zeros((n + 1, n + 1), dtype=np.int32)
    bp_k = np.full((n + 1, n + 1, K), -1, dtype=np.int32)
    bp_r = np.full((n + 1, n + 1, K), -1, dtype=np.int32)
    is_unk = np.zeros(n, dtype=np.bool_)
    n_pruned = 0
    dropped_sum = 0.0
    dropped_max = 0.0
    # ---- leaves
    for i in range(n):
        w = tokens[i]
        cnt = 0
        if w >= 0 and w < V and lex_ptr[w + 1] > lex_ptr[w]:
            for e in range(lex_ptr[w], lex_ptr[w + 1]):
                c = lex_cls[e]
                v = lex_lp[e]
                if has_leaf:
                    v += leaf_pot[i, c]
                if has_span:
                    v += span_pot[i, i + 1]
                inside[i, i + 1, c] = v
                active[i, i + 1, cnt] = c
                cnt += 1
        else:
            is_unk[i] = True
            for c in range(K):
                if unk_lp[c] > NEG:
                    v = unk_lp[c]
                    if has_leaf:
                        v += leaf_pot[i, c]
                    if has_span:
                        v += span_pot[i, i + 1]
                    inside[i, i + 1, c] = v
                    active[i, i + 1, cnt] = c
                    cnt += 1
        n_active[i, i + 1] = cnt
    # ---- internal
    cell = np.full(K, NEG)
    touched = np.zeros(K, dtype=np.int32)
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            nt = 0
            for k in range(i + 1, j):
                na = n_active[i, k]
                if na == 0 or n_active[k, j] == 0:
                    continue
                for ai in range(na):
                    a = active[i, k, ai]
                    va = inside[i, k, a]
                    for r in range(rule_ptr[a], rule_ptr[a + 1]):
                        b = rule_B[r]
                        vb = inside[k, j, b]
                        if vb <= NEG:
                            continue
                        c = rule_C[r]
                        v = va + vb + rule_lp[r]
                        cur = cell[c]
                        if cur <= NEG:
                            cell[c] = v
                            touched[nt] = c
                            nt += 1
                            bp_k[i, j, c] = k
                            bp_r[i, j, c] = r
                        else:
                            if mode == 1:
                                if v > cur:
                                    cell[c] = v
                                    bp_k[i, j, c] = k
                                    bp_r[i, j, c] = r
                            else:
                                cell[c] = _lse(cur, v)
                                if v > cur:
                                    bp_k[i, j, c] = k
                                    bp_r[i, j, c] = r
            if nt == 0:
                continue
            sp = span_pot[i, j] if has_span else 0.0
            # pruning
            if (beam > 0 and nt > beam) or log_thr > NEG:
                vmax = NEG
                for t in range(nt):
                    if cell[touched[t]] > vmax:
                        vmax = cell[touched[t]]
                # total mass
                tot = 0.0
                for t in range(nt):
                    tot += math.exp(cell[touched[t]] - vmax)
                # select: threshold first
                keep = np.zeros(nt, dtype=np.bool_)
                nk = 0
                for t in range(nt):
                    if cell[touched[t]] - vmax >= log_thr:
                        keep[t] = True
                        nk += 1
                if beam > 0 and nk > beam:
                    # keep the top-`beam` among kept
                    vals = np.empty(nk)
                    idxs = np.empty(nk, dtype=np.int32)
                    q = 0
                    for t in range(nt):
                        if keep[t]:
                            vals[q] = cell[touched[t]]
                            idxs[q] = t
                            q += 1
                    order = np.argsort(-vals)
                    for q in range(beam, nk):
                        keep[idxs[order[q]]] = False
                    nk = beam
                if nk < nt:
                    kept = 0.0
                    for t in range(nt):
                        if keep[t]:
                            kept += math.exp(cell[touched[t]] - vmax)
                    d = 1.0 - kept / tot
                    dropped_sum += d
                    if d > dropped_max:
                        dropped_max = d
                    n_pruned += 1
                    for t in range(nt):
                        if not keep[t]:
                            cell[touched[t]] = NEG
            cnt = 0
            for t in range(nt):
                c = touched[t]
                if cell[c] > NEG:
                    inside[i, j, c] = cell[c] + sp
                    active[i, j, cnt] = c
                    cnt += 1
                cell[c] = NEG
            n_active[i, j] = cnt
    # ---- root / chunk layer
    chunk_mass = np.full((n + 1, n + 1), NEG)
    for i in range(n):
        for j in range(i + 1, n + 1):
            m = NEG
            for t in range(n_active[i, j]):
                c = active[i, j, t]
                if root_lp[c] > NEG:
                    v = inside[i, j, c] + root_lp[c]
                    if mode == 1:
                        if v > m:
                            m = v
                    else:
                        m = _lse(m, v)
            chunk_mass[i, j] = m
    prefix = np.full(n + 1, NEG)
    suffix = np.full(n + 1, NEG)
    if use_chunks:
        prefix[0] = 0.0
        for j in range(1, n + 1):
            acc = NEG
            for i in range(0, j):
                m = chunk_mass[i, j]
                if m <= NEG or prefix[i] <= NEG:
                    continue
                v = prefix[i] + m + (lc if i > 0 else 0.0)
                if mode == 1:
                    if v > acc:
                        acc = v
                else:
                    acc = _lse(acc, v)
            prefix[j] = acc
        suffix[n] = ls
        for i in range(n - 1, -1, -1):
            acc = NEG
            for j in range(i + 1, n + 1):
                m = chunk_mass[i, j]
                if m <= NEG or suffix[j] <= NEG:
                    continue
                v = lc + m + suffix[j]
                if mode == 1:
                    if v > acc:
                        acc = v
                else:
                    acc = _lse(acc, v)
            suffix[i] = acc
        log_z = prefix[n] + ls if prefix[n] > NEG else NEG
    else:
        log_z = chunk_mass[0, n]
    n_nodes = 0
    n_apps = 0
    n_real = 0
    n_chunks = 0.0
    if mode == 1 or log_z <= NEG:
        return log_z, n_chunks, n_nodes, n_apps, n_real, n_pruned, dropped_sum, dropped_max, bp_k, bp_r, inside, active, n_active, prefix, chunk_mass
    # ---- outside
    outside = np.full((n + 1, n + 1, K), NEG)
    if use_chunks:
        for i in range(n):
            for j in range(i + 1, n + 1):
                if prefix[i] <= NEG or suffix[j] <= NEG:
                    continue
                base = prefix[i] + (lc if i > 0 else 0.0) + suffix[j]
                for t in range(n_active[i, j]):
                    c = active[i, j, t]
                    if root_lp[c] > NEG:
                        outside[i, j, c] = _lse(outside[i, j, c], root_lp[c] + base)
                        p = math.exp(inside[i, j, c] + root_lp[c] + base - log_z)
                        root_counts[c] += p
                        n_chunks += p
    else:
        for t in range(n_active[0, n]):
            c = active[0, n, t]
            if root_lp[c] > NEG:
                outside[0, n, c] = root_lp[c]
                p = math.exp(inside[0, n, c] + root_lp[c] - log_z)
                root_counts[c] += p
                n_chunks += p
    for length in range(n, 1, -1):
        for i in range(0, n - length + 1):
            j = i + length
            sp = span_pot[i, j] if has_span else 0.0
            for k in range(i + 1, j):
                na = n_active[i, k]
                if na == 0 or n_active[k, j] == 0:
                    continue
                for ai in range(na):
                    a = active[i, k, ai]
                    va = inside[i, k, a]
                    for r in range(rule_ptr[a], rule_ptr[a + 1]):
                        b = rule_B[r]
                        vb = inside[k, j, b]
                        if vb <= NEG:
                            continue
                        c = rule_C[r]
                        oc = outside[i, j, c]
                        if oc <= NEG:
                            continue
                        base = oc + sp + rule_lp[r]
                        outside[i, k, a] = _lse(outside[i, k, a], base + vb)
                        outside[k, j, b] = _lse(outside[k, j, b], base + va)
                        p = math.exp(base + va + vb - log_z)
                        rule_counts[r] += p
                        if p >= app_thr:
                            if n_apps < app_buf.shape[0]:
                                app_buf[n_apps, 0] = r
                                app_buf[n_apps, 1] = i
                                app_buf[n_apps, 2] = k
                                app_buf[n_apps, 3] = j
                                app_buf[n_apps, 4] = p
                                n_apps += 1
                            if has_tid:
                                tl = span_tid[i, k]
                                tr = span_tid[k, j]
                                if tl >= 0 and n_real < real_buf.shape[0]:
                                    real_buf[n_real, 0] = r
                                    real_buf[n_real, 1] = tl
                                    real_buf[n_real, 2] = p
                                    n_real += 1
                                if tr >= 0 and tr != tl and n_real < real_buf.shape[0]:  # one count per realisation type
                                    real_buf[n_real, 0] = r
                                    real_buf[n_real, 1] = tr
                                    real_buf[n_real, 2] = p
                                    n_real += 1
    # ---- lexical counts and node posteriors
    for i in range(n):
        w = tokens[i]
        for t in range(n_active[i, i + 1]):
            c = active[i, i + 1, t]
            o = outside[i, i + 1, c]
            if o <= NEG:
                continue
            p = math.exp(inside[i, i + 1, c] + o - log_z)
            if is_unk[i]:
                unk_counts[c] += p
            else:
                for e in range(lex_ptr[w], lex_ptr[w + 1]):
                    if lex_cls[e] == c:
                        lex_counts[lex_eid[e]] += p
                        break
    for i in range(n):
        for j in range(i + 1, n + 1):
            for t in range(n_active[i, j]):
                c = active[i, j, t]
                o = outside[i, j, c]
                if o <= NEG:
                    continue
                p = math.exp(inside[i, j, c] + o - log_z)
                if p >= post_thr and n_nodes < node_buf.shape[0]:
                    node_buf[n_nodes, 0] = i
                    node_buf[n_nodes, 1] = j
                    node_buf[n_nodes, 2] = c
                    node_buf[n_nodes, 3] = p
                    n_nodes += 1
    return log_z, n_chunks, n_nodes, n_apps, n_real, n_pruned, dropped_sum, dropped_max, bp_k, bp_r, inside, active, n_active, prefix, chunk_mass


# --------------------------------------------------------------------------- Python wrapper
class CompiledArrays:
    """Flat arrays of a :class:`CompiledGrammar` for the kernel."""

    def __init__(self, g: CompiledGrammar, vocab_size: int) -> None:
        K = g.n_classes
        self.K = K
        n_rules = len(g.rules)
        order = sorted(range(n_rules), key=lambda r: (g.rules[r][1], r))
        self.rule_order = np.asarray(order, dtype=np.int64)  # kernel rule index -> g.rules index
        self.rule_C = np.asarray([g.rules[r][0] for r in order], dtype=np.int32)
        self.rule_B = np.asarray([g.rules[r][2] for r in order], dtype=np.int32)
        self.rule_lp = np.asarray([g.rules[r][3] for r in order], dtype=np.float64)
        rule_A = np.asarray([g.rules[r][1] for r in order], dtype=np.int64)
        ptr = np.zeros(K + 1, dtype=np.int64)
        np.add.at(ptr, rule_A + 1, 1)
        self.rule_ptr = np.cumsum(ptr)
        # lexical CSR by token id
        V = max(vocab_size, (max(g.lex) + 1) if g.lex else 0)
        counts = np.zeros(V + 1, dtype=np.int64)
        for w, ents in g.lex.items():
            counts[w + 1] = len(ents)
        self.lex_ptr = np.cumsum(counts)
        n_ent = int(self.lex_ptr[-1])
        self.lex_cls = np.zeros(n_ent, dtype=np.int32)
        self.lex_lp = np.zeros(n_ent, dtype=np.float64)
        self.lex_eid = np.zeros(n_ent, dtype=np.int32)
        self.entry_keys: list[tuple[int, int]] = []  # entry id -> (compact class, token)
        e = 0
        for w in range(V):
            for c, lp in g.lex.get(w, []):
                self.lex_cls[e] = c
                self.lex_lp[e] = lp
                self.lex_eid[e] = len(self.entry_keys)
                self.entry_keys.append((c, w))
                e += 1
        self.unk_lp = np.where(np.isfinite(g.unk), g.unk, NEG).astype(np.float64)
        self.root_lp = np.where(np.isfinite(g.root), g.root, NEG).astype(np.float64)
        cl = getattr(g, "chunk_logp", None)
        self.use_chunks = cl is not None
        self.lc, self.ls = (float(cl[0]), float(cl[1])) if cl is not None else (NEG, NEG)


def run_sentence(
    arrs: CompiledArrays,
    tokens: np.ndarray,
    leaf_pot: np.ndarray | None,
    span_pot: np.ndarray | None,
    span_tid: np.ndarray | None,
    beam: int,
    prune_threshold: float,
    mode: int,
    post_thr: float,
    app_thr: float,
    counts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bufs: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    n = len(tokens)
    K = arrs.K
    has_leaf = leaf_pot is not None
    has_span = span_pot is not None
    has_tid = span_tid is not None
    lp = np.where(np.isfinite(leaf_pot), leaf_pot, NEG).astype(np.float64) if has_leaf else np.zeros((1, 1))
    sp = np.where(np.isfinite(span_pot), span_pot, NEG).astype(np.float64) if has_span else np.zeros((1, 1))
    st = span_tid.astype(np.int32) if has_tid else np.zeros((1, 1), dtype=np.int32)
    log_thr = math.log(prune_threshold) if prune_threshold > 0 else NEG
    return _kernel(
        tokens.astype(np.int64), K, arrs.lex_ptr, arrs.lex_cls, arrs.lex_lp, arrs.lex_eid, arrs.unk_lp,
        arrs.rule_ptr, arrs.rule_C, arrs.rule_B, arrs.rule_lp, arrs.root_lp,
        lp, has_leaf, sp, has_span, int(beam), log_thr, int(mode), arrs.use_chunks, arrs.lc, arrs.ls,
        st, has_tid, float(post_thr), float(app_thr),
        counts[0], counts[1], counts[2], counts[3], bufs[0], bufs[1], bufs[2],
    )


def e_step(vi, g: CompiledGrammar, sids: list[int], collect: bool):
    """E-step over ``sids`` for a :class:`cavi.VariationalInference` (numba backend)."""
    from .cavi import EStepOutput

    vocab_size = int(max((int(s.max()) if len(s) else 0) for s in vi.sentences) + 1) if vi.sentences else 1
    arrs = CompiledArrays(g, vocab_size)
    K = arrs.K
    class_ids = g.class_ids
    rule_keys = g.rule_keys  # type: ignore[attr-defined]
    n_rules = len(arrs.rule_C)
    rule_counts = np.zeros(n_rules)
    lex_counts = np.zeros(max(1, len(arrs.entry_keys)))
    unk_counts = np.zeros(K)
    root_counts = np.zeros(K)
    node_buf = np.zeros((200000, 4))
    app_buf = np.zeros((200000, 5))
    real_buf = np.zeros((200000, 3))
    ec = ExpectedCounts()
    out = EStepOutput()
    real_acc: dict[tuple[int, int], float] = {}
    for sid in sids:
        toks = vi.sentences[sid]
        n = len(toks)
        lp = vi.potentials.leaf_array(sid, n, class_ids, vi.class_map)
        sp = vi.potentials.span_array(sid)
        st = vi.span_types[sid] if vi.span_types is not None else None
        res = run_sentence(arrs, toks, lp, sp, st, vi.max_classes_per_cell, vi.prune_threshold, 0, vi.posterior_threshold, vi.app_threshold,
                           (rule_counts, lex_counts, unk_counts, root_counts), (node_buf, app_buf, real_buf))
        log_z, n_chunks, n_nodes, n_apps, n_real, n_pruned, d_sum, d_max = res[:8]
        out.pruned_cells += int(n_pruned)
        out.n_cells += n * (n + 1) // 2
        if n_pruned:
            out.dropped_mass.append(float(d_sum / n_pruned))
        if log_z <= NEG:
            ec.n_underivable += 1
            out.log_z[sid] = NEG_INF
            continue
        out.log_z[sid] = float(log_z)
        ec.log_z += float(log_z)
        ec.n_sentences += 1
        ec.n_chunks += float(n_chunks)
        for q in range(int(n_real)):
            r, t, p = int(real_buf[q, 0]), int(real_buf[q, 1]), float(real_buf[q, 2])
            key = (r, t)
            real_acc[key] = real_acc.get(key, 0.0) + p
        if collect:
            out.node_posteriors[sid] = [(int(node_buf[q, 0]), int(node_buf[q, 1]), class_ids[int(node_buf[q, 2])], float(node_buf[q, 3])) for q in range(int(n_nodes))]
            out.rule_apps[sid] = [(rule_keys[int(arrs.rule_order[int(app_buf[q, 0])])], int(app_buf[q, 1]), int(app_buf[q, 2]), int(app_buf[q, 3]), float(app_buf[q, 4])) for q in range(int(n_apps))]
    for r in range(n_rules):
        if rule_counts[r] > 0:
            ec.rule[rule_keys[int(arrs.rule_order[r])]] = float(rule_counts[r])
    for e, (c, w) in enumerate(arrs.entry_keys):
        if lex_counts[e] > 0:
            ec.lex[(class_ids[c], w)] = float(lex_counts[e])
    for c in range(K):
        if unk_counts[c] > 0:
            ec.lex[(class_ids[c], UNK)] = float(unk_counts[c])
        if root_counts[c] > 0:
            ec.root[class_ids[c]] = float(root_counts[c])
    for (r, t), p in real_acc.items():
        key = rule_keys[int(arrs.rule_order[r])]
        ec.rule_realisations.setdefault(key, {})[t] = ec.rule_realisations.get(key, {}).get(t, 0.0) + p
    ec.pruning = vi._pruning_stats(out)
    return ec, out


def viterbi_parse(vi, g: CompiledGrammar, sid: int, arrs: "CompiledArrays | None" = None):
    """Best derivation with the numba kernel: returns ``(score, nodes, chunks)`` like :func:`chart.viterbi`.

    Pass a prebuilt ``arrs`` (``CompiledArrays(g, vocab_size)``) when parsing many sentences.
    """
    from .chart import Derivation

    if arrs is None:
        vocab_size = int(max((int(s.max()) if len(s) else 0) for s in vi.sentences) + 1)
        arrs = CompiledArrays(g, vocab_size)
    toks = vi.sentences[sid]
    n = len(toks)
    lp = vi.potentials.leaf_array(sid, n, g.class_ids, vi.class_map)
    sp = vi.potentials.span_array(sid)
    K = arrs.K
    dummy = (np.zeros(1), np.zeros(1), np.zeros(K), np.zeros(K))
    bufs = (np.zeros((1, 4)), np.zeros((1, 5)), np.zeros((1, 3)))
    res = run_sentence(arrs, toks, lp, sp, None, 0, 0.0, 1, 1.0, 1.0, dummy, bufs)
    log_z, _, _, _, _, _, _, _, bp_k, bp_r, inside, active, n_active, prefix, chunk_mass = res
    if log_z <= NEG:
        return None

    def best_class(i: int, j: int) -> int:
        b, bc = NEG, -1
        for t in range(n_active[i, j]):
            c = int(active[i, j, t])
            if arrs.root_lp[c] > NEG and inside[i, j, c] + arrs.root_lp[c] > b:
                b, bc = inside[i, j, c] + arrs.root_lp[c], c
        return bc

    chunks: list[tuple[int, int, int]] = []
    if not arrs.use_chunks:
        chunks.append((0, n, best_class(0, n)))
    else:
        bp = [-1] * (n + 1)
        bestp = [NEG] * (n + 1)
        bestp[0] = 0.0
        for j in range(1, n + 1):
            for i in range(0, j):
                m = chunk_mass[i, j]
                if m <= NEG or bestp[i] <= NEG:
                    continue
                v = bestp[i] + m + (arrs.lc if i > 0 else 0.0)
                if v > bestp[j]:
                    bestp[j], bp[j] = v, i
        j = n
        while j > 0:
            i = bp[j]
            chunks.append((i, j, best_class(i, j)))
            j = i
        chunks.reverse()
    nodes: list[tuple[int, int, int, int, int]] = []
    stack = list(reversed(chunks))
    while stack:
        i, j, c = stack.pop()
        k = int(bp_k[i, j, c])
        r = int(bp_r[i, j, c])
        rr = int(arrs.rule_order[r]) if r >= 0 else -1
        nodes.append((i, j, c, k, rr))
        if k >= 0:
            _, a, b, _ = g.rules[rr]
            stack.append((k, j, b))
            stack.append((i, k, a))
    return Derivation(float(log_z), nodes, chunks[0][2], chunks)
