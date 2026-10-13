"""Sparse count matrices and inverted-index style candidate generation.

All span-type/class x context statistics live in ``scipy.sparse`` CSR
matrices.  Candidate pairs for similarity computation are generated through
the inverted index (context -> rows), which is exactly what a sparse boolean
product ``B @ B.T`` computes, so we never compare all |X|^2 pairs.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def row_sums(m: sp.csr_matrix) -> np.ndarray:
    return np.asarray(m.sum(axis=1)).ravel()


def document_frequency(m: sp.csr_matrix) -> np.ndarray:
    """Number of rows in which each column is non-zero."""
    b = m.tocsr(copy=True)
    b.data = np.ones_like(b.data)
    return np.asarray(b.sum(axis=0)).ravel()


def normalize_rows(m: sp.csr_matrix, col_weights: np.ndarray | None = None) -> sp.csr_matrix:
    """Row-normalise (optionally after column weighting) to probability rows."""
    m = sp.csr_matrix(m, dtype=np.float64)
    if col_weights is not None:
        m = sp.csr_matrix(m @ sp.diags(col_weights.astype(np.float64)))
        m.eliminate_zeros()
    s = row_sums(m)
    s[s == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / s) @ m)


def row_dict(m: sp.csr_matrix, r: int) -> dict[int, float]:
    a, b = m.indptr[r], m.indptr[r + 1]
    return dict(zip(m.indices[a:b].tolist(), m.data[a:b].tolist()))


def row_arrays(m: sp.csr_matrix, r: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = m.indptr[r], m.indptr[r + 1]
    return m.indices[a:b], m.data[a:b]


def candidate_pairs(
    m: sp.csr_matrix, max_df: int, min_shared: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverted-index candidate generation.

    Returns the upper-triangular pairs ``(i, j, shared)`` of rows that share
    at least ``min_shared`` contexts, ignoring contexts with document
    frequency > ``max_df`` (stop-contexts that would create O(n^2) pairs).
    """
    b = m.tocsr(copy=True)
    b.data = np.ones_like(b.data, dtype=np.float32)
    df = np.asarray(b.sum(axis=0)).ravel()
    keep = (df <= max_df) & (df >= 2)
    b = sp.csr_matrix(b @ sp.diags(keep.astype(np.float32)))
    b.eliminate_zeros()
    shared = sp.triu(b @ b.T, k=1).tocoo()
    mask = shared.data >= min_shared
    return shared.row[mask], shared.col[mask], shared.data[mask]


def aggregate_rows(m: sp.csr_matrix, groups: list[list[int]]) -> sp.csr_matrix:
    """Sum the rows of ``m`` for every group (membership-matrix product)."""
    if groups:
        rows = np.concatenate([np.full(len(g), gi, dtype=np.int64) for gi, g in enumerate(groups)])
        cols = np.concatenate([np.asarray(g, dtype=np.int64) for g in groups])
    else:
        rows = np.zeros(0, dtype=np.int64)
        cols = np.zeros(0, dtype=np.int64)
    memb = sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(len(groups), m.shape[0]))
    out = sp.csr_matrix(memb @ m)
    out.sum_duplicates()
    return out


# ------------------------------------------------------------------ similarity
def weighted_jaccard(ia: np.ndarray, da: np.ndarray, ib: np.ndarray, db: np.ndarray) -> float:
    """sum_c min(N_ac, N_bc) / sum_c max(N_ac, N_bc) on sparse rows (sorted indices)."""
    common, pa, pb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    smin = float(np.minimum(da[pa], db[pb]).sum())
    smax = float(da.sum() + db.sum() - smin)
    return smin / smax if smax > 0 else 0.0


def js_similarity(ia: np.ndarray, da: np.ndarray, ib: np.ndarray, db: np.ndarray) -> float:
    """``1 - JS(P_a, P_b)`` (base 2) between the row distributions.

    No smoothing is required: the mixture ``M`` is positive wherever either
    distribution is, and the union of supports is iterated sparsely.
    """
    pa = da / da.sum()
    pb = db / db.sum()
    common, ca, cb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    # contributions on the shared support
    p, q = pa[ca], pb[cb]
    m = 0.5 * (p + q)
    js = 0.5 * float((p * np.log2(p / m)).sum()) + 0.5 * float((q * np.log2(q / m)).sum())
    # contributions off the shared support: p*log(p/(p/2)) = p*1 bit
    mass_a_only = 1.0 - float(p.sum())
    mass_b_only = 1.0 - float(q.sum())
    js += 0.5 * mass_a_only + 0.5 * mass_b_only
    return max(0.0, 1.0 - js)


def bhattacharyya(ia: np.ndarray, da: np.ndarray, ib: np.ndarray, db: np.ndarray) -> float:
    """``sum_c sqrt(P_a(c) P_b(c))`` between the row distributions (in [0, 1])."""
    pa = da / da.sum()
    pb = db / db.sum()
    common, ca, cb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    return float(np.sqrt(pa[ca] * pb[cb]).sum())


SIMILARITIES = {"jaccard": weighted_jaccard, "js": js_similarity, "bhattacharyya": bhattacharyya}


def pairwise_bhattacharyya(p: sp.csr_matrix, max_df: int, min_sim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Bhattacharyya similarity between all rows of a probability
    matrix ``p`` that share at least one context, via ``sqrt(p) sqrt(p)^T``.

    Columns with document frequency > ``max_df`` are ignored (treated as
    uninformative stop-contexts) for the purpose of generating candidates and
    computing the coefficient.
    """
    q = p.tocsr(copy=True)
    df = document_frequency(q)
    keep = (df <= max_df) & (df >= 2)
    q = sp.csr_matrix(q @ sp.diags(keep.astype(np.float64)))
    q.eliminate_zeros()
    q.data = np.sqrt(q.data)
    s = sp.triu(q @ q.T, k=1).tocoo()
    mask = s.data >= min_sim
    return s.row[mask], s.col[mask], s.data[mask]
