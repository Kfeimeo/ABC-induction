"""Multi-scale left/right context features for span occurrences.

For an occurrence ``s[i:j]`` and a scale ``k`` the joint context feature is

    (k, s[i-k:i], s[j:j+k])

with BOS/EOS padding at sentence boundaries.  Optionally one-sided features
(left only / right only) are added as well.  Features are interned to integer
ids; the per-occurrence features are stored in a dense ``int64`` matrix of
shape ``(n_occurrences, n_feature_slots)`` and the span-type x context counts
in a ``scipy.sparse`` CSR matrix (never a dense |X| x |C| matrix).

The interface is deliberately simple so that an adaptive / unbounded context
refinement could replace :func:`extract_contexts` later (v0.1 only implements
fixed scales).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .corpus import BOS_ID, EOS_ID, Corpus
from .spans import SpanInventory

ContextKey = tuple[int, tuple[int, ...], tuple[int, ...]]  # (scale, left, right)

KIND_CODE = {"joint": 0, "left": 1, "right": 2}


@dataclass(frozen=True)
class ContextFeature:
    """Human-readable view of a context feature."""

    left_tokens: tuple[str, ...]
    right_tokens: tuple[str, ...]
    scale: int

    def render(self) -> str:
        left = " ".join(self.left_tokens)
        right = " ".join(self.right_tokens)
        return f"[k={self.scale}] {left} _ {right}".strip()


class ContextVocab:
    """Interns context keys to integer ids.

    Contexts whose total count is below ``min_context_count`` are not stored
    (they receive unique ids >= ``n_stored``); they still carry probability
    mass but can never be shared between span types.
    """

    def __init__(self) -> None:
        self.key_to_id: dict[ContextKey, int] = {}
        self.keys: list[ContextKey] = []
        self.n_unique_extra = 0

    @property
    def n_stored(self) -> int:
        return len(self.keys)

    @property
    def size(self) -> int:
        return self.n_stored + self.n_unique_extra

    def intern(self, key: ContextKey) -> int:
        cid = self.key_to_id.get(key)
        if cid is None:
            cid = len(self.keys)
            self.key_to_id[key] = cid
            self.keys.append(key)
        return cid

    def feature(self, cid: int, corpus: Corpus) -> ContextFeature | None:
        if cid >= self.n_stored:
            return None
        k, left, right = self.keys[cid]
        return ContextFeature(
            tuple(corpus.itos[t] for t in left), tuple(corpus.itos[t] for t in right), k
        )

    def render(self, cid: int, corpus: Corpus) -> str:
        f = self.feature(cid, corpus)
        return f.render() if f is not None else f"<rare-context#{cid}>"


@dataclass
class ContextData:
    """Everything the induction needs about contexts."""

    vocab: ContextVocab
    occ_ctx: np.ndarray  # (n_occ, n_slots) int64 context ids
    slot_scale: np.ndarray  # scale per slot
    slot_kind: list[str]  # 'joint' | 'left' | 'right'
    ctx_scale: np.ndarray  # scale per context column
    ctx_kind: np.ndarray  # 0 joint, 1 left, 2 right per column
    type_ctx: sp.csr_matrix  # span types x contexts counts

    @property
    def n_contexts(self) -> int:
        return int(self.type_ctx.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.type_ctx.nnz)

    def column_weights(self, scale_weights: dict[int, float] | None, sided_weight: float = 1.0) -> np.ndarray:
        """Per-context-column weight derived from the scale and the kind."""
        w = np.ones(self.n_contexts, dtype=np.float64)
        if scale_weights:
            for k, wk in scale_weights.items():
                w[self.ctx_scale == int(k)] *= float(wk)
        if sided_weight != 1.0:
            w[self.ctx_kind != 0] *= sided_weight
        return w

    def occ_context_counts(self, oids: np.ndarray) -> dict[int, float]:
        """Sparse aggregated context counts of a set of occurrences."""
        out: dict[int, float] = {}
        for row in self.occ_ctx[oids].reshape(-1).tolist():
            out[row] = out.get(row, 0.0) + 1.0
        return out


def _context_keys(toks: list[int], start: int, end: int, k: int, one_sided: bool) -> list[tuple[str, ContextKey]]:
    n = len(toks)
    left = tuple([BOS_ID] * max(0, k - start) + toks[max(0, start - k):start])
    right = tuple(toks[end:end + k] + [EOS_ID] * max(0, end + k - n))
    out: list[tuple[str, ContextKey]] = [("joint", (k, left, right))]
    if one_sided:
        out.append(("left", (k, left, ())))
        out.append(("right", (k, (), right)))
    return out


def extract_contexts(
    corpus: Corpus,
    inv: SpanInventory,
    scales: tuple[int, ...] = (1, 2, 3, 4),
    one_sided: bool = False,
    min_context_count: int = 1,
) -> ContextData:
    """Compute multi-scale context features for every span occurrence.

    Args:
        scales: context widths ``k``.
        one_sided: also add left-only and right-only features.
        min_context_count: contexts occurring fewer times than this are not
            stored (they get unique ids and cannot be shared).
    """
    n_occ = inv.n_occurrences
    slot_kind: list[str] = []
    slot_scale: list[int] = []
    for k in scales:
        slot_kind.append("joint")
        slot_scale.append(int(k))
        if one_sided:
            slot_kind += ["left", "right"]
            slot_scale += [int(k), int(k)]
    n_slots = len(slot_kind)
    vocab = ContextVocab()
    occ_ctx = np.zeros((n_occ, n_slots), dtype=np.int64)
    kind_of: list[int] = []
    tok_cache = [ids.tolist() for ids in corpus.token_ids]
    for oid in range(n_occ):
        toks = tok_cache[inv.occ_sid[oid]]
        s, e = int(inv.occ_start[oid]), int(inv.occ_end[oid])
        slot = 0
        for k in scales:
            for kind, key in _context_keys(toks, s, e, int(k), one_sided):
                cid = vocab.intern(key)
                if cid == len(kind_of):
                    kind_of.append(KIND_CODE[kind])
                occ_ctx[oid, slot] = cid
                slot += 1
    ctx_scale = np.asarray([k[0] for k in vocab.keys], dtype=np.int16)
    ctx_kind = np.asarray(kind_of, dtype=np.int8)
    if min_context_count > 1 and n_occ > 0:
        counts = np.bincount(occ_ctx.reshape(-1), minlength=vocab.n_stored)
        keep = counts >= min_context_count
        new_vocab = ContextVocab()
        remap = np.full(vocab.n_stored, -1, dtype=np.int64)
        for old in np.nonzero(keep)[0].tolist():
            remap[old] = new_vocab.intern(vocab.keys[old])
        rare_mask = ~keep[occ_ctx]
        n_rare = int(rare_mask.sum())
        remapped = remap[occ_ctx]
        rare_old = occ_ctx[rare_mask]
        remapped[rare_mask] = np.arange(new_vocab.n_stored, new_vocab.n_stored + n_rare)
        new_vocab.n_unique_extra = n_rare
        ctx_scale = np.concatenate([ctx_scale[keep], ctx_scale[rare_old]])
        ctx_kind = np.concatenate([ctx_kind[keep], ctx_kind[rare_old]])
        occ_ctx = remapped
        vocab = new_vocab
    n_ctx = max(vocab.size, 1)
    cols = occ_ctx.reshape(-1)
    type_rows = np.repeat(inv.occ_tid.astype(np.int64), n_slots)
    type_ctx = sp.csr_matrix(
        (np.ones(len(type_rows), dtype=np.float32), (type_rows, cols)), shape=(inv.n_types, n_ctx)
    )
    type_ctx.sum_duplicates()
    return ContextData(
        vocab=vocab,
        occ_ctx=occ_ctx,
        slot_scale=np.asarray(slot_scale, dtype=np.int16),
        slot_kind=slot_kind,
        ctx_scale=ctx_scale,
        ctx_kind=ctx_kind,
        type_ctx=type_ctx,
    )
