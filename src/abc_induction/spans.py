"""Candidate span enumeration.

Strictly distinguishes

* ``SpanType``  - a surface token sequence such as ``"the dog"`` and
* ``SpanOccurrence`` - one position ``(sentence_id, start, end)`` where that
  token sequence occurs.

No parse trees and no POS information are used here: every contiguous span up
to ``max_span_length`` whose type occurs at least ``min_span_frequency`` times
in the corpus becomes a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .corpus import Corpus


@dataclass(frozen=True)
class SpanType:
    """A surface token sequence (token ids)."""

    tid: int
    tokens: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class SpanOccurrence:
    """One occurrence of a span type at ``sentence[start:end]``."""

    oid: int
    sid: int
    start: int
    end: int
    tid: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _pack(sid: int, start: int, end: int) -> int:
    return (sid << 16) | (start << 8) | end


class SpanInventory:
    """Columnar store of candidate span types and occurrences.

    Attributes:
        types: list of :class:`SpanType` (index = tid).
        occ_sid / occ_start / occ_end / occ_tid: int32 arrays, index = oid.
        type_occ: for each tid, the array of its occurrence ids.
        type_freq: occurrences per type.
        sent_occ: for each sentence, the array of occurrence ids in it.
    """

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.types: list[SpanType] = []
        self.type_index: dict[tuple[int, ...], int] = {}
        self.occ_sid = np.zeros(0, dtype=np.int32)
        self.occ_start = np.zeros(0, dtype=np.int32)
        self.occ_end = np.zeros(0, dtype=np.int32)
        self.occ_tid = np.zeros(0, dtype=np.int32)
        self.type_occ: list[np.ndarray] = []
        self.type_freq = np.zeros(0, dtype=np.int64)
        self._pos: dict[int, int] = {}
        self.sent_occ: list[np.ndarray] = []
        self.n_candidate_spans_total = 0  # before frequency filtering
        self.n_candidate_types_total = 0

    # ---------------------------------------------------------------- access
    @property
    def n_types(self) -> int:
        return len(self.types)

    @property
    def n_occurrences(self) -> int:
        return int(len(self.occ_tid))

    def occurrence(self, oid: int) -> SpanOccurrence:
        return SpanOccurrence(
            oid, int(self.occ_sid[oid]), int(self.occ_start[oid]), int(self.occ_end[oid]), int(self.occ_tid[oid])
        )

    def lookup(self, sid: int, start: int, end: int) -> int:
        """Return the occurrence id at ``(sid, start, end)`` or -1."""
        return self._pos.get(_pack(sid, start, end), -1)

    def type_str(self, tid: int) -> str:
        return self.corpus.render(self.types[tid].tokens)

    def type_length(self, tid: int) -> int:
        return self.types[tid].length

    def occ_str(self, oid: int) -> str:
        return self.type_str(int(self.occ_tid[oid]))

    def iter_types(self) -> Iterator[SpanType]:
        return iter(self.types)

    # ---------------------------------------------------------------- build
    def finalize(self, sids: list[int], starts: list[int], ends: list[int], tids: list[int]) -> None:
        self.occ_sid = np.asarray(sids, dtype=np.int32)
        self.occ_start = np.asarray(starts, dtype=np.int32)
        self.occ_end = np.asarray(ends, dtype=np.int32)
        self.occ_tid = np.asarray(tids, dtype=np.int32)
        n_types = len(self.types)
        order = np.argsort(self.occ_tid, kind="stable")
        counts = np.bincount(self.occ_tid, minlength=n_types) if len(tids) else np.zeros(n_types, dtype=np.int64)
        self.type_freq = counts.astype(np.int64)
        bounds = np.concatenate([[0], np.cumsum(counts)])
        self.type_occ = [order[bounds[t]:bounds[t + 1]].astype(np.int32) for t in range(n_types)]
        order_s = np.argsort(self.occ_sid, kind="stable")
        n_sent = len(self.corpus)
        counts_s = np.bincount(self.occ_sid, minlength=n_sent) if len(sids) else np.zeros(n_sent, dtype=np.int64)
        bounds_s = np.concatenate([[0], np.cumsum(counts_s)])
        self.sent_occ = [order_s[bounds_s[s]:bounds_s[s + 1]].astype(np.int32) for s in range(n_sent)]
        self._pos = {_pack(int(s), int(a), int(b)): oid for oid, (s, a, b) in enumerate(zip(sids, starts, ends))}


def enumerate_spans(corpus: Corpus, max_span_length: int = 8, min_span_frequency: int = 2) -> SpanInventory:
    """Enumerate all contiguous spans ``s[i:j]`` (length <= ``max_span_length``)
    and keep the span types occurring at least ``min_span_frequency`` times.

    A first pass counts span types through 64-bit tuple hashes (memory-light);
    the second pass materialises occurrences of the surviving types only.
    """
    inv = SpanInventory(corpus)
    hashes: list[int] = []
    for ids in corpus.token_ids:
        toks = ids.tolist()
        n = len(toks)
        for i in range(n):
            for j in range(i + 1, min(n, i + max_span_length) + 1):
                hashes.append(hash(tuple(toks[i:j])))
    inv.n_candidate_spans_total = len(hashes)
    if not hashes:
        inv.finalize([], [], [], [])
        return inv
    harr = np.fromiter(hashes, dtype=np.int64, count=len(hashes))
    del hashes
    uniq, cnt = np.unique(harr, return_counts=True)
    inv.n_candidate_types_total = int(len(uniq))
    keep = set(uniq[cnt >= min_span_frequency].tolist())
    del harr, uniq, cnt
    sids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    tids: list[int] = []
    for sid, ids in enumerate(corpus.token_ids):
        toks = ids.tolist()
        n = len(toks)
        for i in range(n):
            for j in range(i + 1, min(n, i + max_span_length) + 1):
                tt = tuple(toks[i:j])
                if hash(tt) not in keep:
                    continue
                tid = inv.type_index.get(tt)
                if tid is None:
                    tid = len(inv.types)
                    inv.type_index[tt] = tid
                    inv.types.append(SpanType(tid, tt))
                sids.append(sid)
                starts.append(i)
                ends.append(j)
                tids.append(tid)
    inv.finalize(sids, starts, ends, tids)
    return inv
