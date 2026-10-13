"""Corpus loading and tokenisation.

A corpus is a list of sentences; every sentence is a tuple of token ids.
Token ids 0 and 1 are reserved for the BOS/EOS boundary markers used by
context extraction (see :mod:`abc_induction.contexts`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

BOS_ID = 0
EOS_ID = 1
BOS = "<BOS>"
EOS = "<EOS>"

_PUNCT_RE = re.compile(r"([.,!?;:\"()\[\]])")


def tokenize(line: str, punctuation_mode: str = "separate_tokens", lowercase: bool = False) -> list[str]:
    """Whitespace tokeniser with configurable punctuation handling.

    Args:
        line: raw sentence string.
        punctuation_mode: ``separate_tokens`` keeps punctuation as its own
            tokens; ``remove`` deletes it; ``keep`` leaves the string untouched.
        lowercase: lowercase the text before splitting.
    """
    if lowercase:
        line = line.lower()
    if punctuation_mode == "separate_tokens":
        line = _PUNCT_RE.sub(r" \1 ", line)
    elif punctuation_mode == "remove":
        line = _PUNCT_RE.sub(" ", line)
    elif punctuation_mode != "keep":
        raise ValueError(f"unknown punctuation_mode {punctuation_mode!r}")
    return line.split()


@dataclass
class Sentence:
    """One tokenised sentence."""

    sid: int
    tokens: tuple[str, ...]


@dataclass
class Corpus:
    """A tokenised corpus with an integer vocabulary.

    ``token_ids[sid]`` is an ``int32`` array of token ids for sentence ``sid``.
    """

    sentences: list[Sentence] = field(default_factory=list)
    vocab: dict[str, int] = field(default_factory=lambda: {BOS: BOS_ID, EOS: EOS_ID})
    itos: list[str] = field(default_factory=lambda: [BOS, EOS])
    token_ids: list[np.ndarray] = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_token_lists(
        cls,
        token_lists: Iterable[list[str]],
        min_sentence_length: int = 1,
        max_sentence_length: int = 15,
        remove_duplicate_sentences: bool = False,
        max_sentences: int | None = None,
    ) -> "Corpus":
        """Build a corpus from already tokenised sentences, applying length filters."""
        corpus = cls()
        seen: set[tuple[str, ...]] = set()
        for toks in token_lists:
            if not (min_sentence_length <= len(toks) <= max_sentence_length):
                continue
            tt = tuple(toks)
            if remove_duplicate_sentences:
                if tt in seen:
                    continue
                seen.add(tt)
            corpus.add_sentence(tt)
            if max_sentences is not None and len(corpus.sentences) >= max_sentences:
                break
        return corpus

    @classmethod
    def from_text_file(
        cls, path: str, punctuation_mode: str = "separate_tokens", lowercase: bool = False, **kwargs
    ) -> "Corpus":
        """Read one sentence per line from a UTF-8 text file."""

        def gen() -> Iterator[list[str]]:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield tokenize(line, punctuation_mode, lowercase)

        return cls.from_token_lists(gen(), **kwargs)

    def add_sentence(self, tokens: tuple[str, ...]) -> int:
        sid = len(self.sentences)
        ids = np.empty(len(tokens), dtype=np.int32)
        for i, tok in enumerate(tokens):
            tid = self.vocab.get(tok)
            if tid is None:
                tid = len(self.itos)
                self.vocab[tok] = tid
                self.itos.append(tok)
            ids[i] = tid
        self.sentences.append(Sentence(sid, tokens))
        self.token_ids.append(ids)
        return sid

    # ------------------------------------------------------------------ query
    def __len__(self) -> int:
        return len(self.sentences)

    @property
    def n_tokens(self) -> int:
        return int(sum(len(s) for s in self.token_ids))

    def render(self, ids: Iterable[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def subset(self, sids: Iterable[int]) -> "Corpus":
        """Return a new corpus containing only the given sentences (re-indexed)."""
        return Corpus.from_token_lists(
            [list(self.sentences[s].tokens) for s in sids], max_sentence_length=10**9
        )
