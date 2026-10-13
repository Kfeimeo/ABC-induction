"""Configurable synthetic grammar generator with full gold annotation.

A grammar consists of latent classes with (weighted) lexical realisations and
binary composition rules ``A + B -> C``.  Generation is top-down from the start
class; every derivation records

* gold constituent spans with their latent class,
* gold occurrence-level category assignments (needed for ambiguity),
* gold composition relations actually used in the sample.

Noise options (Synthetic-3) add rare words, Zipfian lexicons, random token
replacement, filler insertions (frequent non-constituent substrings) and
topic-driven lexical correlations that are irrelevant to syntax.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Rule:
    left: str
    right: str
    result: str
    weight: float = 1.0


@dataclass
class GoldSpan:
    start: int
    end: int
    cls: str
    is_leaf: bool
    noisy: bool = False


@dataclass
class GoldSentence:
    tokens: list[str]
    spans: list[GoldSpan]
    topic: int = 0
    noise_positions: list[int] = field(default_factory=list)


@dataclass
class SyntheticDataset:
    name: str
    sentences: list[GoldSentence]
    classes: list[str]
    rules: list[Rule]
    ambiguous_words: dict[str, dict[str, float]]
    relation_counts: dict[tuple[str, str, str], int]
    config: dict[str, Any]

    def token_lists(self) -> list[list[str]]:
        return [s.tokens for s in self.sentences]

    def gold_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classes": self.classes,
            "rules": [[r.left, r.right, r.result] for r in self.rules],
            "ambiguous_words": self.ambiguous_words,
            "relation_counts": {" ".join(k): v for k, v in self.relation_counts.items()},
            "sentences": [
                {
                    "tokens": s.tokens,
                    "topic": s.topic,
                    "noise_positions": s.noise_positions,
                    "spans": [[g.start, g.end, g.cls, g.is_leaf, g.noisy] for g in s.spans],
                }
                for s in self.sentences
            ],
        }


class SyntheticGrammar:
    """A weighted binary grammar with lexical classes."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.name: str = spec.get("name", "synthetic")
        self.start: str = spec["start"]
        self.max_depth: int = int(spec.get("max_depth", 6))
        self.max_length: int = int(spec.get("max_length", 15))
        self.lexicon: dict[str, dict[str, float]] = {}
        self.lex_weight: dict[str, float] = {}
        for cname, cspec in spec["classes"].items():
            cspec = cspec or {}
            self.lexicon[cname] = {str(w): float(v) for w, v in (cspec.get("lexicon") or {}).items()}
            self.lex_weight[cname] = float(cspec.get("lex_weight", 1.0))
        self.rules: list[Rule] = [
            Rule(r["left"], r["right"], r["result"], float(r.get("weight", 1.0))) for r in spec.get("rules", [])
        ]
        self.ambiguity: dict[str, dict[str, float]] = {
            str(w): {str(c): float(p) for c, p in d.items()} for w, d in (spec.get("ambiguity") or {}).items()
        }
        self.noise: dict[str, Any] = dict(spec.get("noise") or {})
        self.classes = list(spec["classes"].keys())
        self._rules_by_result: dict[str, list[Rule]] = {}
        for r in self.rules:
            self._rules_by_result.setdefault(r.result, []).append(r)
        self._min_depth = self._compute_min_depth()

    @classmethod
    def from_yaml(cls, path: str) -> "SyntheticGrammar":
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def _compute_min_depth(self) -> dict[str, int]:
        inf = 10**6
        md = {c: (0 if self.lexicon.get(c) else inf) for c in self.classes}
        changed = True
        while changed:
            changed = False
            for r in self.rules:
                d = 1 + max(md[r.left], md[r.right])
                if d < md[r.result]:
                    md[r.result] = d
                    changed = True
        for c, d in md.items():
            if d >= inf:
                raise ValueError(f"class {c} cannot be realised")
        return md

    # ------------------------------------------------------------------ setup
    def effective_lexicons(self, rng: random.Random) -> dict[str, dict[str, float]]:
        """Lexicon after applying ambiguity entries, rare words and Zipf weights."""
        lex = {c: dict(d) for c, d in self.lexicon.items()}
        rare_n = int(self.noise.get("rare_words_per_class", 0))
        rare_w = float(self.noise.get("rare_word_weight", 0.05))
        if rare_n > 0:
            for c in self.classes:
                if not lex[c]:
                    continue
                for i in range(rare_n):
                    lex[c][f"{c.lower()}_rare{i}"] = rare_w
        if self.noise.get("zipf_lexicon", False):
            s = float(self.noise.get("zipf_exponent", 1.0))
            for c in self.classes:
                items = sorted(lex[c].items(), key=lambda kv: -kv[1])
                lex[c] = {w: (1.0 / (i + 1) ** s) for i, (w, _) in enumerate(items)}
        for w, dist in self.ambiguity.items():
            for c, p in dist.items():
                # an ordinary-frequency member of each class: the mean lexical weight
                # of the class scaled by 2p, so that p : q controls the split of the
                # word's own occurrences while its overall frequency stays typical
                vals = [v for k, v in lex[c].items() if k not in self.ambiguity]
                mean_w = (sum(vals) / len(vals)) if vals else 1.0
                lex[c][w] = mean_w * 2.0 * p
        return lex

    # --------------------------------------------------------------- generate
    def generate(self, n_sentences: int, seed: int = 42) -> SyntheticDataset:
        rng = random.Random(seed)
        lex = self.effective_lexicons(rng)
        n_topics = int(self.noise.get("topics", 0))
        topic_strength = float(self.noise.get("topic_strength", 0.0))
        topic_lex: list[dict[str, dict[str, float]]] = []
        if n_topics > 0:
            for _t in range(n_topics):
                tl: dict[str, dict[str, float]] = {}
                for c, d in lex.items():
                    tl[c] = {w: v * math.exp(topic_strength * rng.gauss(0, 1)) for w, v in d.items()}
                topic_lex.append(tl)
        else:
            topic_lex = [lex]
        vocab = sorted({w for d in lex.values() for w in d})
        sentences: list[GoldSentence] = []
        rel_counts: dict[tuple[str, str, str], int] = {}
        attempts = 0
        while len(sentences) < n_sentences and attempts < n_sentences * 50:
            attempts += 1
            topic = rng.randrange(len(topic_lex))
            tokens: list[str] = []
            spans: list[GoldSpan] = []
            ok = self._expand(self.start, 0, tokens, spans, topic_lex[topic], rng, rel_counts, local_rels=[])
            if not ok or len(tokens) > self.max_length:
                continue
            sent = GoldSentence(tokens=tokens, spans=spans, topic=topic)
            self._apply_noise(sent, vocab, rng)
            sentences.append(sent)
        if len(sentences) < n_sentences:
            raise RuntimeError("could not generate enough sentences; check max_length / max_depth")
        return SyntheticDataset(
            name=self.name,
            sentences=sentences,
            classes=self.classes,
            rules=self.rules,
            ambiguous_words=self.ambiguity,
            relation_counts=rel_counts,
            config=self.spec,
        )

    def _expand(
        self,
        cls: str,
        depth: int,
        tokens: list[str],
        spans: list[GoldSpan],
        lex: dict[str, dict[str, float]],
        rng: random.Random,
        rel_counts: dict[tuple[str, str, str], int],
        local_rels: list,
    ) -> bool:
        start = len(tokens)
        options: list[tuple[float, Any]] = []
        lexicon = lex.get(cls, {})
        rules = self._rules_by_result.get(cls, [])
        force_terminal = depth >= self.max_depth
        if lexicon:
            options.append((self.lex_weight[cls], "lex"))
        if not force_terminal:
            for r in rules:
                options.append((r.weight, r))
        elif not lexicon:
            # over the depth budget: only the shallowest rules are allowed
            best = min(1 + max(self._min_depth[r.left], self._min_depth[r.right]) for r in rules)
            for r in rules:
                if 1 + max(self._min_depth[r.left], self._min_depth[r.right]) == best:
                    options.append((r.weight, r))
        if not options:
            return False
        total = sum(w for w, _ in options)
        x = rng.random() * total
        choice = options[-1][1]
        for w, o in options:
            x -= w
            if x <= 0:
                choice = o
                break
        if choice == "lex":
            words = list(lexicon.items())
            tw = sum(v for _, v in words)
            y = rng.random() * tw
            word = words[-1][0]
            for w, v in words:
                y -= v
                if y <= 0:
                    word = w
                    break
            tokens.append(word)
            spans.append(GoldSpan(start, start + 1, cls, True))
            return True
        r: Rule = choice
        if not self._expand(r.left, depth + 1, tokens, spans, lex, rng, rel_counts, local_rels):
            return False
        if not self._expand(r.right, depth + 1, tokens, spans, lex, rng, rel_counts, local_rels):
            return False
        spans.append(GoldSpan(start, len(tokens), cls, False))
        key = (r.left, r.right, r.result)
        rel_counts[key] = rel_counts.get(key, 0) + 1
        return True

    def _apply_noise(self, sent: GoldSentence, vocab: list[str], rng: random.Random) -> None:
        p_lex = float(self.noise.get("lexical_noise_rate", 0.0))
        if p_lex > 0:
            for i in range(len(sent.tokens)):
                if rng.random() < p_lex:
                    sent.tokens[i] = rng.choice(vocab)
                    sent.noise_positions.append(i)
                    for g in sent.spans:
                        if g.is_leaf and g.start == i:
                            g.noisy = True
        p_fill = float(self.noise.get("filler_rate", 0.0))
        fillers = list(self.noise.get("filler_tokens", ["uh"]))
        if p_fill > 0 and rng.random() < p_fill and len(sent.tokens) < self.max_length:
            pos = rng.randrange(len(sent.tokens) + 1)
            sent.tokens.insert(pos, rng.choice(fillers))
            for g in sent.spans:
                if g.start >= pos:
                    g.start += 1
                    g.end += 1
                elif g.end > pos:  # span strictly contains the insertion point
                    g.end += 1
            sent.noise_positions = [p + 1 if p >= pos else p for p in sent.noise_positions] + [pos]


def load_grammar(path: str) -> SyntheticGrammar:
    return SyntheticGrammar.from_yaml(path)
