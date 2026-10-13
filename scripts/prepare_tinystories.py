"""Download a prefix of TinyStories (raw text only) and split it into sentences.

    python scripts/prepare_tinystories.py --mb 120 --out data/tinystories_sentences.txt

Only the raw story text is used: no annotation of any kind.  The script
streams the first ``--mb`` megabytes of ``TinyStories-train.txt`` from the
Hugging Face hub (HTTP range request), splits stories into sentences with a
simple punctuation rule, and writes one sentence per line.  Length filtering
and tokenisation happen later in the pipeline (``preprocess`` config).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def download_prefix(url: str, n_bytes: int, path: str) -> None:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}", "User-Agent": "abc-induction/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            if total >= n_bytes:
                break
    print(f"downloaded {total / 1e6:.1f} MB to {path}")


def split_sentences(raw_path: str, out_path: str, max_sentences: int | None) -> int:
    n = 0
    with open(raw_path, encoding="utf-8", errors="ignore") as fh, open(out_path, "w", encoding="utf-8") as out:
        buf: list[str] = []
        for line in fh:
            line = line.strip()
            if not line or line == "<|endoftext|>":
                if buf:
                    text = " ".join(buf)
                    for sent in _SENT_END.split(text):
                        sent = sent.strip()
                        if sent:
                            out.write(sent + "\n")
                            n += 1
                            if max_sentences and n >= max_sentences:
                                return n
                    buf = []
                continue
            buf.append(line)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=120.0, help="megabytes of raw text to download")
    ap.add_argument("--out", default=os.path.join("data", "tinystories_sentences.txt"))
    ap.add_argument("--raw", default=os.path.join("data", "tinystories_raw_prefix.txt"))
    ap.add_argument("--max-sentences", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.exists(args.raw) or os.path.getsize(args.raw) < args.mb * 1e6 * 0.95:
        download_prefix(URL, int(args.mb * 1e6), args.raw)
    else:
        print(f"using cached {args.raw}")
    n = split_sentences(args.raw, args.out, args.max_sentences)
    print(f"wrote {n} sentences to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
