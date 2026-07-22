#!/usr/bin/env python3
"""
mlx_steer.py — steganographic decoding on Apple Silicon (MLX).

Generate text about a topic while forcing every 5th character to carry one bit
of an input bitstream: the character at position p (p % 5 == 0) must satisfy
(ord(char) & 1) == bits[p // 5]. Any token that would break this gets probability 0.

    pip install mlx-lm
    python mlx_steer.py --topic "the sea" --bits 10110
"""

from __future__ import annotations

import argparse

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

STRIDE = 5


def parse_bits(s: str) -> list[int]:
    return [1 if c == "1" else 0 for c in s if c in "01"]


def char_bit(ch: str) -> int:
    """The bit a character encodes: LSB of its codepoint (even -> 0, odd -> 1)."""
    return ord(ch) & 1


class Stego:
    """MLX logits processor. Zeros the probability of any token that would place
    a wrong-parity character on a position that must carry a bitstream bit."""

    def __init__(self, tok, bits: list[int]):
        self.tok = tok
        self.bits = bits
        self.offset = None      # len of `tokens` before generation starts
        self.tables = None      # built lazily once the vocab size is known

    def _build_tables(self, V: int):
        print(f"[stego] indexing {V} tokens (one-time)...")
        pieces = [self.tok.decode([i]) for i in range(V)]
        parity = np.full((V, STRIDE), -1, dtype=np.int8)   # -1 = no char there
        for i, s in enumerate(pieces):
            for k in range(min(STRIDE, len(s))):
                parity[i, k] = char_bit(s[k])
        long_ids = np.where(np.array([len(s) for s in pieces]) > STRIDE)[0]
        self.tables = (parity, pieces, long_ids)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.offset is None:
            self.offset = int(tokens.shape[0])
        if self.tables is None:
            self._build_tables(logits.shape[1])
        parity, pieces, long_ids = self.tables

        gen = tokens[self.offset:].tolist()
        L = len(self.tok.decode(gen)) if gen else 0     # chars generated so far
        r = (STRIDE - L % STRIDE) % STRIDE              # chars until next constrained pos
        p0 = L + r
        b0 = p0 // STRIDE
        if b0 >= len(self.bits):
            return logits                               # bitstream spent -> free

        vec = np.array(logits.astype(mx.float32)).reshape(-1)
        # Constrain position p0 for every token (a token <= STRIDE chars can hit
        # at most this one). col == -1 means the token is too short to reach it.
        col = parity[:, r]
        vec[(col != -1) & (col != self.bits[b0])] = -np.inf
        # Tokens longer than STRIDE can also cover p0+STRIDE, p0+2*STRIDE, ...
        for i in long_ids:
            s = pieces[i]
            for p in range(p0 + STRIDE, L + len(s), STRIDE):
                b = p // STRIDE
                if b < len(self.bits) and char_bit(s[p - L]) != self.bits[b]:
                    vec[i] = -np.inf
                    break

        if np.isneginf(vec).all():
            return logits                               # never mask everything
        return mx.array(vec.reshape(1, -1))


def extract(text: str) -> list[int]:
    """Recover the hidden bitstream: LSB of every STRIDE-th character."""
    return [char_bit(text[p]) for p in range(0, len(text), STRIDE)]


def verify(text: str, bits: list[int]) -> bool:
    """Read the bits back out of `text` and check they match `bits`."""
    got = extract(text)
    print("\n[stego] verification (every 5th char must encode the bitstream):")
    ok_all = True
    for i, want in enumerate(bits):
        if i >= len(got):
            print(f"  pos {i * STRIDE:>4}  <missing>  want={want}  XX")
            ok_all = False
            continue
        ok = got[i] == want
        ok_all &= ok
        print(f"  pos {i * STRIDE:>4}  {text[i * STRIDE]!r:<6} bit={got[i]} want={want}  {'OK' if ok else 'XX'}")
    print(f"[stego] {'PASS' if ok_all else 'FAIL'} — all {len(bits)} bits encoded correctly: {ok_all}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True, help="what to write about")
    ap.add_argument("--bits", required=True, help="bitstream to hide, e.g. 10110")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bits = parse_bits(args.bits)
    if not bits:
        ap.error("--bits must contain at least one 0 or 1")

    model, tokenizer = load(args.model)
    messages = [{"role": "user", "content": f"Write a few sentences about: {args.topic}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    text = generate(
        model, tokenizer, prompt,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=args.temperature),
        logits_processors=[Stego(tokenizer, bits)],
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print(text)
    ok = verify(text, bits)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
11m

