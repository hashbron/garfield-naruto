#!/usr/bin/env python3
"""
mlx_steer.py — steganographic decoding on Apple Silicon (MLX).

Generate text about a topic while forcing every 5th LETTER to carry one bit of
an input bitstream: the j-th letter with j % 5 == 0 must satisfy
(ord(letter) & 1) == bits[j // 5]. Non-letters (spaces, punctuation, digits) are
skipped and never constrained. Any token that would break this gets probability 0.

    pip install mlx-lm
    python mlx_steer.py --topic "the sea" --bits 10110
"""

from __future__ import annotations

import argparse

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

STRIDE = 10


def parse_bits(s: str) -> list[int]:
    return [1 if c == "1" else 0 for c in s if c in "01"]


def char_bit(ch: str) -> int:
    """The bit a character encodes: LSB of its codepoint (even -> 0, odd -> 1)."""
    return ord(ch) & 1


def letter_bits(text: str) -> list[int]:
    """Parity of each letter in `text`, in order (non-letters skipped)."""
    return [char_bit(c) for c in text if c.isalpha()]


def extract(text: str) -> list[int]:
    """Recover the hidden bitstream: the parity of every STRIDE-th letter."""
    return letter_bits(text)[::STRIDE]


class Stego:
    """MLX logits processor. Zeros the probability of any token that would place
    a wrong-parity letter on a letter-position that must carry a bitstream bit."""

    def __init__(self, tok, bits: list[int]):
        self.tok = tok
        self.bits = bits
        self.offset = None      # len of `tokens` before generation starts
        self.tables = None      # built lazily once the vocab size is known

    def _build_tables(self, V: int):
        print(f"[stego] indexing {V} tokens (one-time)...")
        lpar = [letter_bits(self.tok.decode([i])) for i in range(V)]
        parity = np.full((V, STRIDE), -1, dtype=np.int8)   # -1 = no letter there
        for i, ps in enumerate(lpar):
            for m in range(min(STRIDE, len(ps))):
                parity[i, m] = ps[m]
        long_ids = np.where(np.array([len(ps) for ps in lpar]) > STRIDE)[0]
        self.tables = (parity, lpar, long_ids)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.offset is None:
            self.offset = int(tokens.shape[0])
        if self.tables is None:
            self._build_tables(logits.shape[1])
        parity, lpar, long_ids = self.tables

        gen = tokens[self.offset:].tolist()
        nl = sum(c.isalpha() for c in self.tok.decode(gen)) if gen else 0  # letters so far
        m0 = (STRIDE - nl % STRIDE) % STRIDE     # letters into next token until a constrained one
        b0 = (nl + m0) // STRIDE
        if b0 >= len(self.bits):
            return logits                        # bitstream spent -> free

        vec = np.array(logits.astype(mx.float32)).reshape(-1)
        # Constrain the first constrained letter for every token (a token with
        # <= STRIDE letters can reach at most this one). -1 means it can't.
        col = parity[:, m0]
        vec[(col != -1) & (col != self.bits[b0])] = -np.inf
        # Tokens with more than STRIDE letters can reach further constrained letters.
        for i in long_ids:
            ps = lpar[i]
            for m in range(m0 + STRIDE, len(ps), STRIDE):
                b = b0 + (m - m0) // STRIDE
                if b < len(self.bits) and ps[m] != self.bits[b]:
                    vec[i] = -np.inf
                    break

        if np.isneginf(vec).all():
            return logits                        # never mask everything
        return mx.array(vec.reshape(1, -1))


def verify(text: str, bits: list[int]) -> bool:
    """Read the bits back out of `text` and check they match `bits`."""
    letters = [c for c in text if c.isalpha()]
    print("\n[stego] verification (every 5th letter must encode the bitstream):")
    ok_all = True
    for i, want in enumerate(bits):
        j = i * STRIDE
        if j >= len(letters):
            print(f"  letter {j:>4}  <missing>  want={want}  XX")
            ok_all = False
            continue
        c = letters[j]
        ok = char_bit(c) == want
        ok_all &= ok
        print(f"  letter {j:>4}  {c!r:<6} bit={char_bit(c)} want={want}  {'OK' if ok else 'XX'}")
    print(f"[stego] {'PASS' if ok_all else 'FAIL'} — all {len(bits)} bits encoded correctly: {ok_all}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True, help="what to write about")
    ap.add_argument("--bits", required=True, help="bitstream to hide, e.g. 10110")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bits = parse_bits(args.bits)
    if not bits:
        ap.error("--bits must contain at least one 0 or 1")

    model, tokenizer = load(args.model)
    #messages = [{"role": "user", "content": f"Write a few sentences about: {args.topic}"}]
    messages = [{"role": "user", "content": f"Please write a post-modern high literarture short story about an orange cat breaking up with a blonde haired boy ninja."}]
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
