#!/usr/bin/env python3
"""
simple_encode.py — four-bucket steganographic encoding, NO STRIDE.

Every character is a potential encoding slot. At each character position the next
character either encodes the next bit or skips it. Characters split into four
buckets (same structure as the four-bucket build):

    bucket 0        -> alpha, encodes bit 0
    bucket 1        -> alpha, encodes bit 1
    bucket 2 (SKIP)      -> alpha, off-bucket: carries no bit, *penalized*
    bucket 3 (SKIP_FREE) -> non-alpha (space/punct/...): carries no bit, *free*

Because there is no stride, a single token covers as many slots as it has
characters, and may encode several bits at once. A token is allowed only if
*every* one of its characters that lands in a bit-bucket matches the bit it would
carry there (skips defer, they never consume). This is checked for the whole
vocabulary at once (vectorized) each step.

Density is driven by *search*, not by brute penalties: at each step we shortlist
the most promising valid tokens (probability + a bonus for how many bits they
encode), trial-run each one token forward, and score it by whether it keeps the
model in a fluent, un-cornered state. The best-scoring candidate is sampled. This
is what lets a maximal-density (stride-1) constraint still produce plausible text.

    pip install mlx-lm
    python simple_encode.py --topic "the sea" --bits 10110
"""

from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import mlx.core as mx
from mlx_lm import load

SKIP = 2                # penalized bit-less bucket (off-bucket alpha)
SKIP_FREE = 3           # free bit-less bucket (non-alpha: space / punctuation / ...)
BIT_BUCKETS = (0, 1)    # buckets that actually carry a bit

# Relative English letter frequencies (letters only), used once at import time to
# greedily split the alphabet into three near-equiprobable buckets (0, 1, SKIP).
_FREQ = {
    "e": 12.70, "t": 9.06, "a": 8.17, "o": 7.51, "i": 6.97, "n": 6.75,
    "s": 6.33, "h": 6.09, "r": 5.99, "d": 4.25, "l": 4.03, "c": 2.78,
    "u": 2.76, "m": 2.41, "w": 2.36, "f": 2.23, "g": 2.02, "y": 1.97,
    "p": 1.93, "b": 1.49, "v": 0.98, "k": 0.77, "j": 0.15, "x": 0.15,
    "q": 0.10, "z": 0.07,
}


def _make_buckets(freq: dict[str, float], n: int = 3) -> dict[str, int]:
    """Greedily assign each letter to the currently-lightest bucket, so all `n`
    buckets end up with near-equal total frequency in natural text."""
    sums = [0.0] * n
    members: dict[str, int] = {}
    for ch in sorted(freq, key=lambda c: (-freq[c], c)):
        k = min(range(n), key=lambda k: sums[k])
        sums[k] += freq[ch]
        members[ch] = k
    return members


_BUCKET = _make_buckets(_FREQ)


def parse_bits(s: str) -> list[int]:
    return [1 if c == "1" else 0 for c in s if c in "01"]


def char_bucket(ch: str) -> int:
    """Which bucket a character falls in. 0/1 carry a bit; SKIP and SKIP_FREE
    carry none. Non-alpha -> SKIP_FREE (free). Alpha folds to its a-z base
    (é -> e); a letter with no a-z base -> SKIP (a penalized, not free, skip)."""
    if not ch.isalpha():
        return SKIP_FREE
    c = ch.lower()
    if c in _BUCKET:
        return _BUCKET[c]
    for base in unicodedata.normalize("NFKD", c):
        if base in _BUCKET:
            return _BUCKET[base]
    return SKIP


def char_buckets(text: str) -> list[int]:
    """Bucket of every character in `text`, in order."""
    return [char_bucket(c) for c in text]


def extract(text: str) -> list[int]:
    """Recover the hidden bitstream: every bit-bucket character, in order,
    dropping both flavors of skip."""
    return [b for b in char_buckets(text) if b in BIT_BUCKETS]


def verify(text: str, bits: list[int]) -> bool:
    """Check the payload reads back out of `text`."""
    got = extract(text)
    ok = got[:len(bits)] == bits
    print(f"\n[stego] recovered {min(len(got), len(bits))}/{len(bits)} payload bits"
          f"{'' if len(got) <= len(bits) else f' (+{len(got) - len(bits)} free tail bits)'}")
    if not ok:
        for i, want in enumerate(bits):
            if i >= len(got):
                print(f"  bit {i:>4}  <missing>  want={want}  XX")
            elif got[i] != want:
                print(f"  bit {i:>4}  got={got[i]} want={want}  XX")
    print(f"[stego] {'PASS' if ok else 'FAIL'} — all {len(bits)} bits encoded: {ok}")
    return ok


class Encoder:
    """Vectorized stride-1 four-bucket constraint over the whole vocabulary.

    Precomputes, per token, the sequence of bit-bucket values its characters
    would encode (`ENC`), its length (`enc_len`), and how many *penalized* skip
    characters it contains (`nskip`). Given how many payload bits are already
    consumed, `constrain` forbids any token whose encoded bits would disagree
    with the upcoming payload and penalizes penalized-skip characters."""

    def __init__(self, tok, bits: list[int], skip_penalty: float):
        self.tok = tok
        self.bits = np.array(bits, dtype=np.int8)
        self.nbits = len(bits)
        self.skip_penalty = skip_penalty
        self.tables = None

    def _build(self, V: int):
        print(f"[stego] indexing {V} tokens (one-time)...")
        rows: list[list[int]] = []
        nskip = np.zeros(V, dtype=np.float32)
        enc_len = np.zeros(V, dtype=np.int32)
        max_e = 1
        for i in range(V):
            bs = char_buckets(self.tok.decode([i]))
            enc = [x for x in bs if x in BIT_BUCKETS]
            rows.append(enc)
            enc_len[i] = len(enc)
            nskip[i] = sum(1 for x in bs if x == SKIP)
            if len(enc) > max_e:
                max_e = len(enc)
        ENC = np.full((V, max_e), -1, dtype=np.int8)   # -1 = no encoding char here
        for i, enc in enumerate(rows):
            if enc:
                ENC[i, :len(enc)] = enc
        self.tables = (ENC, enc_len, nskip, max_e)

    def bits_of(self, token: int, consumed: int) -> int:
        """How many payload bits token `token` would encode starting at `consumed`."""
        _, enc_len, _, _ = self.tables
        return int(min(int(enc_len[token]), max(0, self.nbits - consumed)))

    def constrain(self, consumed: int, vec: np.ndarray):
        """Return (masked_logits, bits_encoded_per_token) at payload position
        `consumed`. Forbidden tokens -> -inf; penalized-skip chars subtract
        `skip_penalty` each. `bits_encoded` is 0 for forbidden tokens."""
        ENC, enc_len, nskip, E = self.tables
        rem = self.nbits - consumed
        if rem <= 0:                                    # payload spent -> unconstrained
            return vec.copy(), np.zeros(vec.shape[0], dtype=np.int32)
        kk = np.arange(E)
        valid = kk < rem                                # slots still inside the payload
        m = min(E, rem)
        target = np.full(E, -1, dtype=np.int8)
        target[:m] = self.bits[consumed:consumed + m]
        # forbidden: any encoding char that lands on a payload slot with wrong bit
        bad = ((ENC != -1) & valid[None, :] & (ENC != target[None, :])).any(axis=1)
        out = vec - self.skip_penalty * nskip           # penalize off-bucket-letter skips
        out[bad] = -np.inf
        bits_enc = np.minimum(enc_len, rem).astype(np.int32)
        bits_enc[bad] = 0
        return out, bits_enc


# --------------------------------------------------------------------------- #
# sampling / scoring helpers
# --------------------------------------------------------------------------- #
def _vec(logits) -> np.ndarray:
    return np.array(logits.astype(mx.float32)).reshape(-1)


def _logsumexp(v: np.ndarray) -> float:
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return -np.inf
    m = float(finite.max())
    return m + float(np.log(np.exp(v - m).sum()))


def _sample_logits(logits: np.ndarray, temp: float, rng, top_p: float = 1.0) -> int:
    """Sample an index from `logits` (may contain -inf) at temperature `temp`,
    restricted to the top-`top_p` nucleus."""
    if temp <= 1e-6:
        return int(np.argmax(logits))
    z = logits / temp
    z = z - _logsumexp(z)
    p = np.exp(z)
    p[~np.isfinite(logits)] = 0.0
    s = p.sum()
    if s <= 0:
        return int(np.argmax(logits))
    p = p / s
    if top_p < 1.0:
        order = np.argsort(p)[::-1]
        cut = int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1
        keep = np.zeros_like(p, dtype=bool)
        keep[order[:cut]] = True
        p = np.where(keep, p, 0.0)
        p /= p.sum()
    return int(rng.choice(p.size, p=p))


def _antirepeat(vec: np.ndarray, gen_ids: list[int], *, freq_penalty: float,
                window: int, no_repeat_ngram: int) -> np.ndarray:
    """Discourage degenerate repetition, in place on `vec` (finite entries only)."""
    if freq_penalty > 0 and gen_ids:
        from collections import Counter
        for t, ct in Counter(gen_ids[-window:]).items():
            if np.isfinite(vec[t]):
                vec[t] -= freq_penalty * ct
    n = no_repeat_ngram
    if n and n >= 1 and len(gen_ids) >= n - 1:
        prefix = tuple(gen_ids[-(n - 1):]) if n > 1 else ()
        for i in range(len(gen_ids) - n + 1):
            if tuple(gen_ids[i:i + n - 1]) == prefix:
                t = gen_ids[i + n - 1]
                if np.isfinite(vec[t]):
                    vec[t] -= 20.0
    return vec


def _lookahead(enc: Encoder, model, cache, consumed_after: int, nlogits, *, depth: int,
               log_floor: float, eos_ids: set, trim) -> float:
    """Score one already-advanced candidate by how un-cornered it leaves the model.

    Sums a floored 'valid-mass' term over the step right after the candidate plus
    `depth` further greedy constrained steps. Terms are <= 0, so a candidate that
    keeps plenty of natural mass on constraint-valid tokens scores 0, while one
    that paints the model into a corner (little valid mass, i.e. bad for both
    fluency and future density) is penalized. Cache is restored to the
    tokens+[candidate] state; the caller undoes the candidate itself."""
    rawn = _vec(nlogits)
    masked, _ = enc.constrain(consumed_after, rawn)
    total = min(0.0, (_logsumexp(masked) - _logsumexp(rawn)) - log_floor)
    cur_masked, cur_consumed, advanced = masked, consumed_after, 0
    for _ in range(depth):
        if cur_consumed >= enc.nbits:
            break
        nxt = int(np.argmax(cur_masked))
        clog = model(mx.array([[nxt]]), cache=cache)[:, -1, :]
        mx.eval(clog)
        advanced += 1
        if nxt in eos_ids:
            break
        cur_consumed += enc.bits_of(nxt, cur_consumed)
        rawn = _vec(clog)
        cur_masked, _ = enc.constrain(cur_consumed, rawn)
        total += min(0.0, (_logsumexp(cur_masked) - _logsumexp(rawn)) - log_floor)
    if advanced:
        trim(cache, advanced)
    return total


def steer_generate(model, tokenizer, prompt, bits, *, max_tokens=800, temperature=0.8,
                   skip_penalty=1.0, num_candidates=16, lookahead_weight=8.0,
                   lookahead_floor=0.05, bit_bonus=3.0, rollout_depth=0, top_p=0.95,
                   freq_penalty=0.5, rep_window=64, no_repeat_ngram=3,
                   seed=0, verbose=False) -> str:
    """Stride-1 constrained decoding with density-aware candidate search.

    Every character is a slot, so every step (until the payload is spent) is a
    constrained encoding step. Each such step:

      1. Constrain the whole vocab (vectorized) and block EOS while bits remain.
      2. Shortlist the top `num_candidates` valid tokens by  logit + bit_bonus *
         (bits it encodes)  — so the trial set is biased toward real density.
      3. Trial-run each shortlisted token one forward pass; score it with
         `_lookahead` (does it corner the model?) plus its own density bonus.
      4. Sample a winner from the combined scores at `temperature`/`top_p`.

    Once the payload is fully encoded, generation continues unconstrained so the
    text can finish naturally.
    """
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache
    except ImportError:
        def can_trim_prompt_cache(_cache):
            return True

    enc = Encoder(tokenizer, bits, skip_penalty)
    rng = np.random.default_rng(seed)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or
                  ([tokenizer.eos_token_id]
                   if getattr(tokenizer, "eos_token_id", None) is not None else []))
    log_floor = float(np.log(lookahead_floor)) if lookahead_floor > 0 else -np.inf

    tokens = list(prompt)
    offset = len(prompt)
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    enc._build(logits.shape[1])                          # vocab size known now
    use_lookahead = lookahead_weight > 0.0 and num_candidates > 1 and can_trim_prompt_cache(cache)

    consumed = 0
    for _ in range(max_tokens):
        raw = _vec(logits)

        if consumed >= enc.nbits:                        # payload done -> free finish
            base = _antirepeat(raw.copy(), tokens[offset:], freq_penalty=freq_penalty,
                               window=rep_window, no_repeat_ngram=no_repeat_ngram)
            choice = _sample_logits(base, temperature, rng, top_p)
        else:
            masked, bits_enc = enc.constrain(consumed, raw)
            for e in eos_ids:                            # never stop mid-payload
                masked[e] = -np.inf
            base = _antirepeat(masked.copy(), tokens[offset:], freq_penalty=freq_penalty,
                               window=rep_window, no_repeat_ngram=no_repeat_ngram)
            score = base + bit_bonus * bits_enc          # density-aware ranking
            if not use_lookahead:
                choice = _sample_logits(score, temperature, rng, top_p)
            else:
                valid = np.where(np.isfinite(base))[0]
                k = min(num_candidates, valid.size)
                cand = valid[np.argsort(score[valid])[-k:]]  # trial the most promising
                combined = np.empty(k)
                for j, c in enumerate(cand):
                    c = int(c)
                    nlogits = model(mx.array([[c]]), cache=cache)[:, -1, :]
                    mx.eval(nlogits)
                    look = _lookahead(enc, model, cache, consumed + int(bits_enc[c]), nlogits,
                                      depth=rollout_depth, log_floor=log_floor,
                                      eos_ids=eos_ids, trim=trim_prompt_cache)
                    combined[j] = base[c] + bit_bonus * bits_enc[c] + lookahead_weight * look
                    trim_prompt_cache(cache, 1)          # undo the trial token
                choice = int(cand[_sample_logits(combined, temperature, rng, top_p)])
            consumed += int(bits_enc[choice])

        tokens.append(choice)
        if choice in eos_ids:                            # only reachable once payload is done
            break
        logits = model(mx.array([[choice]]), cache=cache)[:, -1, :]
        mx.eval(logits)
        if verbose:
            print(tokenizer.decode([choice]), end="", flush=True)

    if verbose:
        print()
    return tokenizer.decode(tokens[offset:])


def encoding_summary(text: str, bits: list[int]) -> float:
    """Print encoding density: hidden bits per character of cover text."""
    encoded = min(len(bits), len(extract(text)))
    bpc = encoded / len(text) if text else 0.0
    print(f"[stego] density: {encoded} bits / {len(text)} chars = {bpc:.4f} bits/char")
    return bpc


def main() -> None:
    # --- fixed tuning knobs (edit here; kept off the CLI) ---
    LOOKAHEAD_FLOOR = 0.05   # penalize a candidate only below this valid-mass fraction
    TOP_P           = 0.95   # nucleus sampling cutoff
    FREQ_PENALTY    = 0.5    # anti-repetition: penalty per recent token occurrence
    REP_WINDOW      = 64     # anti-repetition: recent-token window
    NO_REPEAT_NGRAM = 3      # anti-repetition: block repeating any n-gram of this size

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True, help="what the cover text is about")
    ap.add_argument("--bits", required=True, help="bitstream to hide, e.g. 10110")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--skip-penalty", type=float, default=1.0,
                    help="logit penalty per off-bucket-letter (SKIP) character")
    ap.add_argument("--candidates", type=int, default=16,
                    help="valid tokens trial-run per step (higher = more density, slower)")
    ap.add_argument("--lookahead-weight", type=float, default=8.0,
                    help="weight of the lookahead corner-avoidance score (0 disables)")
    ap.add_argument("--bit-bonus", type=float, default=3.0,
                    help="score reward per bit a candidate encodes (raises density)")
    ap.add_argument("--rollout-depth", type=int, default=0,
                    help="extra greedy constrained steps to look past each candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bits = parse_bits(args.bits)
    if not bits:
        ap.error("--bits must contain at least one 0 or 1")

    model, tokenizer = load(args.model)
    messages = [{"role": "user", "content": f"Write a short story about: {args.topic}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    text = steer_generate(
        model, tokenizer, prompt, bits,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        skip_penalty=args.skip_penalty,
        num_candidates=args.candidates,
        lookahead_weight=args.lookahead_weight,
        lookahead_floor=LOOKAHEAD_FLOOR,
        bit_bonus=args.bit_bonus,
        rollout_depth=args.rollout_depth,
        top_p=TOP_P,
        freq_penalty=FREQ_PENALTY,
        rep_window=REP_WINDOW,
        no_repeat_ngram=NO_REPEAT_NGRAM,
        seed=args.seed,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print(text)
    ok = verify(text, bits)
    encoding_summary(text, bits)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
