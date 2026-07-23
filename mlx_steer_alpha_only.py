#!/usr/bin/env python3
"""
mlx_steer_alpha_only.py — steganographic encoding on Apple Silicon (MLX).

Generate text about a topic while hiding a bitstream in every 5th LETTER.
The alphabet is split into three buckets of (near) equal probability in natural
text:

    bucket 0 -> encodes bit 0
    bucket 1 -> encodes bit 1
    bucket 2 -> SKIP: carries no bit at all

At each constrained letter position (letter index j with j % 5 == 0):

  * a letter from the *wrong* bit-bucket is forbidden (probability 0),
  * a letter from the *correct* bit-bucket encodes the next bit,
  * a SKIP letter is *penalized but still allowed* — it consumes no bit, so the
    decoder just moves on to the next constrained letter.

The penalty biases generation toward actually encoding, while letting a
high-probability skip letter win when every correct-bucket alternative is
unnatural. A skip is never *incorrect*; it only defers the bit.

    pip install mlx-lm
    python mlx_steer_alpha_only.py --topic "the sea" --bits 10110
"""

from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import mlx.core as mx
from mlx_lm import load

STRIDE = 5
SKIP = 2                # index of the third (bit-less) bucket
SKIP_PENALTY = 4.0      # logit penalty for placing a skip letter on a slot


def parse_bits(s: str) -> list[int]:
    return [1 if c == "1" else 0 for c in s if c in "01"]


# Relative English letter frequencies (letters only). Used once, at import time,
# to greedily partition the alphabet into three near-equiprobable buckets.
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


def char_bucket(ch: str) -> int:
    """Which bucket a letter falls in: 0 or 1 encode that bit, SKIP carries none.

    Totally defined over any `.isalpha()` character: accented letters are folded
    to their base (é -> e, ñ -> n); letters with no a-z base (non-Latin scripts)
    fall into SKIP so they carry no bit and can never be the wrong bit."""
    c = ch.lower()
    if c in _BUCKET:
        return _BUCKET[c]
    for base in unicodedata.normalize("NFKD", c):
        if base in _BUCKET:
            return _BUCKET[base]
    return SKIP


def letter_buckets(text: str) -> list[int]:
    """Bucket of each letter in `text`, in order (non-letters skipped)."""
    return [char_bucket(c) for c in text if c.isalpha()]


def extract(text: str) -> list[int]:
    """Recover the hidden bitstream: read every STRIDE-th letter, dropping skips."""
    return [b for b in letter_buckets(text)[::STRIDE] if b != SKIP]


class Stego:
    """MLX logits processor. Forbids any token that would place a wrong-bit letter
    on a constrained slot, and penalizes (but permits) tokens that place a skip
    letter there — biasing toward encoding while allowing natural skips."""

    def __init__(self, tok, bits: list[int], skip_penalty: float = SKIP_PENALTY):
        self.tok = tok
        self.bits = bits
        self.skip_penalty = skip_penalty
        self.offset = None      # len of `tokens` before generation starts
        self.tables = None      # built lazily once the vocab size is known
        self.last_m0 = None     # letters until the next constrained slot (-1 = bits spent)

    def _build_tables(self, V: int):
        print(f"[stego] indexing {V} tokens (one-time)...")
        lbuck = [letter_buckets(self.tok.decode([i])) for i in range(V)]
        buckets = np.full((V, STRIDE), -1, dtype=np.int8)   # -1 = no letter there
        for i, bs in enumerate(lbuck):
            for m in range(min(STRIDE, len(bs))):
                buckets[i, m] = bs[m]
        long_ids = np.where(np.array([len(bs) for bs in lbuck]) > STRIDE)[0]
        self.tables = (buckets, lbuck, long_ids)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.offset is None:
            self.offset = int(tokens.shape[0])
        if self.tables is None:
            self._build_tables(logits.shape[1])
        buckets, lbuck, long_ids = self.tables

        gen = tokens[self.offset:].tolist()
        letters = [c for c in self.tok.decode(gen) if c.isalpha()] if gen else []
        nl = len(letters)
        m0 = (STRIDE - nl % STRIDE) % STRIDE     # letters into next token until a constrained one
        # bits consumed so far = non-skip letters already sitting on constrained slots
        consumed = sum(char_bucket(letters[j]) != SKIP for j in range(0, nl, STRIDE))
        if consumed >= len(self.bits):
            self.last_m0 = -1
            return logits                        # bitstream spent -> free
        self.last_m0 = m0                        # 0 == the next letter carries a bit
        b = self.bits[consumed]

        vec = np.array(logits.astype(mx.float32)).reshape(-1)
        # First constrained letter each token could place (residue m0). By bucket:
        #   == b       -> encodes the wanted bit          (allowed, unchanged)
        #   == 1 - b   -> encodes the wrong bit           (forbidden)
        #   == SKIP    -> carries no bit                  (allowed, penalized)
        #   == -1      -> token places no letter here     (see below)
        col = buckets[:, m0]
        vec[(col != -1) & (col != b) & (col != SKIP)] = -np.inf
        vec[col == SKIP] -= self.skip_penalty
        # When the constrained letter is imminent (m0 == 0), a letter-free token
        # (col == -1: '*', whitespace, punctuation, EOS) would defer the bit at
        # zero cost and can loop forever ('****...'). Penalize it exactly like a
        # skip so encoding stays the cheapest move, without forbidding filler.
        if m0 == 0:
            vec[col == -1] -= self.skip_penalty
        # Tokens with more than STRIDE letters can reach further constrained slots;
        # simulate each so a wrong bit anywhere in the token is forbidden. Skips
        # inside the token defer the bit index rather than consuming it.
        for i in long_ids:
            bs = lbuck[i]
            bi = consumed
            for m in range(m0, len(bs), STRIDE):
                bk = bs[m]
                if bk == SKIP:
                    continue
                if bi >= len(self.bits):
                    break
                if bk != self.bits[bi]:
                    vec[i] = -np.inf
                    break
                bi += 1

        if np.isneginf(vec).all():
            return logits                        # never mask everything
        return mx.array(vec.reshape(1, -1))


def _vec(logits) -> np.ndarray:
    """An mx/np logits row -> a flat float32 numpy vector."""
    return np.array(logits.astype(mx.float32)).reshape(-1)


def _masked_vector(stego: "Stego", tokens: list[int], logits) -> np.ndarray:
    """Current-step logits after the Stego constraint, as a numpy vector."""
    return _vec(stego(mx.array(tokens), logits))


def _logsumexp(v: np.ndarray) -> float:
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return -np.inf
    m = float(finite.max())
    return m + float(np.log(np.exp(v - m).sum()))


def _sample_logits(logits: np.ndarray, temp: float, rng, top_p: float = 1.0) -> int:
    """Sample an index from `logits` (may contain -inf) at temperature `temp`,
    optionally restricted to the top-`top_p` nucleus."""
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
    """Discourage degenerate repetition, in place on `vec` (only finite entries):

      * freq_penalty: subtract `freq_penalty` per recent occurrence of a token
        (within the last `window` tokens) — dampens over-used tokens.
      * no_repeat_ngram: strongly demote any token that would repeat an n-gram
        already seen — kills phrase loops ('word word word') without punishing
        ordinary reuse of common words the way a raw frequency penalty does.
    """
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
                    vec[t] -= 20.0        # ~e^20: effectively blocked, but never -inf
    return vec


def steer_generate(model, tokenizer, prompt, bits, *, max_tokens=500,
                   temperature=0.8, skip_penalty=SKIP_PENALTY, num_candidates=5,
                   lookahead_weight=1.0, lookahead_floor=0.05, top_p=0.95,
                   freq_penalty=0.5, rep_window=64, no_repeat_ngram=3,
                   seed=0, verbose=False) -> str:
    """Constrained decoding with anti-repetition sampling + lookahead corner-avoidance.

    Every step: apply the Stego constraint (wrong-bit letters -> -inf), then an
    anti-repetition penalty, then sample the full top-`top_p` nucleus at
    `temperature`. Sampling the *full* constrained distribution is what keeps the
    text fluent — we do NOT restrict to a shortlist off-slot.

    Only at a constrained slot (the letter that actually carries a bit, m0 == 0)
    do we optionally rescore: trial-run the top `num_candidates` valid letters one
    token forward and penalize any that would corner the *next* constrained slot,
    i.e. leave it under `lookahead_floor` of the model's mass on a valid token.
    Above the floor a candidate gets no bonus, so variety stays with the natural
    distribution rather than being pulled toward high model certainty (repetition).

    Cost: one forward pass per token, plus `num_candidates` extra only on the ~1
    in STRIDE steps that carry a bit (and none once the bitstream is spent).
    """
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache
    except ImportError:
        def can_trim_prompt_cache(_cache):     # older mlx_lm: assume trimmable
            return True

    stego = Stego(tokenizer, bits, skip_penalty=skip_penalty)
    rng = np.random.default_rng(seed)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or
                  ([tokenizer.eos_token_id]
                   if getattr(tokenizer, "eos_token_id", None) is not None else []))
    log_floor = float(np.log(lookahead_floor)) if lookahead_floor > 0 else -np.inf

    tokens = list(prompt)
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    stego.offset = len(prompt)      # generated tokens = everything after the prompt

    use_lookahead = lookahead_weight > 0.0 and num_candidates > 1
    if use_lookahead and not can_trim_prompt_cache(cache):
        print("[stego] cache is not trimmable; lookahead disabled")
        use_lookahead = False

    for _ in range(max_tokens):
        masked = _masked_vector(stego, tokens, logits)    # constraint applied (sets last_m0)
        base = _antirepeat(masked.copy(), tokens[len(prompt):], freq_penalty=freq_penalty,
                           window=rep_window, no_repeat_ngram=no_repeat_ngram)

        if use_lookahead and stego.last_m0 == 0:          # only at a bit-carrying slot
            valid = np.where(np.isfinite(base))[0]
            k = min(num_candidates, valid.size)
            cand = valid[np.argsort(base[valid])[-k:]]    # highest-logit valid letters
            combined = np.empty(k)
            for j, c in enumerate(cand):
                c = int(c)
                nlogits = model(mx.array([[c]]), cache=cache)[:, -1, :]
                mx.eval(nlogits)
                # log-fraction of the model's mass left on pattern-valid next
                # tokens; floored so only genuinely cornered candidates lose out.
                look = min(0.0, (_logsumexp(_masked_vector(stego, tokens + [c], nlogits))
                                 - _logsumexp(_vec(nlogits))) - log_floor)
                combined[j] = base[c] + lookahead_weight * look
                trim_prompt_cache(cache, 1)               # undo the trial token
            choice = int(cand[_sample_logits(combined, temperature, rng, top_p)])
        else:
            choice = _sample_logits(base, temperature, rng, top_p)

        tokens.append(choice)
        if choice in eos_ids:
            break
        logits = model(mx.array([[choice]]), cache=cache)[:, -1, :]
        mx.eval(logits)
        if verbose:
            print(tokenizer.decode([choice]), end="", flush=True)

    if verbose:
        print()
    return tokenizer.decode(tokens[len(prompt):])


def verify(text: str, bits: list[int]) -> bool:
    """Read the bits back out of `text` and check they match `bits`."""
    letters = [c for c in text if c.isalpha()]
    slot_chars = letters[::STRIDE]
    slot_buckets = [char_bucket(c) for c in slot_chars]
    print("\n[stego] verification (every 5th letter; SKIP-bucket letters carry no bit):")
    ok_all = True
    bi = 0
    for k, bk in enumerate(slot_buckets):
        ch = slot_chars[k]
        j = k * STRIDE
        if bk == SKIP:
            print(f"  letter {j:>4}  {ch!r:<6} bucket=SKIP  (skipped)")
            continue
        if bi >= len(bits):
            print(f"  letter {j:>4}  {ch!r:<6} bit={bk}  (past end of bitstream)")
            continue
        ok = bk == bits[bi]
        ok_all &= ok
        print(f"  letter {j:>4}  {ch!r:<6} bit={bk} want={bits[bi]}  {'OK' if ok else 'XX'}")
        bi += 1
    for r in range(bi, len(bits)):
        print(f"  bit {r:>4}  <missing>  want={bits[r]}  XX")
        ok_all = False
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
    ap.add_argument("--skip-penalty", type=float, default=SKIP_PENALTY,
                    help="logit penalty for placing a skip letter on a slot")
    ap.add_argument("--candidates", type=int, default=5,
                    help="pattern-valid tokens to trial-run per step (1 disables lookahead)")
    ap.add_argument("--lookahead-weight", type=float, default=1.0,
                    help="weight of the lookahead corner-avoidance score (0 disables)")
    ap.add_argument("--lookahead-floor", type=float, default=0.05,
                    help="only penalize a candidate if it leaves <this fraction of model mass valid")
    ap.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling cutoff")
    ap.add_argument("--freq-penalty", type=float, default=0.5,
                    help="logit penalty per recent occurrence of a token (anti-repetition)")
    ap.add_argument("--rep-window", type=int, default=64,
                    help="how many recent tokens the frequency penalty looks back over")
    ap.add_argument("--no-repeat-ngram", type=int, default=3,
                    help="block repeating any n-gram of this size (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bits = parse_bits(args.bits)
    if not bits:
        ap.error("--bits must contain at least one 0 or 1")

    model, tokenizer = load(args.model)
    #messages = [{"role": "user", "content": f"Write a few sentences about: {args.topic}"}]
    messages = [{"role": "user", "content": f"Please write a post-modern high literarture short story about an orange boy cat breaking up with a blonde haired boy ninja."}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    text = steer_generate(
        model, tokenizer, prompt, bits,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        skip_penalty=args.skip_penalty,
        num_candidates=args.candidates,
        lookahead_weight=args.lookahead_weight,
        lookahead_floor=args.lookahead_floor,
        top_p=args.top_p,
        freq_penalty=args.freq_penalty,
        rep_window=args.rep_window,
        no_repeat_ngram=args.no_repeat_ngram,
        seed=args.seed,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print(text)
    ok = verify(text, bits)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()