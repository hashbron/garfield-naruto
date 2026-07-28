#!/usr/bin/env python3
"""
mlx_steer_alpha_only.py — steganographic encoding on Apple Silicon (MLX).

Generate text about a topic while hiding a bitstream in every STRIDE-th
CHARACTER (letters, punctuation, and whitespace all count). Characters are
split into four buckets:

    bucket 0        -> alpha, encodes bit 0
    bucket 1        -> alpha, encodes bit 1
    bucket 2 (SKIP)      -> alpha, off-bucket: carries no bit, *penalized*
    bucket 3 (SKIP_FREE) -> non-alpha (punctuation/whitespace/...): carries no
                             bit, *never penalized*

The alpha buckets 0/1/2 are a near-equiprobable split of English letter
frequency, same as before; bucket 3 catches everything that isn't a letter.

At each constrained character position (index j with j % STRIDE == 0):

  * a character from the *wrong* bit-bucket is forbidden (probability 0),
  * a character from the *correct* bit-bucket encodes the next bit,
  * a SKIP letter is *penalized but still allowed* — it consumes no bit, so the
    decoder just moves on to the next constrained slot,
  * a SKIP_FREE character (space, comma, period, ...) is allowed at *no cost* —
    ordinary punctuation and whitespace never fight the constraint.

The SKIP penalty biases generation toward actually encoding, while letting a
high-probability skip letter win when every correct-bucket alternative is
unnatural, and letting natural punctuation/whitespace fall wherever it wants.
A skip is never *incorrect*; it only defers the bit.

    pip install mlx-lm
    python mlx_steer_alpha_only.py --topic "the sea" --bits 10110
"""

from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import mlx.core as mx
from mlx_lm import load

STRIDE = 1              # every STRIDE-th character is a constrained slot; 1 is
                         # brutally restrictive (forces almost every letter of
                         # every word to comply) and tends to collapse into
                         # incoherent text — 5 gives the model room to breathe
SKIP = 2                # index of the penalized bit-less bucket (off-bucket alpha)
SKIP_FREE = 3           # index of the free bit-less bucket (non-alpha: punct/whitespace/...)
SKIP_PENALTY = 5.0      # logit penalty for placing a *penalized* skip character on a slot


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


zero_set = {'c', 'e', 'y', 'w', 'r', 'k', 'n', 'A', 'O', 'C', 'D', 'F', 'G'}
one_set = {'i', 'l', 'b', 'h', 'u', 'v', 't', 'g', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'E', 'B', 'P'}
two_set = {'z', 'j', 'a', 'q', 'p', 'o', 'x', 's', 'f', 'm', 'd', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '|', '\\', '/', '_', '*', '\'', '\n', '\t'}

bucket_dict = {}

for char in zero_set:
    bucket_dict[char] = 0
for char in one_set:
    bucket_dict[char] = 1
for char in two_set:
    bucket_dict[char] = SKIP


_BUCKET = bucket_dict #_make_buckets(_FREQ)


def char_bucket(ch: str) -> int:
##    """Which bucket a character falls in: 0 or 1 encode that bit; SKIP and
##    SKIP_FREE both carry none.
##
##    Non-alpha characters (punctuation, whitespace, digits, ...) always land in
##    SKIP_FREE. Alpha characters are folded to their base (é -> e, ñ -> n);
##    letters with no a-z base (non-Latin scripts) fall into SKIP, so they carry
##    no bit and can never be the wrong bit — just a (penalized) skip rather than
##    a free one, since they're still "letters" rather than punctuation."""
##    if not ch.isalpha():
##        return SKIP_FREE
##    c = ch.lower()
##    if c in _BUCKET:
##        return _BUCKET[c]
##    for base in unicodedata.normalize("NFKD", c):
##        if base in _BUCKET:
##            return _BUCKET[base]
##    return SKIP
    
    if ch in _BUCKET:
        return _BUCKET[ch]
    for base in unicodedata.normalize("NFKD", ch):
        if base in _BUCKET:
            return _BUCKET[base]
    return SKIP_FREE
    


def char_buckets(text: str) -> list[int]:
    """Bucket of each character in `text`, in order. Every character counts now
    (not just letters) — punctuation and whitespace land in SKIP_FREE."""
    return [char_bucket(c) for c in text]


def extract(text: str) -> list[int]:
    """Recover the hidden bitstream: read every STRIDE-th character, dropping
    both flavors of skip."""
    return [b for b in char_buckets(text)[::STRIDE] if b not in (SKIP, SKIP_FREE)]


class Stego:
    """MLX logits processor. Forbids any token that would place a wrong-bit
    character on a constrained slot; penalizes (but permits) tokens that place
    an off-bucket letter (SKIP) there; and fully permits, at no cost, tokens
    that place punctuation/whitespace (SKIP_FREE) there — biasing toward
    encoding while letting natural skips and ordinary prose punctuation
    through."""

    def __init__(self, tok, bits: list[int], skip_penalty: float = SKIP_PENALTY):
        self.tok = tok
        self.bits = bits
        self.skip_penalty = skip_penalty
        self.offset = None      # len of `tokens` before generation starts
        self.tables = None      # built lazily once the vocab size is known
        self.last_m0 = None     # letters until the next constrained slot (-1 = bits spent)

    def _build_tables(self, V: int):
        print(f"[stego] indexing {V} tokens (one-time)...")
        cbuck = [char_buckets(self.tok.decode([i])) for i in range(V)]
        buckets = np.full((V, STRIDE), -1, dtype=np.int8)   # -1 = no character there
        for i, bs in enumerate(cbuck):
            for m in range(min(STRIDE, len(bs))):
                buckets[i, m] = bs[m]
        long_ids = np.where(np.array([len(bs) for bs in cbuck]) > STRIDE)[0]
        self.tables = (buckets, cbuck, long_ids)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.offset is None:
            self.offset = int(tokens.shape[0])
        if self.tables is None:
            self._build_tables(logits.shape[1])
        buckets, cbuck, long_ids = self.tables

        gen = tokens[self.offset:].tolist()
        chars = list(self.tok.decode(gen)) if gen else []
        nl = len(chars)
        m0 = (STRIDE - nl % STRIDE) % STRIDE     # chars into next token until a constrained one
        # bits consumed so far = non-skip characters already sitting on constrained slots
        consumed = sum(char_bucket(chars[j]) not in (SKIP, SKIP_FREE) for j in range(0, nl, STRIDE))
        if consumed >= len(self.bits):
            self.last_m0 = -1
            return logits                        # bitstream spent -> free
        self.last_m0 = m0                        # 0 == the next character carries a bit
        b = self.bits[consumed]
        wrong = 1 - b

        vec = np.array(logits.astype(mx.float32)).reshape(-1)
        # First constrained character each token could place (residue m0). By bucket:
        #   == b         -> encodes the wanted bit           (allowed, unchanged)
        #   == wrong     -> encodes the wrong bit             (forbidden)
        #   == SKIP      -> off-bucket letter, no bit         (allowed, penalized)
        #   == SKIP_FREE -> punctuation/whitespace, no bit    (allowed, free)
        #   == -1        -> token places no character here    (see below)
        col = buckets[:, m0]
        vec[col == wrong] = -np.inf
        vec[col == SKIP] -= self.skip_penalty
        # When the constrained slot is imminent (m0 == 0), a token that places
        # nothing there at all (col == -1, e.g. certain special/control tokens)
        # would defer the bit at zero cost. Penalize it like a paid skip so
        # encoding — or landing a free punctuation/whitespace skip — stays
        # cheaper than stalling. Ordinary punctuation/whitespace tokens are
        # SKIP_FREE (col == 3) and are untouched by this.

        #if m0 == 0:
        #    vec[col == -1] -= 5.0
        # CHANGED THIS^
        
        # Tokens with more than STRIDE characters can reach further constrained
        # slots; simulate each so a wrong bit anywhere in the token is forbidden.
        # Either flavor of skip inside the token defers the bit index rather
        # than consuming it.
        for i in long_ids:
            bs = cbuck[i]
            bi = consumed
            for m in range(m0, len(bs), STRIDE):
                bk = bs[m]
                if bk == SKIP or bk == SKIP_FREE:
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


def steer_generate(model, tokenizer, prompt, bits, *, max_tokens=2000,
                   temperature=0.8, skip_penalty=SKIP_PENALTY, num_candidates=10,
                   lookahead_weight=5.0, lookahead_floor=0.05, top_p=0.95,
                   freq_penalty=0.5, rep_window=64, no_repeat_ngram=3,
                   seed=0, verbose=False, disable_constraint=False) -> str:
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

    `disable_constraint=True` skips the Stego mask entirely (anti-repeat still
    applies) — a debugging knob to check the model is fluent on its own, with
    everything else in the pipeline held constant.
    """
    print(f"[stego] STRIDE={STRIDE}  bits={''.join(map(str, bits))} "
          f"({len(bits)} bits)  disable_constraint={disable_constraint}")
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

    use_lookahead = lookahead_weight > 0.0 and num_candidates > 1 and not disable_constraint
    if use_lookahead and not can_trim_prompt_cache(cache):
        print("[stego] cache is not trimmable; lookahead disabled")
        use_lookahead = False

    for _ in range(max_tokens):
        if disable_constraint:
            masked = _vec(logits)
        else:
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
    slot_chars = list(text)[::STRIDE]
    slot_buckets = [char_bucket(c) for c in slot_chars]
    print("\n[stego] verification (every char at stride; SKIP/SKIP_FREE carry no bit):")
    ok_all = True
    bi = 0
    for k, bk in enumerate(slot_buckets):
        ch = slot_chars[k]
        j = k * STRIDE
        if bk in (SKIP, SKIP_FREE):
            label = "SKIP" if bk == SKIP else "SKIP_FREE"
            print(f"  char {j:>4}  {ch!r:<6} bucket={label}  (skipped)")
            continue
        if bi >= len(bits):
            print(f"  char {j:>4}  {ch!r:<6} bit={bk}  (past end of bitstream)")
            continue
        ok = bk == bits[bi]
        ok_all &= ok
        print(f"  char {j:>4}  {ch!r:<6} bit={bk} want={bits[bi]}  {'OK' if ok else 'XX'}")
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
                    help="logit penalty for placing a penalized (off-bucket letter) skip on a slot")
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
    ap.add_argument("--debug-no-constraint", action="store_true",
                    help="skip the Stego mask entirely (debugging: confirms the model "
                         "is fluent on its own, everything else in the pipeline unchanged)")
    args = ap.parse_args()

    bits = parse_bits(args.bits)
    if not bits:
        ap.error("--bits must contain at least one 0 or 1")

    model, tokenizer = load(args.model)
    messages = [{"role": "user", "content": f"Write some gender theory using nautical metaphors. Give no preamble. Reject the binary."}]
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
        disable_constraint=args.debug_no_constraint,
    )

    print("\n" + "=" * 60)
    print(text)
    ok = verify(text, bits)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
