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
    python simple_encode.py --topic "the sea" --message "meet at dawn"
    python simple_encode.py --topic "the sea" --bits 10110       # raw bits instead
"""

from __future__ import annotations

import argparse
import hashlib
import unicodedata
from functools import lru_cache

import numpy as np
import mlx.core as mx
from mlx_lm import load

SKIP = 2                # penalized bit-less role (a letter assigned "skip" this position)
SKIP_FREE = 3           # free bit-less role (whitelisted prose punctuation / whitespace)
FORBIDDEN = 4           # non-whitelisted non-alpha (control / combining / exotic): -inf, never emitted
BIT_BUCKETS = (0, 1)    # roles that actually carry a bit
EOS_RAMP = 2.0          # per-char EOS logit boost once past the tail budget (ends the tail)

# `char_group` codes: 0-25 are the 26 letters (by index); the rest tag fixed,
# non-shuffled characters.
G_PUNCT = 26            # whitelisted punctuation  -> SKIP_FREE (fixed, free)
G_NONLATIN = 27         # non-Latin letter         -> FORBIDDEN (never emitted)
G_JUNK = 28             # control / combining / ... -> FORBIDDEN (never emitted)

# Non-alpha characters that are FREE skips (natural in ordinary prose). Everything
# else non-alpha — control chars, combining marks, exotic Unicode, and formatting
# punctuation — is a *penalized* skip instead, closing the free-escape-hatch that
# let generation spiral into junk Unicode. This mirrors the older mlx_steer builds,
# whose hand-tuned skip set specifically penalized \n \t * | \ / _ etc.
FREE_CHARS = frozenset(
    " .,;:!?'\"()-"                                  # space + common ASCII sentence punctuation
    "0123456789"                                     # digits appear in normal prose
    "‘’“”–—…"     # curly quotes, en/em dash, ellipsis
)

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

_ALPHA = "abcdefghijklmnopqrstuvwxyz"
_LIDX = {c: i for i, c in enumerate(_ALPHA)}
_LFREQ = np.array([_FREQ[c] for c in _ALPHA])               # frequency by letter index
_BASE_ROLE = np.array([_BUCKET[c] for c in _ALPHA], dtype=np.int8)  # key=None assignment
_BASE_ROLE.setflags(write=False)


def parse_bits(s: str) -> list[int]:
    return [1 if c == "1" else 0 for c in s if c in "01"]


@lru_cache(maxsize=1 << 18)
def _letter_roles(key, pos: int) -> np.ndarray:
    """Keyed assignment of all 26 letters to roles {bit0=0, bit1=1, SKIP=2} at
    character position `pos`. This reshuffles *membership* every position (not just
    relabels three fixed groups), so no fixed letter clustering survives — over
    text each letter lands in each role about equally. A greedy pass in a keyed
    order fills the currently lightest role, keeping each role ~1/3 of letter
    frequency for fluency. key=None returns the fixed base assignment."""
    if key is None:
        return _BASE_ROLE
    seed = int.from_bytes(hashlib.sha256(f"{key}|{pos}".encode()).digest()[:8], "big")
    order = np.random.default_rng(seed).permutation(26)
    sums = [0.0, 0.0, 0.0]
    role = np.empty(26, dtype=np.int8)
    for L in order:
        r = int(np.argmin(sums))
        role[L] = r
        sums[r] += _LFREQ[L]
    role.setflags(write=False)
    return role


def char_group(ch: str) -> int:
    """Key-independent class of a character: 0-25 = letter index (shuffled per
    position); G_PUNCT = whitelisted punctuation; G_NONLATIN = non-Latin letter;
    G_JUNK = control/combining/exotic. The last two are never emitted."""
    if not ch.isalpha():
        return G_PUNCT if ch in FREE_CHARS else G_JUNK
    c = ch.lower()
    if c in _LIDX:
        return _LIDX[c]
    for base in unicodedata.normalize("NFKD", c):
        if base in _LIDX:
            return _LIDX[base]
    return G_NONLATIN


def char_role(ch: str, pos: int = 0, key=None) -> int:
    """Bit-role of `ch` at character position `pos` under `key`: 0/1 encode a bit;
    SKIP/SKIP_FREE carry none; FORBIDDEN is never emitted. Only the 26 letters are
    shuffled by the key — punctuation is a free skip, and non-Latin letters and junk
    are forbidden (they are always-legal characters, i.e. escape hatches)."""
    g = char_group(ch)
    if g in (G_JUNK, G_NONLATIN):
        return FORBIDDEN
    if g < 26:                                          # a letter
        return int(_letter_roles(key, pos)[g])
    return SKIP_FREE


def char_bucket(ch: str) -> int:
    """Unkeyed base role — used by tests and reporting."""
    return char_role(ch, 0, None)


def extract(text: str, key=None) -> list[int]:
    """Recover the hidden bitstream: at each character position apply the keyed
    alphabet and keep the bit-carrying roles, dropping every flavor of skip."""
    return [r for pos, ch in enumerate(text)
            for r in (char_role(ch, pos, key),) if r in BIT_BUCKETS]


def verify(text: str, bits: list[int], key=None) -> bool:
    """Check the payload reads back out of `text` under `key`."""
    got = extract(text, key)
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
    """Vectorized keyed stride-1 constraint over the whole vocabulary.

    Precomputes, per token, the `char_group` code (`CHR`) of each of its characters
    (0-25 letter index, G_PUNCT, G_NONLATIN, or -1 padding) and a junk flag. At each
    step `constrain` applies the secret key's per-character-position letter->role
    map to get each character's live role, then (vectorized) forbids any token whose
    bit-carrying characters would disagree with the upcoming payload and penalizes
    skip characters. `key=None` reproduces the fixed unkeyed scheme."""

    def __init__(self, tok, bits: list[int], skip_penalty: float, key=None):
        self.tok = tok
        self.bits = np.array(bits, dtype=np.int8)
        self.nbits = len(bits)
        self.bits_pad = np.append(self.bits, np.int8(-1))   # sentinel for out-of-range gathers
        self.skip_penalty = skip_penalty
        self.key = key
        self.tables = None

    def _build(self, V: int):
        print(f"[stego] indexing {V} tokens (one-time)...")
        self.junk = np.zeros(V, dtype=bool)
        max_c = 1
        rows = []
        for i in range(V):
            gs = [char_group(c) for c in self.tok.decode([i])]
            rows.append(gs)
            # A FORBIDDEN *role* does not block a token by itself — only this mask
            # does — so every never-emit class has to be collected here. Non-Latin
            # letters (CJK/Greek/Cyrillic) were previously a penalized-but-always-
            # legal SKIP, i.e. a guaranteed escape hatch that got used once pricing
            # closed the space and punctuation ones.
            self.junk[i] = (G_JUNK in gs) or (G_NONLATIN in gs)
            if len(gs) > max_c:
                max_c = len(gs)
        CHR = np.full((V, max_c), -1, dtype=np.int16)   # char_group code per char, -1 = padding
        for i, gs in enumerate(rows):
            if gs:
                CHR[i, :len(gs)] = gs
        self.CHR, self.max_c = CHR, max_c

    def _roles(self, char_pos: int):
        """(V, max_c) live role of each token character at the given char position."""
        E = self.max_c
        rmap = np.stack([_letter_roles(self.key, char_pos + k) for k in range(E)])   # (E, 26)
        role = rmap[np.arange(E)[None, :], np.clip(self.CHR, 0, 25)]  # letters -> bit0/bit1/SKIP
        role = np.where(self.CHR == G_PUNCT, SKIP_FREE, role)        # punctuation: free skip
        role = np.where(self.CHR == G_NONLATIN, FORBIDDEN, role)     # non-Latin letter: never emit
        role = np.where(self.CHR == G_JUNK, FORBIDDEN, role)         # junk: never emitted
        role = np.where(self.CHR == -1, -1, role)                    # padding
        return role

    def token_bits(self, token: int, char_pos: int, consumed: int) -> int:
        """How many payload bits `token` encodes if placed at `char_pos` given `consumed`."""
        rem = self.nbits - consumed
        if rem <= 0:
            return 0
        n = 0
        for k, ch in enumerate(self.tok.decode([token])):
            if char_role(ch, char_pos + k, self.key) in BIT_BUCKETS:
                n += 1
                if n >= rem:
                    break
        return n

    def constrain(self, consumed: int, char_pos: int, vec: np.ndarray):
        """Return (masked_logits, bits_encoded_per_token) at payload bit `consumed`
        and character position `char_pos`. Wrong-bit and junk tokens -> -inf; each
        skip character subtracts `skip_penalty`."""
        rem = self.nbits - consumed
        if rem <= 0:                                    # payload spent -> only junk forbidden
            out = vec.copy()
            out[self.junk] = -np.inf
            return out, np.zeros(vec.shape[0], dtype=np.int32)
        role = self._roles(char_pos)                    # (V, max_c) live roles
        enc_mask = (role == 0) | (role == 1)            # bit-carrying characters
        cum = np.cumsum(enc_mask, axis=1) - enc_mask    # payload-bit offset of each enc char
        bit_pos = consumed + cum
        within = enc_mask & (bit_pos < self.nbits)      # enc chars still inside the payload
        target = self.bits_pad[np.clip(bit_pos, 0, self.nbits)]
        bad = (within & (role != target)).any(axis=1) | self.junk
        out = vec - self.skip_penalty * (role == SKIP).sum(axis=1)
        out[bad] = -np.inf
        bits_enc = within.sum(axis=1).astype(np.int32)
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


def _lookahead(enc: Encoder, model, cache, consumed_after: int, char_pos_after: int,
               nlogits, *, depth: int, eos_ids: set, trim) -> float:
    """Fluency of committing to an already-advanced candidate.

    Greedily runs the *constrained* rollout `depth` steps forward and returns the
    mean log-probability the model itself assigns to each token it is forced onto.
    Near 0 => the constraint isn't fighting the model here (fluent continuation);
    very negative => it keeps being pushed onto tokens it dislikes (garbled ahead).
    Tracks the character position so the keyed alphabet stays consistent. Cache is
    restored to the tokens+[candidate] state; the caller undoes the candidate."""
    rawn = _vec(nlogits)
    total, steps, advanced = 0.0, 0, 0
    cur, cpos = consumed_after, char_pos_after
    for _ in range(depth):
        masked, _ = enc.constrain(cur, cpos, rawn)
        if cur < enc.nbits:                          # don't let the probe "stop" to dodge
            for e in eos_ids:
                masked[e] = -np.inf
        nxt = int(np.argmax(masked))
        total += float(rawn[nxt]) - _logsumexp(rawn)  # log P_model of the forced token
        steps += 1
        clog = model(mx.array([[nxt]]), cache=cache)[:, -1, :]
        mx.eval(clog)
        advanced += 1
        if nxt in eos_ids:
            break
        cur += enc.token_bits(nxt, cpos, cur)
        cpos += len(enc.tok.decode([nxt]))
        rawn = _vec(clog)
    if advanced:
        trim(cache, advanced)
    return total / steps if steps else 0.0


def steer_generate(model, tokenizer, prompt, bits, *, max_tokens=800, temperature=0.8,
                   skip_penalty=1.0, num_candidates=12, fluency_weight=3.0,
                   bit_bonus=2.0, rollout_depth=3, tail_chars=200, top_p=0.95,
                   freq_penalty=0.5, rep_window=64, no_repeat_ngram=3,
                   seed=0, key=None, stream=False) -> str:
    """Stride-1 constrained decoding with a fluency-driven candidate search.

    Every character is a slot, so every step (until the payload is spent) is a
    constrained encoding step. Each such step:

      1. Constrain the whole vocab (vectorized) and block EOS while bits remain.
      2. Shortlist the top `num_candidates` valid tokens by the model's own
         probability — the fluent options, encoding or skip alike.
      3. Trial-run each; score it with `_lookahead` (mean log-prob of the forced
         constrained continuation = fluency) plus a `bit_bonus` per bit it
         encodes, so encoding happens where it stays natural.
      4. Sample a winner from the combined scores at `temperature`/`top_p`.

    Once the payload is fully encoded, generation continues unconstrained. After
    `tail_chars` more characters, EOS is progressively up-weighted (by `EOS_RAMP`
    per overshoot char) so the model ends the cover text at a natural point soon
    after — instead of a hard cut on the last hidden bit. `tail_chars == 0` steers
    to an end as soon as the payload is in.
    """
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache
    except ImportError:
        def can_trim_prompt_cache(_cache):
            return True

    enc = Encoder(tokenizer, bits, skip_penalty, key)
    rng = np.random.default_rng(seed)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or
                  ([tokenizer.eos_token_id]
                   if getattr(tokenizer, "eos_token_id", None) is not None else []))

    tokens = list(prompt)
    offset = len(prompt)
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    enc._build(logits.shape[1])                          # vocab size known now
    for e in eos_ids:                                    # EOS decodes to '<|..|>' (junk chars) but
        if e < enc.junk.size:                            # must stay usable to end the cover text
            enc.junk[e] = False
    use_lookahead = num_candidates > 1 and rollout_depth > 0 and can_trim_prompt_cache(cache)

    consumed, tail = 0, 0
    for _ in range(max_tokens):
        in_tail = consumed >= enc.nbits
        raw = _vec(logits)

        if in_tail:                                      # payload done -> free finish
            base = _antirepeat(raw.copy(), tokens[offset:], freq_penalty=freq_penalty,
                               window=rep_window, no_repeat_ngram=no_repeat_ngram)
            base[enc.junk] = -np.inf                     # keep junk out of the free tail too
            if tail >= tail_chars:                       # past the tail budget: up-weight EOS,
                for e in eos_ids:                        # ramping with overshoot so the model
                    base[e] += EOS_RAMP * (tail - tail_chars + 1)  # ends at a natural point soon
            choice = _sample_logits(base, temperature, rng, top_p)
        else:
            char_pos = len(tokenizer.decode(tokens[offset:]))   # keyed alphabet needs the position
            masked, bits_enc = enc.constrain(consumed, char_pos, raw)
            for e in eos_ids:                            # never stop mid-payload
                masked[e] = -np.inf
            base = _antirepeat(masked.copy(), tokens[offset:], freq_penalty=freq_penalty,
                               window=rep_window, no_repeat_ngram=no_repeat_ngram)
            if not use_lookahead:
                choice = _sample_logits(base + bit_bonus * bits_enc, temperature, rng, top_p)
            else:
                logp = base - _logsumexp(base)               # immediate fluency (log-softmax)
                valid = np.where(np.isfinite(base))[0]
                k = min(num_candidates, valid.size)
                cand = valid[np.argsort(base[valid])[-k:]]   # shortlist = what the model wants
                combined = np.empty(k)
                for j, c in enumerate(cand):
                    c = int(c)
                    nlogits = model(mx.array([[c]]), cache=cache)[:, -1, :]
                    mx.eval(nlogits)
                    look = _lookahead(enc, model, cache, consumed + int(bits_enc[c]),
                                      char_pos + len(tokenizer.decode([c])), nlogits,
                                      depth=rollout_depth, eos_ids=eos_ids, trim=trim_prompt_cache)
                    combined[j] = logp[c] + fluency_weight * look + bit_bonus * bits_enc[c]
                    trim_prompt_cache(cache, 1)          # undo the trial token
                choice = int(cand[_sample_logits(combined, temperature, rng, top_p)])
            consumed += int(bits_enc[choice])

        if choice in eos_ids:                            # natural end (payload already encoded);
            break                                        # don't append, so no EOS marker leaks in
        tokens.append(choice)
        if stream:                                       # print each token the moment it's finalized
            print(tokenizer.decode([choice]), end="", flush=True)
        if in_tail:
            tail += len(tokenizer.decode([choice]))
        logits = model(mx.array([[choice]]), cache=cache)[:, -1, :]
        mx.eval(logits)

    if stream:
        print()
    return tokenizer.decode(tokens[offset:])


def encoding_summary(text: str, bits: list[int], key=None) -> float:
    """Print encoding density: hidden bits per character of cover text."""
    encoded = min(len(bits), len(extract(text, key)))
    bpc = encoded / len(text) if text else 0.0
    print(f"[stego] density: {encoded} bits / {len(text)} chars = {bpc:.4f} bits/char")
    return bpc


def build_prompt(tokenizer, topic: str, think: bool):
    """Chat prompt for the cover text.

    Thinking models (SmolLM3, Qwen3, ...) accept `enable_thinking`; leaving it off
    stops the model emitting a <think> reasoning block, which would otherwise get
    stego-encoded into the payload as garbage. Models whose tokenizer doesn't take
    the argument fall back to the plain template unchanged."""
    messages = [{"role": "user", "content": f"Write a short story about: {topic}"}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             enable_thinking=think)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def main() -> None:
    # --- fixed tuning knobs (edit here; kept off the CLI) ---
    TOP_P           = 0.95   # nucleus sampling cutoff
    FREQ_PENALTY    = 0.5    # anti-repetition: penalty per recent token occurrence
    REP_WINDOW      = 64     # anti-repetition: recent-token window
    NO_REPEAT_NGRAM = 3      # anti-repetition: block repeating any n-gram of this size

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True, help="what the cover text is about")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--message", help="secret text to compress (Unishox2) and hide")
    src.add_argument("--bits", help="raw bitstream to hide instead, e.g. 10110")
    ap.add_argument("--model", default="mlx-community/SmolLM3-3B-8bit",
                    help="mlx-community model id (8-bit / less-peaky models give better constrained fluency)")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--tail-chars", type=int, default=200,
                    help="free cover text after the payload; past it, EOS is up-weighted to end naturally")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--skip-penalty", type=float, default=1.0,
                    help="logit penalty per off-bucket-letter (SKIP) character")
    ap.add_argument("--candidates", type=int, default=12,
                    help="valid tokens trial-run per step (more = better search, slower)")
    ap.add_argument("--fluency-weight", type=float, default=3.0,
                    help="weight of the continuation-fluency score (0 = ignore fluency)")
    ap.add_argument("--bit-bonus", type=float, default=2.0,
                    help="score reward per bit a candidate encodes (raises density vs fluency)")
    ap.add_argument("--rollout-depth", type=int, default=3,
                    help="constrained steps looked ahead to judge fluency (0 disables lookahead)")
    ap.add_argument("--think", action="store_true",
                    help="allow the model to emit <think> reasoning (default off; thinking "
                         "models like SmolLM3/Qwen3 otherwise encode a reasoning block as garbage)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", default=None,
                    help="secret key: shuffles the letter->bit alphabet per character; the "
                         "recipient needs the same key to decode (omit for the fixed alphabet)")
    ap.add_argument("--quiet", action="store_true",
                    help="don't stream the cover text as it is generated (default: stream progress)")
    args = ap.parse_args()

    codec = None
    if args.message is not None:
        import payload_codec as codec
        bits = codec.compress_to_bits(args.message)
        print(f"[codec] {codec.ratio_report(args.message, bits)}")
    else:
        bits = parse_bits(args.bits)
        if not bits:
            ap.error("--bits must contain at least one 0 or 1")

    model, tokenizer = load(args.model)
    prompt = build_prompt(tokenizer, args.topic, args.think)

    text = steer_generate(
        model, tokenizer, prompt, bits,
        max_tokens=args.max_tokens,
        tail_chars=args.tail_chars,
        temperature=args.temperature,
        skip_penalty=args.skip_penalty,
        num_candidates=args.candidates,
        fluency_weight=args.fluency_weight,
        bit_bonus=args.bit_bonus,
        rollout_depth=args.rollout_depth,
        top_p=TOP_P,
        freq_penalty=FREQ_PENALTY,
        rep_window=REP_WINDOW,
        no_repeat_ngram=NO_REPEAT_NGRAM,
        seed=args.seed,
        key=args.key,
        stream=not args.quiet,
    )

    if args.quiet:                          # streaming already showed the text live
        print("\n" + "=" * 60)
        print(text)
    ok = verify(text, bits, args.key)
    encoding_summary(text, bits, args.key)
    if codec is not None and ok:
        recovered = codec.decompress_from_bits(extract(text, args.key))
        match = recovered == args.message
        print(f"[codec] recovered message: {recovered!r}")
        print(f"[codec] {'MATCH — message round-trips' if match else 'MISMATCH'}")
        ok = match
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
