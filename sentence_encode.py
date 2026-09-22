#!/usr/bin/env python3
"""
sentence_encode.py — steganographic encoding, one sentence at a time.

Hides a bitstream in fluent cover text by constraining which characters the model
may emit: at each character position a character either carries the next payload
bit, or skips. Characters split into roles:

    role 0                -> alpha, encodes bit 0
    role 1                -> alpha, encodes bit 1
    role 2 (SKIP)         -> alpha, off-bucket: carries no bit, *penalized*
    role 3 (SKIP_FREE)    -> non-alpha (space/punct/...): carries no bit, *free*
    role 4 (FORBIDDEN)    -> control / combining / non-Latin: never emitted

A key reshuffles which letters hold which role, per character position, so the
same letter carries different bits at different places in the text. Decoding
needs only the text and the key — never the model — which is why the browser
decoder in script.js can be a faithful reimplementation of a specification
rather than of a library internal.

A sentence is the unit of acceptance. Generation runs to a terminator, the finished
sentence is scored, and a bad one is rolled back — KV cache, token list and payload
position — then retried from the same state with fresh sampling. Accepted sentences
are never revisited, so cost is bounded per sentence rather than per paragraph, and
one garbled clause no longer discards the good text around it.

Scoring is free: per-token excess surprise (NLL - entropy) comes from logits already
computed during generation, so judging a sentence costs no extra forward pass. Three
signals decide acceptance, each aimed at a failure mode seen in practice:

    shocks   tokens far above the model's own expectation      (garbled text)
    coined   words absent from the system dictionary           (invented words)
    caps     capitalised-word rate                             (title-register drift)

Known limit: retries inherit the prefix, so a sentence-level loop cannot escape a
bad *opening*. When the accept rate collapses (nothing accepted over many attempts)
the sample is usually unsalvageable and is better restarted with another seed.

    pip install mlx-lm
    python sentence_encode.py --topic "the sea" --message "meet at dawn" --key k
    python sentence_encode.py --topic "the sea" --bits 10110100 --key k --clean
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import unicodedata
from functools import lru_cache

# Quiet the backends: huggingface_hub and transformers read these when they are
# imported, which `mlx_lm` does below — so they must be set first.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # "Fetching N files" bars
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")     # tokenizer warnings

import numpy as np


# --------------------------------------------------------------------------- #
# model backend
# --------------------------------------------------------------------------- #
# Only five operations touch the inference engine: load, open a cache, run one
# forward, read the logits as a numpy vector, and roll the cache back. Isolating
# them here lets the same constraint machinery run on MLX (Apple Silicon, local)
# and on PyTorch/CUDA (a Hugging Face Space) without a second copy of the encoder.
# Imports are deferred so neither engine has to be installed for the other to work.

class _MLXBackend:
    name = "mlx"

    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        self._mx, self._load = mx, load
        self._make, self._trim = make_prompt_cache, trim_prompt_cache

    def load(self, model_id):
        return self._load(model_id)

    def new_cache(self, model):
        return self._make(model)

    def forward(self, model, ids, cache):
        out = model(self._mx.array([list(ids)]), cache=cache)[:, -1, :]
        self._mx.eval(out)
        return out

    def to_numpy(self, logits):
        return np.array(logits.astype(self._mx.float32)).reshape(-1)

    def trim(self, cache, n):
        self._trim(cache, n)


class _TorchBackend:
    """transformers + DynamicCache. `crop(-n)` is the analogue of MLX's
    trim_prompt_cache, and is what makes sentence-level rollback possible."""
    name = "torch"

    def __init__(self, device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        self._torch, self._DynamicCache = torch, DynamicCache
        self._AM, self._AT = AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

    def load(self, model_id):
        tok = self._AT.from_pretrained(model_id)
        model = self._AM.from_pretrained(model_id, dtype=self.dtype).to(self.device)
        model.eval()
        return model, tok

    def new_cache(self, model):
        cache = self._DynamicCache(config=model.config)
        # Sliding-window and linear-attention layers drop past states as they go,
        # so crop() raises unless recording is switched on first.
        if hasattr(cache, "activate_past_recording"):
            cache.activate_past_recording()
        return cache

    def forward(self, model, ids, cache):
        t = self._torch
        x = t.tensor([list(ids)], dtype=t.long, device=self.device)
        with t.inference_mode():
            return model(input_ids=x, past_key_values=cache, use_cache=True).logits[:, -1, :]

    def to_numpy(self, logits):
        return logits.float().cpu().numpy().reshape(-1)

    def trim(self, cache, n):
        cache.crop(-int(n))


_BACKEND = None


def backend():
    """The active backend, chosen once. STEGO_BACKEND=torch|mlx overrides the
    default, which prefers MLX and falls back to PyTorch."""
    global _BACKEND
    if _BACKEND is None:
        want = os.environ.get("STEGO_BACKEND", "").lower()
        if want == "torch":
            _BACKEND = _TorchBackend()
        elif want == "mlx":
            _BACKEND = _MLXBackend()
        else:
            try:
                _BACKEND = _MLXBackend()
            except ImportError:
                _BACKEND = _TorchBackend()
    return _BACKEND


def set_backend(b):
    """Install a backend explicitly (a Space loads the model once at startup)."""
    global _BACKEND
    _BACKEND = b


def torch_backend(device=None, dtype=None):
    """Explicit PyTorch backend. A ZeroGPU Space pins device="cuda" at import
    time, because ZeroGPU wants models placed on cuda at module level."""
    return _TorchBackend(device, dtype)


def mlx_backend():
    return _MLXBackend()


SKIP = 2                # penalized bit-less role (a letter assigned "skip" this position)
SKIP_FREE = 3           # free bit-less role (whitelisted prose punctuation / whitespace)
FORBIDDEN = 4           # non-whitelisted non-alpha (control / combining / exotic): -inf, never emitted
BIT_BUCKETS = (0, 1)    # roles that actually carry a bit

# Tokens longer than this are marked junk and never emitted. The constraint matrix
# is (vocab x longest token), so one 128-character token — the vocab's longest are
# whitespace runs and '//------' rules — made 95% of that rectangle padding and
# cost 22x the necessary work per step. Capping at 24 drops 0.4% of the vocab,
# almost all of it formatting artefacts unwanted in prose anyway.
MAX_TOKEN_CHARS = 24

# `char_group` codes: 0-25 are the 26 letters (by index); the rest tag fixed,
# non-shuffled characters.
G_PUNCT = 26            # whitelisted punctuation  -> SKIP_FREE (fixed, free)
G_NONLATIN = 27         # non-Latin letter         -> FORBIDDEN (never emitted)
G_JUNK = 28             # control / combining / ... -> FORBIDDEN (never emitted)

# Non-alpha characters that are FREE skips (natural in ordinary prose). Everything
# else non-alpha — control chars, combining marks, exotic Unicode, and formatting
# punctuation — is a *penalized* skip instead, closing the free-escape-hatch that
# let generation spiral into junk Unicode: \n \t * | \ / _ and friends are all
# forbidden rather than free.
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


@contextlib.contextmanager
def _muffled(active: bool):
    """Send stdout/stderr to /dev/null while active, and yield the *real* stdout so
    a caller can still write to it on purpose. That is what lets --clean stream the
    cover text live while everything else stays suppressed."""
    console = sys.stdout
    if not active:
        yield console
        return
    with open(os.devnull, "w") as null, \
            contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
        yield console


def parse_bits(s: str) -> list[int]:
    return [1 if c == "1" else 0 for c in s if c in "01"]


# --------------------------------------------------------------------------- #
# keyed alphabet / constraint
# --------------------------------------------------------------------------- #


def _letter_order(key, pos: int) -> list[int]:
    """Keyed order in which the 26 letters claim their roles.

    Uses SHA-256 alone — no numpy RNG — so JavaScript can derive the identical
    order (see the decoder in script.js). numpy's PCG64 is not reasonably reproducible
    outside numpy, which would have made the browser decoder a re-implementation
    of a library internal rather than of a specification.

    26 sort keys of 4 bytes each; ties broken by letter index so the order is
    fully determined."""
    stream = b"".join(hashlib.sha256(f"{key}|{pos}|{b}".encode()).digest()
                      for b in range(4))
    return [i for _, i in sorted(
        (int.from_bytes(stream[i * 4:i * 4 + 4], "big"), i) for i in range(26))]


@lru_cache(maxsize=1 << 18)
def _letter_roles(key, pos: int) -> np.ndarray:
    """Keyed assignment of all 26 letters to roles {bit0=0, bit1=1, SKIP=2} at
    character position `pos`. This reshuffles *membership* every position (not just
    relabels three fixed groups), so no fixed letter clustering survives — over
    text each letter lands in each role about equally. A greedy pass in the keyed
    order fills the currently lightest role, keeping each role ~1/3 of letter
    frequency for fluency. key=None returns the fixed base assignment."""
    if key is None:
        return _BASE_ROLE
    sums = [0.0, 0.0, 0.0]
    role = np.empty(26, dtype=np.int8)
    for L in _letter_order(key, pos):
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

    def __init__(self, tok, bits: list[int], skip_penalty: float, key=None,
                 topk: int = 0):
        self.tok = tok
        self.bits = np.array(bits, dtype=np.int8)
        self.nbits = len(bits)
        self.bits_pad = np.append(self.bits, np.int8(-1))   # sentinel for out-of-range gathers
        self.skip_penalty = skip_penalty
        # Restrict the per-step constraint to the `topk` highest-logit tokens
        # instead of the whole vocabulary. The constraint is (vocab x max_c)
        # elementwise work, so this is the single cheapest speedup available.
        # Measured on real generations: at topk=2048 the legal tokens inside the
        # window still hold 99.3% of the post-mask probability mass (5th pct
        # 96.9%), with a median of 406 legal tokens and no step ever starved.
        # 0 disables it and uses the full vocabulary.
        self.topk = int(topk)
        self.key = key
        self.tables = None

    # Static per-tokenizer tables, shared across Encoder instances. Building them
    # decodes every token in the vocabulary (~0.7 s for 128k) and the result
    # depends only on the tokenizer — not on the key, the bits, or the payload —
    # so a long-lived process (a Space serving requests) pays for it once instead
    # of on every call. `junk` is deliberately excluded: callers mutate it to
    # un-forbid EOS, so each Encoder gets its own copy.
    _SHARED: dict = {}
    _SHARED_FIELDS = ("CHR", "max_c", "_letter_at", "_clipped", "_fixed")

    def _build(self, V: int):
        ck = (id(self.tok), V)
        hit = Encoder._SHARED.get(ck)
        if hit is not None:
            for k in Encoder._SHARED_FIELDS:
                setattr(self, k, hit[k])
            self.junk = hit["junk"].copy()
            return
        print(f"[stego] indexing {V} tokens (one-time)...")
        self.junk = np.zeros(V, dtype=bool)
        max_c = 1
        rows = []
        for i in range(V):
            gs = [char_group(c) for c in self.tok.decode([i])]
            if len(gs) > MAX_TOKEN_CHARS:      # never emitted, so truncating is safe
                gs = gs[:MAX_TOKEN_CHARS]      # and keeps the whole matrix narrow
                self.junk[i] = True
            rows.append(gs)
            # A FORBIDDEN *role* does not block a token by itself — only this mask
            # does — so every never-emit class has to be collected here. Non-Latin
            # letters (CJK/Greek/Cyrillic) were previously a penalized-but-always-
            # legal SKIP, i.e. a guaranteed escape hatch that got used once pricing
            # closed the space and punctuation ones.
            self.junk[i] |= (G_JUNK in gs) or (G_NONLATIN in gs)
            if len(gs) > max_c:
                max_c = len(gs)
        CHR = np.full((V, max_c), -1, dtype=np.int16)   # char_group code per char, -1 = padding
        for i, gs in enumerate(rows):
            if gs:
                CHR[i, :len(gs)] = gs
        self.CHR, self.max_c = CHR, max_c
        # Static per-character facts, computed once instead of on every step: the
        # comparisons below used to run over the full matrix each token.
        self._letter_at = (CHR >= 0) & (CHR < 26)
        self._clipped = np.clip(CHR, 0, 25)
        self._fixed = np.where(CHR == G_PUNCT, np.int8(SKIP_FREE),
                      np.where(CHR == -1, np.int8(-1), np.int8(FORBIDDEN)))
        cached = {k: getattr(self, k) for k in Encoder._SHARED_FIELDS}
        cached["junk"] = self.junk.copy()
        Encoder._SHARED[ck] = cached

    def _roles(self, char_pos: int, rows=None):
        """Live role of each token character at the given char position, for all
        tokens or just `rows` — shape (V, max_c) or (len(rows), max_c)."""
        E = self.max_c
        rmap = np.stack([_letter_roles(self.key, char_pos + k) for k in range(E)])   # (E, 26)
        clipped = self._clipped if rows is None else self._clipped[rows]
        gathered = rmap[np.arange(E)[None, :], clipped]              # letters -> bit0/bit1/SKIP
        # Everything that is not a letter has a role fixed at build time, so one
        # select against the precomputed table replaces four full-matrix compares.
        letter_at = self._letter_at if rows is None else self._letter_at[rows]
        fixed = self._fixed if rows is None else self._fixed[rows]
        return np.where(letter_at, gathered, fixed)

    def constrain(self, consumed: int, char_pos: int, vec: np.ndarray):
        """Return (masked_logits, bits_encoded_per_token) at payload bit `consumed`
        and character position `char_pos`. Wrong-bit and junk tokens -> -inf; each
        skip character subtracts `skip_penalty`."""
        rem = self.nbits - consumed
        if rem <= 0:                                    # payload spent -> only junk forbidden
            out = vec.copy()
            out[self.junk] = -np.inf
            return out, np.zeros(vec.shape[0], dtype=np.int32)
        V = vec.shape[0]
        rows = None
        if self.topk and self.topk < V:
            rows = np.argpartition(vec, -self.topk)[-self.topk:]

        role = self._roles(char_pos, rows)              # (n, max_c) live roles
        enc_mask = (role == 0) | (role == 1)            # bit-carrying characters
        cum = np.cumsum(enc_mask, axis=1) - enc_mask    # payload-bit offset of each enc char
        bit_pos = consumed + cum
        within = enc_mask & (bit_pos < self.nbits)      # enc chars still inside the payload
        target = self.bits_pad[np.clip(bit_pos, 0, self.nbits)]
        junk = self.junk if rows is None else self.junk[rows]
        bad = (within & (role != target)).any(axis=1) | junk
        sub = (vec if rows is None else vec[rows]) \
            - self.skip_penalty * (role == SKIP).sum(axis=1)
        sub[bad] = -np.inf
        sub_bits = within.sum(axis=1).astype(np.int32)
        sub_bits[bad] = 0

        if rows is None:
            return sub, sub_bits
        if not np.isfinite(sub).any():                  # window starved: redo on all
            self_topk, self.topk = self.topk, 0
            try:
                return self.constrain(consumed, char_pos, vec)
            finally:
                self.topk = self_topk
        out = np.full(V, -np.inf)
        out[rows] = sub
        bits_enc = np.zeros(V, dtype=np.int32)
        bits_enc[rows] = sub_bits
        return out, bits_enc


# --------------------------------------------------------------------------- #
# sampling / scoring helpers
# --------------------------------------------------------------------------- #
def _vec(logits) -> np.ndarray:
    return backend().to_numpy(logits)


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
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            enable_thinking=think)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return _as_ids(out)


def _as_ids(out) -> list[int]:
    """Token ids from whatever apply_chat_template returned.

    mlx_lm hands back a plain list of ids; transformers 5.x hands back a
    BatchEncoding, and iterating that yields its *keys* (strings), which fails
    far downstream with a confusing error rather than here."""
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if isinstance(out, str):
        raise TypeError("chat template returned text, not token ids; "
                        "call it with tokenize=True")
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):     # batched -> take the row
        out = list(out[0])
    return [int(i) for i in out]


# --------------------------------------------------------------------------- #
# sentence-level control loop
# --------------------------------------------------------------------------- #


TERMINATORS = ".!?"
DICT_PATH = "/usr/share/dict/words"

# A token this far above the model's own expectation reads as a jolt. Genuine
# text from this model tops out near 3 nats, so 5 is comfortably outside it.
SHOCK_NATS = 5.0


def load_vocab() -> set[str]:
    try:
        with open(DICT_PATH, encoding="utf-8", errors="ignore") as fh:
            return {w.strip().lower() for w in fh if w.strip()}
    except OSError:
        return set()


def _known(w: str, vocab: set[str]) -> bool:
    """Dictionary lookup that tolerates inflection. /usr/share/dict/words is web2,
    a lemma-only list with no plurals or verb forms — without this, ordinary words
    like "waves" and "painted" read as coined and the metric inverts (measured:
    genuine text scored a *worse* coined rate than stego text)."""
    if w in vocab:
        return True
    for suf, adds in (("s", [""]), ("es", ["", "e"]), ("ed", ["", "e"]),
                      ("ing", ["", "e"]), ("ly", [""]), ("er", ["", "e"]),
                      ("est", ["", "e"]), ("d", [""]), ("n", [""])):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            stem = w[:-len(suf)]
            if any(stem + a in vocab for a in adds):
                return True
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in vocab:
                return True         # doubled consonant: "shimmering" -> "shimmer"
            if stem.endswith("i") and stem[:-1] + "y" in vocab:
                return True         # "carried" -> "carry"
    return False


def coined_words(text: str, vocab: set[str]) -> list[str]:
    """Words the dictionary does not know — the artifact a reader actually notices
    ("Serbucare", "fortiye-fife"). Hyphenated and apostrophe forms are checked
    part-by-part; short fragments are ignored as noise. Proper nouns count as
    coined, which is why the genuine baseline is measured the same way."""
    if not vocab:
        return []
    out = []
    for raw in text.split():
        cleaned = "".join(c for c in raw.lower() if c.isalpha() or c in "'-")
        parts = [p for p in cleaned.replace("'", "-").split("-") if len(p) > 2]
        if parts and not all(_known(p, vocab) for p in parts):
            out.append(raw)
    return out


def register_oddity(text: str) -> float:
    """Fraction of non-initial words that are Capitalized — a register check.

    Word-salad often passes the dictionary test (every word is real) but arrives as
    a title: "Pacific Maiden Awake on a Quiet Shore". Ordinary prose sits near the
    proper-noun rate; a heading sits near 1.0."""
    words = [w for w in text.split() if w[:1].isalpha()]
    if len(words) < 4:
        return 0.0
    return sum(w[:1].isupper() for w in words[1:]) / (len(words) - 1)


def payload_span(text: str, nbits: int, key) -> int | None:
    """Characters consumed to carry the whole payload, or None if it never fit.
    Density must be measured over this span — the free tail after it carries no
    payload and dividing by the whole text roughly halves the apparent density."""
    n = 0
    for i, ch in enumerate(text):
        if char_role(ch, i, key) in BIT_BUCKETS:
            n += 1
            if n >= nbits:
                return i + 1
    return None


MIN_SENTENCE_TOKENS = 2   # a terminator may not be the entire sentence


def _can_end(sent: str, n_tokens: int) -> bool:
    """Whether `sent` is allowed to finish yet.

    The terminator check runs immediately after the first sampled token, so
    without this a sentence can be the single token "." — complete, zero shocks,
    zero coined words, and therefore the *best*-scoring candidate of a round.
    Committing one of those makes no progress and shreds the prose. Requiring a
    couple of tokens and at least one letter costs nothing on real sentences and
    removes the degenerate optimum."""
    return n_tokens >= MIN_SENTENCE_TOKENS and any(c.isalpha() for c in sent)


def _sentence_done(text: str) -> bool:
    """True once `text` looks like a finished sentence: a terminator, optionally
    followed by a closing quote/bracket and whitespace."""
    t = text.rstrip()
    if not t:
        return False
    while t and t[-1] in '"\')]”’':
        t = t[:-1]
    return bool(t) and t[-1] in TERMINATORS


class Budget:
    """Acceptance thresholds for one sentence.

    Defaults come from the measured genuine baseline: unconstrained text from this
    model has shock_rate ~0.00 and worst excess ~3 nats, so anything a reader would
    call garbled sits far outside. They are deliberately loose enough to be
    satisfiable — too tight and every attempt fails, and the fallback (best of N)
    quietly becomes the only path."""

    def __init__(self, max_shocks=1, worst=12.0, max_coined=1, hard_abort=18.0,
                 max_caps=0.35):
        self.max_shocks = max_shocks      # tokens above SHOCK_NATS tolerated
        self.worst = worst                # ceiling on the single worst token
        self.max_coined = max_coined      # invented words tolerated
        self.hard_abort = hard_abort      # give up mid-sentence past this
        # Capitalised-word rate. Under constraint the model sometimes slides into
        # title register ("Echoes of the Tide: A Melanchy Voyage...") and stops
        # emitting terminators altogether, which defeats sentence segmentation.
        # Measured: prose ~0.11, a title ~0.67.
        self.max_caps = max_caps

    def verdict(self, shocks, worst, coined, caps=0.0):
        return (shocks <= self.max_shocks and worst <= self.worst
                and coined <= self.max_coined and caps <= self.max_caps)


def _grow_sentence(model, tok, enc, cache, logits, *, tokens, offset, consumed,
                   rng, temperature, top_p, bit_bonus, eos_ids, budget,
                   max_chars, allow_abort=True):
    """Generate one sentence. Returns a dict describing it; the caller decides
    whether to keep it. Leaves the cache advanced by `n_tokens` either way."""
    new, excesses, cur, hit_eos, complete = [], [], consumed, False, False
    sent = ""
    while True:
        raw = _vec(logits)
        lse = _logsumexp(raw)
        char_pos = len(tok.decode(tokens[offset:])) + len(sent)
        masked, bits_enc = enc.constrain(cur, char_pos, raw)
        if cur < enc.nbits:                    # never stop mid-payload
            for e in eos_ids:
                masked[e] = -np.inf
        base = _antirepeat(masked.copy(), tokens[offset:] + new,
                              freq_penalty=0.5, window=64, no_repeat_ngram=3)
        base = base + bit_bonus * bits_enc
        if not np.isfinite(base).any():
            break
        p = np.exp(raw - lse)
        H = float(-(p * (raw - lse)).sum())

        # One draw, taken as the model gives it. Token-level redraws (resampling a
        # jolt at the same step) were tried and reverted: fast, but they bias every
        # step toward the model's mode, which is the same axis the detectability
        # work is trying not to disturb. Quality control belongs at the sentence
        # level, where it can be judged and measured.
        choice = _sample_logits(base, temperature, rng, top_p)
        ex = (lse - float(raw[choice])) - H
        excesses.append(ex)

        if choice in eos_ids:
            hit_eos = True
            break
        s = tok.decode([choice])
        new.append(choice)
        sent += s
        cur += int(bits_enc[choice])
        logits = backend().forward(model, [choice], cache)
        # Early abort. Stop at the *budget*, not just at catastrophe: once the
        # sentence has already blown its shock allowance it cannot be accepted, so
        # every further token is paid for and then discarded.
        # Only a catastrophic token stops the sentence early. Aborting on the shock
        # *budget* was tried and reverted: a truncated sentence accumulates fewer
        # shocks and so outscored a finished one, so fragments won the ranking and
        # were committed mid-word ("sp" + "ikit" -> "spikit").
        if allow_abort and ex > budget.hard_abort:
            break                                  # aborted: `complete` stays False
        if (_sentence_done(sent) and _can_end(sent, len(new))) or len(sent) >= max_chars:
            complete = True
            break

    exa = np.array(excesses) if excesses else np.array([0.0])
    coined = len(coined_words(sent, _VOCAB))
    caps = register_oddity(sent)
    shocks = int((exa > SHOCK_NATS).sum())
    complete = complete or hit_eos
    return {"tokens": new, "text": sent, "consumed": cur, "eos": hit_eos,
            "complete": complete,
            "logits": logits, "n": len(new), "worst": float(exa.max()),
            "shocks": shocks, "coined": coined, "mean_ex": float(exa.mean()),
            "caps": caps,
            "ok": complete and budget.verdict(shocks, float(exa.max()), coined, caps),
            "bits_gained": cur - consumed}


_VOCAB: set[str] = set()


def _eos_ids(tok) -> set[int]:
    """End-of-text token ids, normalised across tokenizer flavours.

    mlx_lm's wrapper exposes `eos_token_ids` as a collection; a plain
    transformers tokenizer exposes it as a bare int (and some have only
    `eos_token_id`). Feeding the int straight to set() raises TypeError, which is
    the kind of difference that only shows up once the same code runs on a second
    backend."""
    out = set()
    for attr in ("eos_token_ids", "eos_token_id"):
        v = getattr(tok, attr, None)
        if v is None:
            continue
        out |= {int(v)} if isinstance(v, int) else {int(x) for x in v}
    return out


def generate(model, tok, prompt, bits, *, key=None, attempts=6, temperature=0.9,
             top_p=0.95, bit_bonus=1.0, skip_penalty=1.0,
             max_sentence_chars=200, budget=None, seed=0, verbose=False,
             stream_to=None, topk=0):
    """Encode `bits`, accepting one sentence at a time."""
    global _VOCAB
    if not _VOCAB:
        _VOCAB = load_vocab()
    budget = budget or Budget()
    enc = Encoder(tok, bits, skip_penalty, key, topk=topk)
    rng = np.random.default_rng(seed)
    eos_ids = _eos_ids(tok)

    tokens, offset = list(prompt), len(prompt)
    cache = backend().new_cache(model)
    logits = backend().forward(model, prompt, cache)
    enc._build(logits.shape[1])
    for e in eos_ids:
        if e < enc.junk.size:
            enc.junk[e] = False

    consumed = 0
    stats = {"accepted": 0, "rejected": 0, "fallback": 0}
    while consumed < enc.nbits:
        best = accepted = None
        def attempt(allow_abort=True):
            return _grow_sentence(
                model, tok, enc, cache, logits, tokens=tokens, offset=offset,
                consumed=consumed, rng=rng, temperature=temperature, top_p=top_p,
                bit_bonus=bit_bonus, eos_ids=eos_ids, budget=budget,
                max_chars=max_sentence_chars,
                allow_abort=allow_abort)

        for _ in range(attempts):
            cand = attempt()
            # rank fallbacks by how badly they miss, then by bits earned
            # Completeness dominates. An aborted fragment stops early and so
            # accumulates fewer shocks — without this it would outrank a finished
            # sentence, get committed mid-word, and split the next one across the
            # join ("sp" + "ikit" -> "spikit").
            key_ = (not cand["complete"], cand["caps"] > budget.max_caps,
                    cand["coined"], cand["shocks"], cand["worst"],
                    -cand["bits_gained"])
            if best is None or key_ < best[0]:
                best = (key_, cand)
            if cand["ok"]:
                stats["accepted"] += 1
                accepted = cand
                break
            stats["rejected"] += 1
            if cand["n"]:                       # roll the cache back and retry
                backend().trim(cache, cand["n"])
        # An accepted candidate is the one still in the cache: the loop breaks
        # before trimming it. `best` ranks coined words ahead of shocks, so a
        # *rejected* candidate can outrank an accepted one — taking it here would
        # replay its tokens on top of the accepted ones, leaving the cache
        # holding a sentence that never appears in the output and conditioning
        # everything after it on text the reader never sees.
        cand = accepted if accepted is not None else best[1]
        if not cand["ok"]:
            stats["fallback"] += 1
        if not cand["complete"]:
            # Every attempt aborted on the shock budget, so nothing reached a
            # terminator. Committing a fragment splits the following word across
            # the join ("sp" + "ikit" -> "spikit"), so spend one pass with the
            # abort lifted to get a sentence that can actually be committed. The
            # savings are kept: the cheap aborts already filtered the attempts.
            # Only the *abort* is lifted, not the budget: relaxing the thresholds
            # too would stamp the result "ok" no matter how bad it is, which is
            # exactly the mislabelling that hid a shocks=12 sentence behind an
            # [ok] tag. Stopping rules and judging rules are separate concerns.
            cand = attempt(allow_abort=False)
        elif not cand["ok"] and cand["tokens"]:
            # The best candidate was rolled back along with the rest, so its tokens
            # are no longer in the cache. Replay them deterministically rather than
            # generating afresh: sampling again would commit a *different* sentence
            # than the one just selected, making the whole comparison decorative.
            for t in cand["tokens"]:
                clog = backend().forward(model, [t], cache)
            cand = dict(cand, logits=clog)
        if verbose:
            print(f"  [{'ok ' if cand['ok'] else 'FB '}] shocks={cand['shocks']} "
                  f"worst={cand['worst']:5.1f} coined={cand['coined']} "
                  f"caps={cand['caps']:.2f} "
                  f"+{cand['bits_gained']}b  "
                  f"{cand['text'].strip()[:64]!r}")
        if stream_to is not None:       # committed — emit it now, not at the end
            print(cand["text"], end="", flush=True, file=stream_to)
        tokens.extend(cand["tokens"])
        consumed = cand["consumed"]
        logits = cand["logits"]
        if cand["eos"] or not cand["n"]:
            break

    return tok.decode(tokens[offset:]), stats


def _generate_and_verify(args, bits, codec, stream_to=None):
    """Generate and check the payload, printing progress. Mirrors the same helper
    Under --clean the caller decides whether this output is shown; the exit code
    still reports verification either way."""
    model, tok = backend().load(args.model)
    prompt = build_prompt(tok, args.topic, args.think)
    text, stats = generate(
        model, tok, prompt, bits, key=args.key, attempts=args.attempts,
        temperature=args.temperature, bit_bonus=args.bit_bonus,
        skip_penalty=args.skip_penalty, topk=args.topk,
        seed=args.seed, verbose=True, stream_to=stream_to,
        budget=Budget(max_shocks=args.max_shocks, max_coined=args.max_coined,
                      max_caps=args.max_caps))
    if stream_to is None:               # streaming already showed it live
        print("\n" + "=" * 60)
        print(text.strip())
    ok = verify(text, bits, args.key)
    encoding_summary(text, bits, args.key)
    span = payload_span(text, len(bits), args.key)
    if span:
        print(f"[stego] payload region: {len(bits) / span:.4f} bits/char "
              f"({span} of {len(text)} chars carry it; the rest finishes the sentence)")
    print(f"[stego] sentences: {stats['accepted']} accepted, "
          f"{stats['rejected']} rejected, {stats['fallback']} fallback")
    if codec is not None and ok:
        recovered = codec.decompress_from_bits(extract(text, args.key))
        match = recovered == args.message
        print(f"[codec] recovered message: {recovered!r}")
        print(f"[codec] {'MATCH — message round-trips' if match else 'MISMATCH'}")
        ok = match
    return text, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True, help="what the cover text is about")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--message", help="secret text to compress (Unishox2) and hide")
    src.add_argument("--bits", help="raw bitstream to hide instead, e.g. 10110")
    ap.add_argument("--model", default="mlx-community/SmolLM3-3B-8bit",
                    help="mlx-community model id (8-bit / less-peaky models give "
                         "better constrained fluency)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--skip-penalty", type=float, default=1.0,
                    help="logit penalty per off-bucket-letter (SKIP) character")
    ap.add_argument("--bit-bonus", type=float, default=1.0,
                    help="logit bonus per bit a token would encode, added before "
                         "sampling (raises density at the cost of fluency)")
    ap.add_argument("--topk", type=int, default=0,
                    help="constrain only the N highest-logit tokens each step "
                         "instead of the whole vocabulary. 2048 keeps 99.3%% of the "
                         "post-mask probability mass and cuts the per-step "
                         "constraint from ~33ms to ~2ms (0 = whole vocabulary)")
    ap.add_argument("--attempts", type=int, default=6,
                    help="regenerations allowed per sentence before taking the best")
    ap.add_argument("--max-shocks", type=int, default=1,
                    help="tokens far above the model's expectation tolerated per sentence")
    ap.add_argument("--max-coined", type=int, default=1,
                    help="invented (non-dictionary) words tolerated per sentence")
    ap.add_argument("--max-caps", type=float, default=0.35,
                    help="capitalised-word rate ceiling; catches title-register drift")
    ap.add_argument("--think", action="store_true",
                    help="allow the model to emit <think> reasoning (default off; thinking "
                         "models like SmolLM3/Qwen3 otherwise encode a reasoning block as garbage)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", default=None,
                    help="secret key: shuffles the letter->bit alphabet per character; the "
                         "recipient needs the same key to decode (omit for the fixed alphabet)")
    ap.add_argument("--clean", action="store_true",
                    help="print only the cover text — no progress, no verification. The "
                         "exit code still reports whether the payload verified.")
    args = ap.parse_args()

    codec = None
    if args.message is not None:
        try:
            import payload_codec as codec
        except ModuleNotFoundError as exc:      # the Unishox2 wheel is an extra
            raise SystemExit(
                f"[codec] --message needs the Unishox2 codec, but {exc.name!r} is missing.\n"
                f"  install it:  pip install unishox2-py3\n"
                f"  or hide a raw bitstream instead with --bits") from None
        bits = codec.compress_to_bits(args.message)
        if not args.clean:
            print(f"[codec] {codec.ratio_report(args.message, bits)}")
    else:
        bits = parse_bits(args.bits)
        if not bits:
            ap.error("--bits must contain at least one 0 or 1")

    # --clean streams each sentence to the real console as it is committed, and
    # muffles everything else. Sentences are the unit here: a rejected one is
    # regenerated, so nothing can be emitted until it has been accepted.
    with _muffled(args.clean) as console:
        text, ok = _generate_and_verify(args, bits, codec,
                                        stream_to=console if args.clean else None)
    if args.clean:
        print()                         # terminate the streamed line
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
