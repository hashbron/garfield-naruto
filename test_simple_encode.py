#!/usr/bin/env python3
"""Standalone logic test for simple_encode.py — no model / GPU needed.

Stubs MLX + a mock tokenizer/model so the constraint, search, and cache logic can
be checked in milliseconds. Run:  python test_simple_encode.py
"""
import sys, types, numpy as np, random, io, contextlib

# --- stub mlx / mlx_lm so simple_encode imports without Apple-Silicon deps ---
mxc = types.ModuleType("mlx.core")
mxc.array = lambda x: np.asarray(x); mxc.float32 = np.float32; mxc.eval = lambda *a, **k: None
sys.modules["mlx"] = types.ModuleType("mlx"); sys.modules["mlx.core"] = mxc
sys.modules["mlx_lm"] = types.ModuleType("mlx_lm"); sys.modules["mlx_lm"].load = None
cmod = types.ModuleType("mlx_lm.models.cache")
class Cache:
    def __init__(self): self.seq = []
cmod.make_prompt_cache = lambda model, *a, **k: [Cache()]
cmod.trim_prompt_cache = lambda cache, n: (cache[0].seq.__delitem__(slice(-n, None)), n)[1]
cmod.can_trim_prompt_cache = lambda cache: True
sys.modules["mlx_lm.models"] = types.ModuleType("mlx_lm.models")
sys.modules["mlx_lm.models.cache"] = cmod

import simple_encode as s

def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

WORDS = ["the", "cat", "orange", "sea", "boy", "night", "dream", " the", " a", " and", ".", ", "]
LETTERS = [chr(c) for c in range(97, 123)]
VOCAB = WORDS + LETTERS + [" ", ".", ",", "\n", ""]
EOS = len(VOCAB) - 1
class Tok:
    eos_token_id = EOS
    def decode(self, ids):
        if isinstance(ids, int): ids = [ids]
        return "".join(VOCAB[i] for i in ids)
tok = Tok(); V = len(VOCAB)


def oracle(token_str, consumed, bits):
    """Ground-truth (forbidden, bits_encoded) by direct simulation."""
    bi = consumed
    for ch in token_str:
        bk = s.char_bucket(ch)
        if bk in (s.SKIP, s.SKIP_FREE):
            continue
        if bi >= len(bits):
            break
        if bk != bits[bi]:
            return True, 0
        bi += 1
    return False, bi - consumed


def test_constrain_vs_oracle():
    mism = checks = 0
    for trial in range(400):
        bits = [random.Random(trial).randint(0, 1) for _ in range(random.Random(trial + 1).randint(1, 12))]
        enc = s.Encoder(tok, bits, skip_penalty=1.0); enc._build(V)
        for consumed in range(len(bits) + 1):
            out, be = enc.constrain(consumed, np.zeros(V, np.float32))
            for i in range(V):
                checks += 1
                of, obe = oracle(VOCAB[i], consumed, bits)
                vf = bool(np.isneginf(out[i]))
                if vf != of or (not of and int(be[i]) != obe):
                    mism += 1
                    if mism <= 8:
                        print(f"  MISMATCH tok={VOCAB[i]!r} consumed={consumed} bits={bits} "
                              f"vec=({vf},{int(be[i])}) oracle=({of},{obe})")
    print(f"(1) constrain vs oracle: {checks} checks, mismatches={mism}")
    return mism == 0


def make_model(seed):
    r = random.Random(seed)
    def model(inputs, cache=None):
        ids = np.asarray(inputs).reshape(-1).tolist(); cache[0].seq.extend(ids)
        L = len(ids); out = np.zeros((1, L, V))
        row = np.array([r.random() * 2 for _ in range(V)])
        for i in range(len(WORDS)): row[i] += 2.0
        row[EOS] = -4.0
        out[0, -1] = row
        return out
    return model


def wrongbit(text, bits):
    for bi, b in enumerate(s.extract(text)):
        if bi < len(bits) and b != bits[bi]:
            return True
    return False


def test_roundtrip():
    fails = viol = 0; dens = []
    for t in range(120):
        bits = [random.Random(1000 + t).randint(0, 1) for _ in range(random.Random(t).randint(3, 10))]
        txt = quiet(s.steer_generate, make_model(t), tok, [VOCAB.index("the")], bits,
                    max_tokens=400, temperature=0.7, num_candidates=8, lookahead_weight=8.0,
                    bit_bonus=3.0, rollout_depth=1, seed=t)
        if s.extract(txt)[:len(bits)] != bits: fails += 1
        if wrongbit(txt, bits): viol += 1
        dens.append(min(len(bits), len(s.extract(txt))) / max(len(txt), 1))
    print(f"(2) round-trip x120: encode-fails={fails}  wrong-bit={viol}  "
          f"mean density={np.mean(dens):.3f} bits/char")
    return fails == 0 and viol == 0


def test_cache_neutral():
    caches = []
    orig = cmod.make_prompt_cache
    cmod.make_prompt_cache = lambda model, *a, **k: (lambda c: (caches.append(c) or c))(orig(model))
    bits = [1, 0, 1, 1, 0, 0, 1]
    txt = quiet(s.steer_generate, make_model(5), tok, [VOCAB.index("the")], bits,
                max_tokens=300, num_candidates=8, lookahead_weight=8.0, rollout_depth=2, seed=5)
    cmod.make_prompt_cache = orig
    ok = tok.decode(caches[0][0].seq[1:]) == txt and s.extract(txt)[:len(bits)] == bits
    print(f"(3) cache decodes to output exactly & encodes: {ok}")
    return ok


if __name__ == "__main__":
    print("bucket sample:", {c: s.char_bucket(c) for c in "the cat."})
    results = [test_constrain_vs_oracle(), test_roundtrip(), test_cache_neutral()]
    print("\nALL PASS" if all(results) else "\nFAILURES PRESENT")
    sys.exit(0 if all(results) else 1)
