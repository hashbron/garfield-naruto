#!/usr/bin/env python3
"""Standalone logic test for sentence_encode.py — no model / GPU needed.

Stubs MLX + a mock tokenizer/model so the constraint, sentence loop and cache logic can
be checked in milliseconds. Run:  python test_sentence_encode.py
"""
import sys, types, numpy as np, random, io, contextlib

# --- stub mlx / mlx_lm so sentence_encode imports without Apple-Silicon deps ---
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

import sentence_encode as s

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


def oracle(token_str, consumed, char_pos, bits, key):
    """Ground-truth (forbidden, bits_encoded) by direct simulation under `key`."""
    bi = consumed
    for k, ch in enumerate(token_str):
        r = s.char_role(ch, char_pos + k, key)
        if r == s.FORBIDDEN:
            return True, 0
        if r in (s.SKIP, s.SKIP_FREE):
            continue
        if bi >= len(bits):
            break
        if r != bits[bi]:
            return True, 0
        bi += 1
    return False, bi - consumed


def test_constrain_vs_oracle():
    mism = checks = 0
    for key in (None, "secret-key", "another"):
        for trial in range(60):
            bits = [random.Random(trial).randint(0, 1) for _ in range(random.Random(trial + 1).randint(1, 12))]
            enc = s.Encoder(tok, bits, skip_penalty=1.0, key=key); enc._build(V)
            for consumed in range(len(bits) + 1):
                for char_pos in (0, 1, 7, 13):
                    out, be = enc.constrain(consumed, char_pos, np.zeros(V, np.float32))
                    for i in range(V):
                        checks += 1
                        of, obe = oracle(VOCAB[i], consumed, char_pos, bits, key)
                        vf = bool(np.isneginf(out[i]))
                        if vf != of or (not of and int(be[i]) != obe):
                            mism += 1
                            if mism <= 8:
                                print(f"  MISMATCH key={key} tok={VOCAB[i]!r} consumed={consumed} "
                                      f"pos={char_pos} bits={bits} vec=({vf},{int(be[i])}) oracle=({of},{obe})")
    print(f"(1) constrain vs oracle (keyed): {checks} checks, mismatches={mism}")
    return mism == 0


def test_keyed_roundtrip():
    fails = wrong_key_differs = 0
    for t in range(40):
        bits = [random.Random(9000 + t).randint(0, 1) for _ in range(random.Random(t).randint(4, 10))]
        key = f"key-{t}"
        txt, _ = quiet(s.generate, make_model(t), tok, [VOCAB.index("the")], bits,
                       key=key, attempts=3, temperature=0.7, seed=t)
        if s.extract(txt, key)[:len(bits)] != bits:
            fails += 1
        if s.extract(txt, key + "X")[:len(bits)] != bits:   # wrong key -> garbage (usually)
            wrong_key_differs += 1
    print(f"(4) keyed round-trip x40: right-key-fails={fails}  wrong-key-differs={wrong_key_differs}/40")
    return fails == 0


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
        txt, _ = quiet(s.generate, make_model(t), tok, [VOCAB.index("the")], bits,
                       attempts=3, temperature=0.7, seed=t)
        if s.extract(txt)[:len(bits)] != bits: fails += 1
        if wrongbit(txt, bits): viol += 1
        dens.append(min(len(bits), len(s.extract(txt))) / max(len(txt), 1))
    print(f"(2) round-trip x120: encode-fails={fails}  wrong-bit={viol}  "
          f"mean density={np.mean(dens):.3f} bits/char")
    return fails == 0 and viol == 0


def test_cache_neutral():
    """The KV cache must hold exactly the committed text and nothing else.

    A rejected sentence is rolled back with trim_prompt_cache; an accepted one is
    left in place. Committing a candidate other than the accepted one would leave
    a sentence in the cache that never appears in the output, so every later token
    would be conditioned on text the reader never sees. This asserts it does not."""
    caches = []
    # the cache is opened through the backend shim, so patch the live backend
    b = s.backend()
    orig = b.new_cache
    b.new_cache = lambda model, *a, **k: (lambda c: (caches.append(c) or c))(orig(model))
    ok = True
    for t in range(12):
        caches.clear()
        bits = [random.Random(500 + t).randint(0, 1) for _ in range(random.Random(t).randint(4, 12))]
        txt, _ = quiet(s.generate, make_model(t), tok, [VOCAB.index("the")], bits,
                       attempts=4, temperature=0.8, seed=t)
        ok &= tok.decode(caches[0][0].seq[1:]) == txt
        ok &= s.extract(txt)[:len(bits)] == bits
    b.new_cache = orig
    print(f"(3) cache holds exactly the committed text, x12: {ok}")
    return ok


if __name__ == "__main__":
    print("role sample:", {c: s.char_role(c, 0, None) for c in "the cat."})
    results = [test_constrain_vs_oracle(), test_roundtrip(), test_cache_neutral(),
               test_keyed_roundtrip()]
    print("\nALL PASS" if all(results) else "\nFAILURES PRESENT")
    sys.exit(0 if all(results) else 1)
