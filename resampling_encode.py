#!/usr/bin/env python3
"""
resampling_encode.py — sample-then-select, instead of per-token search.

`simple_encode.py` spends its compute *inside* each step: it shortlists
`--candidates` tokens and rolls each one `--rollout-depth` steps forward, costing
candidates x (1 + depth) forward passes per token (48 at the 12/3 defaults). That
buys a locally-good token but, as measured, it also collapses selection to a near
argmax (~1.3 effective choices) — so the search is expensive *and* barely explores.

This file spends the same budget differently: generate whole paragraphs cheaply
(one forward pass per token, sampling straight from the constrained distribution),
do it N times, then score the finished samples and keep the best. At N=36 that is
still under the 48-pass-per-token budget, and it has two structural advantages:

  * scoring sees the *finished* text, so it can measure what actually matters —
    the worst word in the paragraph, not the average token;
  * nothing is committed greedily, so a bad paragraph is discarded rather than
    patched, and the sampling stays a real sample of the constrained distribution.

Metrics per sample:
  fluency      worst-case, not mean: out-of-dictionary ("coined") words, plus the
               p95/max per-token excess surprise. Averages hide the one garbled
               word a reader notices, which is exactly how earlier tuning went wrong.
  density      bits per character of the PAYLOAD REGION only. Dividing by the whole
               text counts the free tail and understates density roughly 2x.
  detectability mean/std of per-token excess surprise, and the residual against the
               genuine temperature manifold (std vs mean) — the temperature-invariant
               over-dispersion signal.

    python resampling_encode.py --topic "the sea" --bits 10110100 --key demo
"""

from __future__ import annotations

import argparse
import sys

# simple_encode sets the backend-quieting env vars at import time, so it has to be
# imported before anything pulls in transformers.
import simple_encode as se

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# Genuine temperature manifold, fitted over SmolLM3 samples at T 0.7-1.7:
#   std(excess) ~= MANIFOLD_A * mean(excess) + MANIFOLD_B
# The residual off this line is the over-dispersion an attacker can test for even
# when they are allowed to fit any decoding temperature. --calibrate refits it.
MANIFOLD_A, MANIFOLD_B = 0.832, 1.298
# The fit was measured over T 0.7-1.7, i.e. mean excess in roughly this band. Outside
# it the residual is an extrapolation and can be gamed: a sample at mean +4.4 once
# scored a near-genuine residual while reading as word-salad.
MANIFOLD_VALID = (-0.8, 3.6)
DICT_PATH = "/usr/share/dict/words"

# A token this far above the model's own expectation reads as a jolt. Genuine text
# from this model tops out near 3 nats, so 5 is comfortably outside it.
SHOCK_NATS = 5.0


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
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
        if se.char_role(ch, i, key) in se.BIT_BUCKETS:
            n += 1
            if n >= nbits:
                return i + 1
    return None


def token_excess(model, tok, prompt_ids, text: str) -> np.ndarray:
    """Per-token excess surprise (NLL - entropy) under the base model. ~0 means the
    token was as surprising as the model expected; large positive is a shock."""
    try:
        gen = tok.encode(text, add_special_tokens=False)
    except TypeError:
        gen = tok.encode(text)
    if len(gen) < 3:
        return np.array([0.0])
    full = list(prompt_ids) + list(gen)
    logits = model(mx.array(full)[None])[0]
    mx.eval(logits)
    g0, out = len(prompt_ids), []
    for j in range(len(gen)):
        row = np.array(logits[g0 + j - 1].astype(mx.float32))
        m = row.max()
        e = np.exp(row - m)
        logp = row - m - np.log(e.sum())
        p = e / e.sum()
        out.append(-float(logp[full[g0 + j]]) - float(-(p * logp).sum()))
    return np.array(out)


def score_sample(model, tok, prompt_ids, text, bits, key, vocab) -> dict:
    ex = token_excess(model, tok, prompt_ids, text)
    span = payload_span(text, len(bits), key)
    got = se.extract(text, key)[:len(bits)]
    coined = coined_words(text, vocab)
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    return {
        # how *pervasive* the jolts are, as opposed to how bad the single worst one
        # is: word-salad is many moderate shocks, not one spike.
        "shock_rate": float((ex > SHOCK_NATS).mean()),
        "caps_rate": register_oddity(text),
        "word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "space_rate": text.count(" ") / max(len(text), 1),
        "resid_valid": MANIFOLD_VALID[0] <= float(ex.mean()) <= MANIFOLD_VALID[1],
        "text": text,
        "recovered": got == list(bits),
        "chars": len(text),
        "density": (len(bits) / span) if span else 0.0,     # payload region only
        "coined": len(coined),
        "coined_rate": len(coined) / max(len(words), 1),
        "coined_words": coined[:4],
        "worst_excess": float(ex.max()),
        "p95_excess": float(np.percentile(ex, 95)),
        "mean_excess": float(ex.mean()),
        "std_excess": float(ex.std()),
        "residual": float(ex.std() - (MANIFOLD_A * ex.mean() + MANIFOLD_B)),
    }


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def cheap_sample(model, tok, prompt, bits, *, seed, key, args) -> str:
    """One paragraph, one forward pass per token: rollout_depth=0 disables the
    lookahead search entirely, so this samples the constrained distribution
    directly rather than re-ranking a shortlist toward its argmax."""
    return se.steer_generate(
        model, tok, prompt, bits,
        max_tokens=se.token_budget(len(bits), args.tail_chars),
        tail_chars=args.tail_chars,
        temperature=args.temperature,
        skip_penalty=args.skip_penalty,
        num_candidates=1,
        rollout_depth=0,                 # <- the whole point: no per-token search
        bit_bonus=args.bit_bonus,
        top_p=se.TOP_P,
        freq_penalty=se.FREQ_PENALTY,
        rep_window=se.REP_WINDOW,
        no_repeat_ngram=se.NO_REPEAT_NGRAM,
        seed=seed,
        key=key,
    )


def genuine_sample(model, tok, prompt, *, seed, temperature, max_new) -> str:
    """Unconstrained text from the same prompt — the reference the metrics are
    judged against (a coined-word count means nothing without one)."""
    rng = np.random.default_rng(seed)
    eos = set(getattr(tok, "eos_token_ids", None) or
              ([tok.eos_token_id] if getattr(tok, "eos_token_id", None) is not None else []))
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    out = []
    for _ in range(max_new):
        c = se._sample_logits(se._vec(logits), temperature, rng, se.TOP_P)
        if c in eos:
            break
        out.append(c)
        logits = model(mx.array([[c]]), cache=cache)[:, -1, :]
        mx.eval(logits)
    return tok.decode(out)


RANKERS = {
    # fluency first: what a reader actually rejects
    # coherence first: invented words, then how pervasive the jolts are, then register
    "fluency": lambda r: (r["coined_rate"], r["shock_rate"], r["caps_rate"], r["worst_excess"]),
    "density": lambda r: (-r["density"], r["coined_rate"]),
    "detect":  lambda r: (abs(r["residual"]), r["coined_rate"]),
    "balanced": lambda r: (r["coined_rate"] + abs(r["residual"]) / 4 - r["density"], ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--message")
    src.add_argument("--bits")
    ap.add_argument("--model", default="mlx-community/SmolLM3-3B-8bit")
    ap.add_argument("--samples", type=int, default=36,
                    help="paragraphs to generate and score (36 ~ the cost of the "
                         "12-candidate/3-depth search this replaces)")
    ap.add_argument("--calibrate", type=int, default=3,
                    help="unconstrained samples generated as the reference baseline")
    ap.add_argument("--rank", choices=sorted(RANKERS), default="fluency")
    ap.add_argument("--tail-chars", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--skip-penalty", type=float, default=1.0)
    ap.add_argument("--bit-bonus", type=float, default=1.0,
                    help="logit reward per encoded bit. Measured sweep: 2.0 suppresses "
                         "word breaks (a space earns no bonus) and inflates word length "
                         "to 6.2 vs 4.9 genuine; 1.0 lands on the genuine space rate at "
                         "only ~18%% less density")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", default=None)
    ap.add_argument("--show", type=int, default=8, help="rows of the ranking to print")
    ap.add_argument("--clean", action="store_true", help="print only the winning text")
    args = ap.parse_args()

    codec = None
    if args.message is not None:
        try:
            import payload_codec as codec
        except ModuleNotFoundError as exc:
            raise SystemExit(f"--message needs {exc.name!r}: pip install unishox2-py3") from None
        bits = codec.compress_to_bits(args.message)
    else:
        bits = se.parse_bits(args.bits)
        if not bits:
            ap.error("--bits must contain at least one 0 or 1")

    log = (lambda *a: None) if args.clean else print
    with se._muffled(args.clean):
        model, tok = load(args.model)
        prompt = se.build_prompt(tok, args.topic, think=False)
        vocab = load_vocab()
        if not vocab:
            log(f"[warn] no dictionary at {DICT_PATH}; coined-word counts disabled")

        # --- reference baseline ------------------------------------------------
        ref = []
        for i in range(args.calibrate):
            g = genuine_sample(model, tok, prompt, seed=10_000 + i,
                               temperature=args.temperature, max_new=160)
            ex = token_excess(model, tok, prompt, g)
            words = [w for w in g.split() if any(c.isalpha() for c in w)]
            ref.append({"coined_rate": len(coined_words(g, vocab)) / max(len(words), 1),
                        "worst_excess": float(ex.max()), "mean": float(ex.mean()),
                        "std": float(ex.std()),
                        "shock_rate": float((ex > SHOCK_NATS).mean()),
                        "caps_rate": register_oddity(g),
                        "word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
                        "space_rate": g.count(" ") / max(len(g), 1)})
            log(f"[ref {i+1}/{args.calibrate}] coined={ref[-1]['coined_rate']:.3f} "
                f"worst={ref[-1]['worst_excess']:.1f}")

        # --- candidate paragraphs ---------------------------------------------
        rows, seen = [], {}
        for i in range(args.samples):
            text = cheap_sample(model, tok, prompt, bits, seed=args.seed + i,
                                key=args.key, args=args)
            if text in seen:                     # identical sampling collapse
                seen[text] += 1
                log(f"[{i+1}/{args.samples}] DUPLICATE of sample {seen[text]}")
                continue
            seen[text] = i + 1
            r = score_sample(model, tok, prompt, text, bits, args.key, vocab)
            r["i"] = i
            rows.append(r)
            log(f"[{i+1}/{args.samples}] rec={r['recovered']!s:5} dens={r['density']:.3f} "
                f"coined={r['coined']:>2} worst={r['worst_excess']:5.1f} "
                f"resid={r['residual']:+.2f}")

    ok_rows = [r for r in rows if r["recovered"]]
    if not ok_rows:
        raise SystemExit("no sample carried the full payload — raise --samples or --tail-chars")
    ok_rows.sort(key=RANKERS[args.rank])
    best = ok_rows[0]

    if args.clean:
        print(best["text"])
        raise SystemExit(0)

    uniq = len(rows)
    print(f"\n{'='*78}\n{uniq}/{args.samples} distinct, {len(ok_rows)} carried the payload"
          f"   (ranked by {args.rank})\n{'='*78}")
    print(f"{'#':>3}{'dens':>7}{'wlen':>6}{'space':>7}{'coined':>7}{'shock':>7}"
          f"{'worst':>7}{'meanEx':>8}{'resid':>8}")
    for r in ok_rows[:args.show]:
        flag = "" if r["resid_valid"] else "!"   # ! = outside the calibrated band
        print(f"{r['i']:>3}{r['density']:>7.3f}{r['word_len']:>6.2f}{r['space_rate']:>7.3f}"
              f"{r['coined_rate']:>7.3f}{r['shock_rate']:>7.3f}{r['worst_excess']:>7.1f}"
              f"{r['mean_excess']:>+8.2f}{r['residual']:>+7.2f}{flag:>1}")
    if ref:
        m = lambda k: np.mean([x[k] for x in ref])
        print(f"{'ref':>3}{'--':>7}{m('word_len'):>6.2f}{m('space_rate'):>7.3f}"
              f"{m('coined_rate'):>7.3f}{m('shock_rate'):>7.3f}{m('worst_excess'):>7.1f}"
              f"{m('mean'):>+8.2f}"
              f"{m('std')-(MANIFOLD_A*m('mean')+MANIFOLD_B):>+7.2f}   <- unconstrained baseline")
    print(f"\nbest (#{best['i']}): density {best['density']:.3f} b/char, "
          f"{best['coined']} coined {best['coined_words']}, worst excess {best['worst_excess']:.1f}")
    print("-" * 78)
    print(best["text"].strip())
    raise SystemExit(0)


if __name__ == "__main__":
    main()
