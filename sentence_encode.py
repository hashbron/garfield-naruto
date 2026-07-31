#!/usr/bin/env python3
"""
sentence_encode.py — steganographic encoding, one sentence at a time.

Hides a bitstream in fluent cover text by constraining which characters the model
may emit: at each character position a character either carries the next payload
bit, or skips. The constraint machinery (keyed alphabet, vectorised whole-vocab
masking, payload framing) lives in simple_encode.py; this file adds the control
loop around it.

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

    python sentence_encode.py --topic "the sea" --message "meet at dawn" --key k
    python sentence_encode.py --topic "the sea" --bits 10110100 --key k --clean
"""
from __future__ import annotations

import argparse
import sys

import simple_encode as se          # sets the backend-quieting env vars on import

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

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
        if se.char_role(ch, i, key) in se.BIT_BUCKETS:
            n += 1
            if n >= nbits:
                return i + 1
    return None




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
                   skip_penalty, max_chars, allow_abort=True):
    """Generate one sentence. Returns a dict describing it; the caller decides
    whether to keep it. Leaves the cache advanced by `n_tokens` either way."""
    new, excesses, cur, hit_eos, complete = [], [], consumed, False, False
    sent = ""
    while True:
        raw = se._vec(logits)
        lse = se._logsumexp(raw)
        char_pos = len(tok.decode(tokens[offset:])) + len(sent)
        masked, bits_enc = enc.constrain(cur, char_pos, raw)
        if cur < enc.nbits:                    # never stop mid-payload
            for e in eos_ids:
                masked[e] = -np.inf
        base = se._antirepeat(masked.copy(), tokens[offset:] + new,
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
        choice = se._sample_logits(base, temperature, rng, top_p)
        ex = (lse - float(raw[choice])) - H
        excesses.append(ex)

        if choice in eos_ids:
            hit_eos = True
            break
        s = tok.decode([choice])
        new.append(choice)
        sent += s
        cur += int(bits_enc[choice])
        clog = model(mx.array([[choice]]), cache=cache)[:, -1, :]
        mx.eval(clog)
        logits = clog
        # Early abort. Stop at the *budget*, not just at catastrophe: once the
        # sentence has already blown its shock allowance it cannot be accepted, so
        # every further token is paid for and then discarded.
        # Only a catastrophic token stops the sentence early. Aborting on the shock
        # *budget* was tried and reverted: a truncated sentence accumulates fewer
        # shocks and so outscored a finished one, so fragments won the ranking and
        # were committed mid-word ("sp" + "ikit" -> "spikit").
        if allow_abort and ex > budget.hard_abort:
            break                                  # aborted: `complete` stays False
        if _sentence_done(sent) or len(sent) >= max_chars:
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


def generate(model, tok, prompt, bits, *, key=None, attempts=6, temperature=0.9,
             top_p=0.95, bit_bonus=1.0, skip_penalty=1.0,
             max_sentence_chars=200, budget=None, seed=0, verbose=False,
             stream_to=None):
    """Encode `bits`, accepting one sentence at a time."""
    global _VOCAB
    if not _VOCAB:
        _VOCAB = load_vocab()
    budget = budget or Budget()
    enc = se.Encoder(tok, bits, skip_penalty, key)
    rng = np.random.default_rng(seed)
    eos_ids = set(getattr(tok, "eos_token_ids", None) or
                  ([tok.eos_token_id] if getattr(tok, "eos_token_id", None) is not None else []))

    tokens, offset = list(prompt), len(prompt)
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    enc._build(logits.shape[1])
    for e in eos_ids:
        if e < enc.junk.size:
            enc.junk[e] = False

    consumed = 0
    stats = {"accepted": 0, "rejected": 0, "fallback": 0}
    while consumed < enc.nbits:
        best = None
        def attempt(allow_abort=True):
            return _grow_sentence(
                model, tok, enc, cache, logits, tokens=tokens, offset=offset,
                consumed=consumed, rng=rng, temperature=temperature, top_p=top_p,
                bit_bonus=bit_bonus, eos_ids=eos_ids, budget=budget,
                skip_penalty=skip_penalty, max_chars=max_sentence_chars,
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
                break
            stats["rejected"] += 1
            if cand["n"]:                       # roll the cache back and retry
                trim_prompt_cache(cache, cand["n"])
        cand = best[1]
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
                clog = model(mx.array([[t]]), cache=cache)[:, -1, :]
                mx.eval(clog)
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
    in simple_encode.py so both scripts behave identically under --clean: the
    caller decides whether this output is shown, and the exit code still reports
    verification either way."""
    model, tok = load(args.model)
    prompt = se.build_prompt(tok, args.topic, args.think)
    text, stats = generate(
        model, tok, prompt, bits, key=args.key, attempts=args.attempts,
        temperature=args.temperature, bit_bonus=args.bit_bonus,
        skip_penalty=args.skip_penalty,
        seed=args.seed, verbose=True, stream_to=stream_to,
        budget=Budget(max_shocks=args.max_shocks, max_coined=args.max_coined,
                      max_caps=args.max_caps))
    if stream_to is None:               # streaming already showed it live
        print("\n" + "=" * 60)
        print(text.strip())
    ok = se.verify(text, bits, args.key)
    se.encoding_summary(text, bits, args.key)
    span = payload_span(text, len(bits), args.key)
    if span:
        print(f"[stego] payload region: {len(bits) / span:.4f} bits/char "
              f"({span} of {len(text)} chars carry it; the rest finishes the sentence)")
    print(f"[stego] sentences: {stats['accepted']} accepted, "
          f"{stats['rejected']} rejected, {stats['fallback']} fallback")
    if codec is not None and ok:
        recovered = codec.decompress_from_bits(se.extract(text, args.key))
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
                    help="score reward per bit a candidate encodes (raises density "
                         "vs fluency; 2.0 suppresses word breaks, see the sweep)")
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
        bits = se.parse_bits(args.bits)
        if not bits:
            ap.error("--bits must contain at least one 0 or 1")

    # --clean streams each sentence to the real console as it is committed, and
    # muffles everything else. Sentences are the unit here: a rejected one is
    # regenerated, so nothing can be emitted until it has been accepted.
    with se._muffled(args.clean) as console:
        text, ok = _generate_and_verify(args, bits, codec,
                                        stream_to=console if args.clean else None)
    if args.clean:
        print()                         # terminate the streamed line
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
