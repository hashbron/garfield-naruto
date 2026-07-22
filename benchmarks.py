#!/usr/bin/env python3
"""
stego_bench.py — one-command detectability benchmark for the letter-stego scheme.

Scores text against a fixed, reproducible clean corpus with a battery of BLIND
detectors (they know nothing about your stride, vocab, or bit-mapping), then
aggregates each detector into an AUC:

    AUC = 0.50  -> indistinguishable from clean (the goal)
    AUC = 1.00  -> trivially detected

The target is indistinguishability from ordinary model output *including higher
temperatures*. So the reference corpus is a TEMPERATURE MIXTURE, and the table
prints clean high-temperature CONTROL rows: their headline is the floor a scheme
must reach. A scheme at or below the floor is indistinguishable from ordinary
temperature variation.

Run:
    python stego_bench.py                     # synthetic scheme battery + controls
    python stego_bench.py --dir texts/        # AUC of your text files vs the corpus
    python stego_bench.py --file one.txt      # anomaly scores for a single text

Detectors: periodicity (position x residue), marginal (residue distribution),
bigram (letter naturalness), comb (periodic surprisal; real log-probs via
--sur-dir, else a unigram surrogate).
"""

import argparse
import hashlib
import math
import os
import random

# ==========================================================================
# Statistics
# ==========================================================================

def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x)."""
    if x <= 0 or a <= 0:
        return 1.0
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1.0 - a, 1e300, 0.0
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(stat: float, df: int) -> float:
    return 1.0 if df <= 0 else _gammq(df / 2.0, stat / 2.0)


def _neglog10_sf(stat: float, df: int) -> float:
    p = chi2_sf(stat, df)
    return 300.0 if p <= 0 else -math.log10(p)


def _avg_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    return ranks


def auc_se(clean: list[float], stego: list[float]):
    """AUC that a higher score means 'stego', with Hanley-McNeil SE."""
    nc, nt = len(clean), len(stego)
    if nc == 0 or nt == 0:
        return 0.5, 0.0
    ranks = _avg_ranks(clean + stego)
    auc = (sum(ranks[nc:]) - nt * (nt + 1) / 2.0) / (nc * nt)
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (nt - 1) * (q1 - auc * auc)
           + (nc - 1) * (q2 - auc * auc)) / (nc * nt)
    return auc, math.sqrt(max(var, 0.0))


# ==========================================================================
# Reference corpus and synthetic schemes (letter-level generators)
# ==========================================================================

_FREQ = {
    'e': 12.702, 't': 9.056, 'a': 8.167, 'o': 7.507, 'i': 6.966, 'n': 6.749,
    's': 6.327, 'h': 6.094, 'r': 5.987, 'd': 4.253, 'l': 4.025, 'c': 2.782,
    'u': 2.758, 'm': 2.406, 'w': 2.360, 'f': 2.228, 'g': 2.015, 'y': 1.974,
    'p': 1.929, 'b': 1.492, 'v': 0.978, 'k': 0.772, 'j': 0.153, 'x': 0.150,
    'q': 0.095, 'z': 0.074,
}
_LETTERS = list(_FREQ)


def _weights(T: float):
    return [(_FREQ[c] / 100.0) ** (1.0 / T) for c in _LETTERS]


def gen_clean(n: int, T: float, rng: random.Random) -> str:
    return "".join(rng.choices(_LETTERS, weights=_weights(T), k=n))


def _residue_pools(k: int):
    mask = (1 << k) - 1
    pools: dict[int, tuple[list[str], list[float]]] = {}
    nat: dict[int, float] = {}
    for c in _LETTERS:
        r = ord(c) & mask
        pools.setdefault(r, ([], []))
        pools[r][0].append(c)
        pools[r][1].append(_FREQ[c])
        nat[r] = nat.get(r, 0.0) + _FREQ[c]
    return pools, nat


def make_scheme(k, positions, correct_freq, stride=5, base_T=1.0):
    """positions 'fixed'|'random'; correct_freq draws carrier residues from the
    natural residue distribution instead of uniform. Returns fn(rng, n)->text."""
    pools, nat = _residue_pools(k)
    res_ids = list(nat)
    nat_w = [nat[r] for r in res_ids]
    w = _weights(base_T)

    def gen(rng: random.Random, n: int) -> str:
        out = []
        for i in range(n):
            carrier = (i % stride == 0) if positions == "fixed" \
                else (rng.random() < 1.0 / stride)
            if carrier:
                target = (rng.choices(res_ids, weights=nat_w, k=1)[0]
                          if correct_freq else rng.getrandbits(k))
                pool = pools.get(target)
                if pool:
                    out.append(rng.choices(pool[0], weights=pool[1], k=1)[0])
                    continue
            out.append(rng.choices(_LETTERS, weights=w, k=1)[0])
        return "".join(out)

    return gen


def build_controls(ref_T):
    # clean text at the matched temperature (sanity ~0.5) and hotter (context)
    return {
        f"clean T={ref_T:g} (matched)": lambda r, n: gen_clean(n, ref_T, r),
        "clean T=1.4 (hotter)":         lambda r, n: gen_clean(n, 1.4, r),
        "clean T=2.0 (hotter)":         lambda r, n: gen_clean(n, 2.0, r),
    }


def build_schemes(base_T):
    return {
        "baseline k=1":         make_scheme(1, "fixed", False, base_T=base_T),
        "baseline k=2":         make_scheme(2, "fixed", False, base_T=base_T),
        "varied-pos k=2":       make_scheme(2, "random", False, base_T=base_T),
        "freq-corrected k=2":   make_scheme(2, "fixed", True, base_T=base_T),
        "varied+corrected k=2": make_scheme(2, "random", True, base_T=base_T),
    }


# ==========================================================================
# Blind detectors — each: sample (text, surprisals|None) -> suspicion score
# ==========================================================================

def residues(text: str, k: int) -> list[int]:
    m = (1 << k) - 1
    return [ord(c) & m for c in text if c.isalpha()]


def _chi2_bin_vs_rest(res, stride, phase, R):
    car, rst = [0] * R, [0] * R
    for i, r in enumerate(res):
        (car if i % stride == phase else rst)[r] += 1
    nc, nr = sum(car), sum(rst)
    if nc == 0 or nr == 0:
        return 0.0, 0
    N, stat, nz = nc + nr, 0.0, 0
    for j in range(R):
        colt = car[j] + rst[j]
        if colt == 0:
            continue
        nz += 1
        for tot, obs in ((nc, car[j]), (nr, rst[j])):
            e = tot * colt / N
            stat += (obs - e) ** 2 / e
    return stat, nz - 1


def det_periodicity(text, max_stride, max_k):
    best = 0.0
    for k in range(1, max_k + 1):
        res = residues(text, k)
        R = 1 << k
        for S in range(2, max_stride + 1):
            for ph in range(S):
                best = max(best, _neglog10_sf(*_chi2_bin_vs_rest(res, S, ph, R)))
    return best


def fit_marginal(texts, k):
    R = 1 << k
    counts = [1.0] * R
    for t in texts:
        for r in residues(t, k):
            counts[r] += 1
    tot = sum(counts)
    return [c / tot for c in counts]


def det_marginal(text, k, ref):
    R = 1 << k
    obs = [0] * R
    for r in residues(text, k):
        obs[r] += 1
    N = sum(obs)
    if N == 0:
        return 0.0
    stat = sum((obs[j] - ref[j] * N) ** 2 / (ref[j] * N) for j in range(R))
    return _neglog10_sf(stat, R - 1)


def fit_bigram(texts):
    idx = {c: i for i, c in enumerate(_LETTERS)}
    M = len(_LETTERS)
    counts = [[1.0] * M for _ in range(M)]
    for t in texts:
        ls = [c.lower() for c in t if c.isalpha() and c.lower() in idx]
        for a, b in zip(ls, ls[1:]):
            counts[idx[a]][idx[b]] += 1
    logp = [[math.log(counts[a][b] / sum(counts[a])) for b in range(M)]
            for a in range(M)]
    return idx, logp


def det_bigram(text, model):
    idx, logp = model
    ls = [c.lower() for c in text if c.isalpha() and c.lower() in idx]
    if len(ls) < 2:
        return 0.0
    return -sum(logp[idx[a]][idx[b]] for a, b in zip(ls, ls[1:])) / (len(ls) - 1)


def fit_surprisal(texts):
    counts: dict[str, int] = {}
    for t in texts:
        for c in t:
            if c.isalpha():
                counts[c.lower()] = counts.get(c.lower(), 0) + 1
    tot = sum(counts.values()) + len(_LETTERS)
    return ({c: -math.log((counts.get(c, 0) + 1) / tot) for c in _LETTERS},
            -math.log(1.0 / tot))


def _comb_pvalue(sur, stride, phase):
    car = [s for i, s in enumerate(sur) if i % stride == phase]
    rst = [s for i, s in enumerate(sur) if i % stride != phase]
    if len(car) < 2 or len(rst) < 2:
        return 1.0
    mc, mr = sum(car) / len(car), sum(rst) / len(rst)
    vc = sum((x - mc) ** 2 for x in car) / (len(car) - 1)
    vr = sum((x - mr) ** 2 for x in rst) / (len(rst) - 1)
    se = math.sqrt(vc / len(car) + vr / len(rst))
    z = (mc - mr) / se if se > 0 else 0.0
    return 0.5 * math.erfc(z / math.sqrt(2.0))      # one-sided carrier > rest


def det_comb(sample, smodel, default, max_stride):
    text, sur = sample
    letters = [c for c in text if c.isalpha()]
    s = sur if (sur is not None and len(sur) == len(letters)) \
        else [smodel.get(c.lower(), default) for c in letters]
    best = 0.0
    for S in range(2, max_stride + 1):
        for ph in range(S):
            p = _comb_pvalue(s, S, ph)
            best = max(best, 300.0 if p <= 0 else -math.log10(p))
    return best


def build_detectors(fit_texts, max_stride, max_k):
    ref = {k: fit_marginal(fit_texts, k) for k in range(1, max_k + 1)}
    bigram = fit_bigram(fit_texts)
    smodel, sdef = fit_surprisal(fit_texts)
    return {
        "periodicity": lambda s: det_periodicity(s[0], max_stride, max_k),
        "marginal":    lambda s: max(det_marginal(s[0], k, ref[k]) for k in ref),
        "bigram":      lambda s: det_bigram(s[0], bigram),
        "comb":        lambda s: det_comb(s, smodel, sdef, max_stride),
    }


DET_NAMES = ["periodicity", "marginal", "bigram", "comb"]


# ==========================================================================
# Corpus building (stable seeding — reproducible across processes)
# ==========================================================================

def _seed(master, tag, i):
    h = hashlib.sha256(f"{master}:{tag}:{i}".encode()).hexdigest()
    return int(h[:8], 16)


def gen_samples(gen, tag, master, m, n):
    return [(gen(random.Random(_seed(master, tag, i)), n), None) for i in range(m)]


def reference(master, m, n, ref_T):
    """Fixed fit/eval clean corpus at the matched temperature. Predictable per seed."""
    gen = lambda r, nn: gen_clean(nn, ref_T, r)
    return (gen_samples(gen, "ref-fit", master, m, n),
            gen_samples(gen, "ref-eval", master, m, n))


# ==========================================================================
# Scoring / reporting
# ==========================================================================

def auc_row(detectors, clean_eval, samples):
    return {name: auc_se([f(s) for s in clean_eval], [f(s) for s in samples])
            for name, f in detectors.items()}


def _headline(dets):
    """Worst-case detector: its AUC and SE (the one with the highest AUC)."""
    name = max(DET_NAMES, key=lambda d: dets[d][0])
    return dets[name]


def print_table(results):
    w = 24
    head = f"{'row':<{w}} | " + " | ".join(f"{d:^13}" for d in DET_NAMES) + f" | {'HEADLINE':^9}"
    print(head)
    print("-" * len(head))
    for label, dets in results.items():
        auc, se = _headline(dets)
        cells = " | ".join(f"{dets[d][0]:.3f}±{1.96*dets[d][1]:.2f}" for d in DET_NAMES)
        # indistinguishable = worst detector cannot beat chance at 95% confidence
        mark = "  indistinguishable" if auc - 1.96 * se <= 0.5 else ""
        print(f"{label:<{w}} | {cells} | {auc:^9.3f}{mark}")
    print("\nAUC 0.500 = indistinguishable (goal). HEADLINE = worst-case detector's AUC.")
    print("'indistinguishable' = even the worst detector's 95% CI includes 0.5.")
    print("periodicity & comb are temperature-invariant; marginal & bigram move with")
    print("temperature (see the hotter clean controls) — match --ref-temp to compare fairly.")


def score_one(text, detectors, clean_eval):
    """Anomaly z-scores for a single text vs the clean-eval score distribution."""
    print(f"{'detector':<14} {'score':>10} {'clean mean':>12} {'z':>7}")
    print("-" * 46)
    hi = 0.0
    for name, f in detectors.items():
        cs = [f(s) for s in clean_eval]
        mu = sum(cs) / len(cs)
        sd = (sum((c - mu) ** 2 for c in cs) / max(1, len(cs) - 1)) ** 0.5 or 1e-9
        val = f((text, None))
        z = (val - mu) / sd
        hi = max(hi, z)
        print(f"{name:<14} {val:>10.3f} {mu:>12.3f} {z:>7.2f}")
    print(f"\nheadline anomaly z = {hi:.2f}  (higher = more distinguishable from clean)")


def load_dir(path, sur_path=None):
    out = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if not os.path.isfile(fp):
            continue
        with open(fp, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        sur = None
        if sur_path and os.path.isfile(os.path.join(sur_path, fn)):
            with open(os.path.join(sur_path, fn)) as sf:
                sur = [float(x) for x in sf if x.strip()]
        out.append((text, sur))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=60, help="samples per side")
    ap.add_argument("--n", type=int, default=1500, help="letters per synthetic sample")
    ap.add_argument("--seed", type=int, default=0, help="corpus seed (predictable)")
    ap.add_argument("--ref-temp", type=float, default=1.0,
                    help="reference/clean temperature (match your generation temp)")
    ap.add_argument("--max-stride", type=int, default=8)
    ap.add_argument("--max-k", type=int, default=2)
    ap.add_argument("--clean-dir", help="real clean corpus (else synthetic reference)")
    ap.add_argument("--dir", help="directory of candidate text files -> AUC vs corpus")
    ap.add_argument("--sur-dir", help="optional per-letter -log p files (real comb)")
    ap.add_argument("--file", help="single text file -> anomaly z-scores")
    ap.add_argument("--check", action="store_true", help="run twice, assert identical")
    args = ap.parse_args()

    if args.clean_dir:                          # real clean corpus, split fit/eval
        clean = load_dir(args.clean_dir)
        h = len(clean) // 2
        fit, ev = clean[:h], clean[h:]
    else:
        fit, ev = reference(args.seed, args.samples, args.n, args.ref_temp)
    detectors = build_detectors([s[0] for s in fit], args.max_stride, args.max_k)

    if args.file:
        with open(args.file, encoding="utf-8", errors="ignore") as f:
            print(f"[single file] {args.file} vs corpus\n")
            score_one(f.read(), detectors, ev)
        return

    if args.dir:
        cand = load_dir(args.dir, args.sur_dir)
        print(f"[dir] {len(cand)} files vs corpus; comb={'real' if args.sur_dir else 'surrogate'}\n")
        print_table({"candidates": auc_row(detectors, ev, cand)})
        return

    controls, schemes = build_controls(args.ref_temp), build_schemes(args.ref_temp)
    results = {name: auc_row(detectors, ev, gen_samples(gen, name, args.seed,
                                                        args.samples, args.n))
               for name, gen in {**controls, **schemes}.items()}
    print(f"clean corpus @ T={args.ref_temp:g} vs controls + schemes — "
          f"{args.samples}/side, {args.n} letters\n")
    print_table(results)

    if args.check:
        fit2, ev2 = reference(args.seed, args.samples, args.n, args.ref_temp)
        d2 = build_detectors([s[0] for s in fit2], args.max_stride, args.max_k)
        r2 = {n: auc_row(d2, ev2, gen_samples(g, n, args.seed, args.samples, args.n))
              for n, g in {**controls, **schemes}.items()}
        same = all(abs(results[s][d][0] - r2[s][d][0]) < 1e-12
                   for s in results for d in DET_NAMES)
        print(f"\n[reproducibility] identical across two runs: {same}")


if __name__ == "__main__":
    main()