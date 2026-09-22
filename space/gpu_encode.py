"""GPU-resident batched encoder, for ZeroGPU / CUDA Spaces.

The CPU path in sentence_encode.py regenerates a rejected sentence up to
`attempts` times *sequentially*. Measured on real runs, ~82% of sampled tokens
belong to attempts that are later discarded, so almost all of the wall clock is
spent on work that runs one row at a time.

Those attempts are independent, so on a GPU they are one batch:

    cache.batch_repeat_interleave(N)   fan a batch-1 prefix out to N rows
    ... sample N sentences in parallel, each with its own payload position ...
    cache.batch_select_indices([w])    keep the winner's KV, drop the rest
    cache.crop(-(steps - n_w))         trim the winner's surplus tokens

`batch_select_indices` also removes the rollback/replay logic entirely: the
winner's KV is already in place, so nothing has to be recomputed.

Everything per-step lives on the device -- roles, masking, penalties, anti-repeat
and nucleus sampling -- and only the chosen token ids come back. That matters
because the constraint is O(batch) work: at batch 32 the CPU version would cost
~64 ms/step against a ~7 ms GPU step, so batching without moving the constraint
would simply relocate the bottleneck.

Hardware (ZeroGPU, 2026): NVIDIA RTX Pro 6000 Blackwell, 96 GB GDDR7, 1.79 TB/s.
`large` (the default) is half a card -- 48 GB, 1x quota; `xlarge` is the full
card -- 96 GB, 2x quota. For a 3B model the KV cache is ~72 KiB/token, so batch
is never memory-bound; see `autosize_batch`.
"""
from __future__ import annotations

import numpy as np
import torch

import sentence_encode as se


# --------------------------------------------------------------------------- #
# batch sizing
# --------------------------------------------------------------------------- #
def model_geometry(model) -> dict:
    c = model.config
    heads = getattr(c, "num_attention_heads", 1)
    kv = getattr(c, "num_key_value_heads", heads) or heads
    hd = getattr(c, "head_dim", None) or (getattr(c, "hidden_size", 0) // max(heads, 1))
    return {"layers": getattr(c, "num_hidden_layers", 0), "kv_heads": kv,
            "head_dim": hd, "vocab": getattr(c, "vocab_size", 0)}


def autosize_batch(model, seq_budget=1024, reserve=0.30, cap=16, floor=4,
                   verbose=True) -> int:
    """Rows to run in parallel.

    Decode is memory-*bandwidth* bound: the weights are read once per step no
    matter how many rows ride along, so extra rows are nearly free until the
    step turns compute-bound. For a 3B model on this hardware that crossover is
    far above any batch we would want, and the KV cache is tiny, so VRAM is not
    the binding constraint -- `cap` is. It exists because best-of-N stops paying
    off long before the GPU saturates, and because every row still costs one
    tokenizer decode and one bookkeeping pass on the CPU each step.
    """
    g = model_geometry(model)
    esz = 2 if model.dtype in (torch.float16, torch.bfloat16) else 4
    kv_per_tok = 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * esz
    # per row: KV for the whole sequence, plus a logits row and the (V, max_c)
    # role scratch the constraint allocates per row
    per_row = kv_per_tok * seq_budget + g["vocab"] * 4 * 8
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
    else:
        free = total = 8 << 30
    fits = int((free * (1.0 - reserve)) // max(per_row, 1))
    n = max(floor, min(cap, fits))
    if verbose:
        print(f"[gpu] {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB | "
              f"KV {kv_per_tok/1024:.0f} KiB/token | {per_row/2**20:.1f} MiB/row | "
              f"fits {fits} -> batch {n}", flush=True)
    return n


# --------------------------------------------------------------------------- #
# device-resident constraint
# --------------------------------------------------------------------------- #
class GpuConstraint:
    """se.Encoder's constraint, evaluated for N rows at once on the device.

    Each row carries its own payload position and character position, so the
    keyed alphabet differs per row; the whole thing is still one set of batched
    gathers. Semantics follow Encoder.constrain exactly -- wrong-bit and junk
    tokens to -inf, skip characters priced at `skip_penalty`.
    """

    def __init__(self, tok, bits, skip_penalty, key, device, max_pos=4096,
                 vocab_size=None):
        base = se.Encoder(tok, bits, skip_penalty, key)
        # must match the model's logits width, which can exceed len(tok)
        base._build(int(vocab_size or len(tok)))
        self.device, self.nbits = device, len(bits)
        self.skip_penalty = float(skip_penalty)
        self.max_c = base.max_c
        self.junk_np = base.junk

        d = device
        self.clipped = torch.as_tensor(base._clipped.astype(np.int64)).to(d)   # (V,C)
        self.letter = torch.as_tensor(base._letter_at).to(d)                   # (V,C)
        self.fixed = torch.as_tensor(base._fixed.astype(np.int64)).to(d)       # (V,C)
        self.junk = torch.as_tensor(base.junk).to(d)                           # (V,)
        self.bits_pad = torch.as_tensor(
            np.append(np.asarray(bits, dtype=np.int64), -1)).to(d)             # (nbits+1,)
        # position -> letter-role table, precomputed once. _letter_roles is a
        # SHA-256 schedule on the CPU; hoisting it out of the step loop is what
        # lets a step be pure tensor work.
        self.max_pos = max_pos
        table = np.stack([se._letter_roles(key, p) for p in range(max_pos)])    # (P,26)
        self.pos_role = torch.as_tensor(table.astype(np.int64)).to(d)
        self.V = self.clipped.shape[0]

    def ensure_pos(self, need: int, key):
        if need + self.max_c < self.max_pos:
            return
        grow = int(need + self.max_c + 1024)
        table = np.stack([se._letter_roles(key, p) for p in range(grow)])
        self.pos_role = torch.as_tensor(table.astype(np.int64)).to(self.device)
        self.max_pos = grow

    def __call__(self, consumed: torch.Tensor, char_pos: torch.Tensor,
                 logits: torch.Tensor):
        """consumed/char_pos: (N,) int64. logits: (N,V) float.
        Returns (masked (N,V), bits_enc (N,V) int64)."""
        N, V, C = logits.shape[0], self.V, self.max_c
        done = consumed >= self.nbits

        # (N, C, 26) keyed roles for each row's next C character positions
        offs = torch.arange(C, device=self.device).unsqueeze(0)                # (1,C)
        rows = (char_pos.unsqueeze(1) + offs).clamp_(0, self.max_pos - 1)      # (N,C)
        rmap = self.pos_role[rows]                                             # (N,C,26)

        # letters -> their live role; everything else keeps its build-time role.
        # Advanced indexing rather than gather-on-an-expanded-view: the latter
        # would broadcast a (N,V,C,26) shape, 26x more elements than the answer.
        n_ix = torch.arange(N, device=self.device).view(N, 1, 1)
        c_ix = torch.arange(C, device=self.device).view(1, 1, C)
        role = rmap[n_ix, c_ix, self.clipped.unsqueeze(0)]                     # (N,V,C)
        role = torch.where(self.letter.unsqueeze(0), role, self.fixed.unsqueeze(0))

        enc = (role == 0) | (role == 1)
        cum = torch.cumsum(enc.long(), dim=2) - enc.long()                     # bit offset
        bit_pos = consumed.view(N, 1, 1) + cum
        within = enc & (bit_pos < self.nbits)
        target = self.bits_pad[bit_pos.clamp(0, self.nbits)]
        bad = (within & (role != target)).any(dim=2) | self.junk.unsqueeze(0)

        out = logits - self.skip_penalty * (role == se.SKIP).sum(dim=2)
        out = out.masked_fill(bad, float("-inf"))
        bits_enc = within.sum(dim=2).masked_fill(bad, 0)

        # rows whose payload is already spent are unconstrained but still may not
        # emit junk -- matching Encoder.constrain's rem <= 0 branch
        if done.any():
            free = logits.masked_fill(self.junk.unsqueeze(0), float("-inf"))
            out = torch.where(done.view(N, 1), free, out)
            bits_enc = torch.where(done.view(N, 1), torch.zeros_like(bits_enc), bits_enc)
        return out, bits_enc


# --------------------------------------------------------------------------- #
# sampling, on device
# --------------------------------------------------------------------------- #
def sample_rows(scores, temperature, top_p, gen):
    """Nucleus sample one token per row from (N,V) scores containing -inf."""
    if temperature <= 1e-6:
        return scores.argmax(dim=-1)
    z = scores / temperature
    probs = torch.softmax(z.float(), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    if 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, dim=-1, descending=True)
        keep = (torch.cumsum(sp, dim=-1) - sp) < top_p
        keep[..., 0] = True
        probs = torch.zeros_like(probs).scatter_(-1, si, sp * keep)
    tot = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(tot > 0, probs / tot.clamp_min(1e-30),
                        torch.zeros_like(probs))
    dead = (probs.sum(dim=-1) <= 0)
    if dead.any():                       # fully cornered row: fall back to argmax
        probs[dead] = 0.0
        probs[dead, scores[dead].argmax(dim=-1)] = 1.0
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


def antirepeat_rows(scores, hist, freq_penalty, window, no_repeat_ngram):
    """Per-row frequency penalty and n-gram ban, applied in place on (N,V)."""
    if freq_penalty > 0:
        for i, h in enumerate(hist):
            if not h:
                continue
            ids, counts = np.unique(np.asarray(h[-window:]), return_counts=True)
            t_ids = torch.as_tensor(ids, device=scores.device, dtype=torch.long)
            t_cnt = torch.as_tensor(counts, device=scores.device, dtype=scores.dtype)
            row = scores[i]
            finite = torch.isfinite(row[t_ids])
            row[t_ids] = torch.where(finite, row[t_ids] - freq_penalty * t_cnt,
                                     row[t_ids])
    n = no_repeat_ngram
    if n and n >= 1:
        for i, h in enumerate(hist):
            if len(h) < n - 1:
                continue
            prefix = tuple(h[-(n - 1):]) if n > 1 else ()
            banned = [h[j + n - 1] for j in range(len(h) - n + 1)
                      if tuple(h[j:j + n - 1]) == prefix]
            if banned:
                scores[i, torch.as_tensor(sorted(set(banned)),
                                          device=scores.device)] = float("-inf")
    return scores


# --------------------------------------------------------------------------- #
# batched sentence generation
# --------------------------------------------------------------------------- #
class _Row:
    __slots__ = ("sent", "toks", "consumed", "start", "ex", "done", "eos",
                 "complete", "logits")

    def __init__(self, consumed):
        self.sent, self.toks, self.consumed = "", [], consumed
        self.start = consumed
        self.ex, self.done, self.eos, self.complete = [], False, False, False
        self.logits = None

    def verdict(self, budget, vocab):
        exa = np.array(self.ex) if self.ex else np.array([0.0])
        coined = len(se.coined_words(self.sent, vocab))
        caps = se.register_oddity(self.sent)
        shocks = int((exa > se.SHOCK_NATS).sum())
        worst = float(exa.max())
        gained = self.consumed - self.start
        # A sentence that carries no payload bit makes no progress, so it cannot
        # be "acceptable" however clean it reads. Without this a row that emits a
        # bare "." scores zero shocks, zero coined words and a low worst token,
        # and therefore wins the round -- best-of-N finds that degenerate optimum
        # immediately, where the CPU path hid it by stopping at the first
        # acceptable candidate rather than the best one.
        ok = ((self.complete or self.eos) and gained > 0
              and budget.verdict(shocks, worst, coined, caps))
        return {"shocks": shocks, "worst": worst, "coined": coined, "caps": caps,
                "ok": ok, "n": len(self.toks), "gained": gained}


def generate_batched(model, tok, prompt, bits, *, key=None, batch=8,
                     temperature=0.9, top_p=0.95, bit_bonus=1.0, skip_penalty=1.0,
                     max_sentence_chars=200, budget=None, seed=0, device="cuda",
                     verbose=False, max_rounds=40):
    """Encode `bits`, generating `batch` candidate sentences per round in parallel.

    One round replaces what the CPU path does in up to `batch` sequential passes.
    The winner's KV is kept with batch_select_indices, so no sentence is ever
    replayed."""
    from transformers import DynamicCache

    budget = budget or se.Budget()
    vocab = se._VOCAB or se.load_vocab()
    con = GpuConstraint(tok, bits, skip_penalty, key, device,
                        vocab_size=model.config.vocab_size)
    nbits = len(bits)
    eos_ids = sorted(se._eos_ids(tok))
    pad_id = eos_ids[0] if eos_ids else 0
    gen = torch.Generator(device=device).manual_seed(int(seed))

    cache = DynamicCache(config=model.config)
    if hasattr(cache, "activate_past_recording"):
        cache.activate_past_recording()
    tokens, offset = list(prompt), len(prompt)
    with torch.inference_mode():
        logits1 = model(input_ids=torch.tensor([list(prompt)], device=device),
                        past_key_values=cache, use_cache=True).logits[:, -1, :]

    consumed, stats = 0, {"accepted": 0, "rejected": 0, "fallback": 0, "rounds": 0,
                          "steps": 0}
    eos_t = torch.as_tensor(eos_ids, device=device, dtype=torch.long) if eos_ids else None

    while consumed < nbits and stats["rounds"] < max_rounds:
        stats["rounds"] += 1
        N = int(batch)
        base_chars = len(tok.decode(tokens[offset:]))
        con.ensure_pos(base_chars + max_sentence_chars + 8, key)
        cache.batch_repeat_interleave(N)
        logits = logits1.expand(N, -1).contiguous()
        rows = [_Row(consumed) for _ in range(N)]
        steps = 0
        max_steps = max(8, max_sentence_chars)     # chars >= tokens, so this bounds it

        with torch.inference_mode():
            while steps < max_steps and not all(r.done for r in rows):
                raw = logits.float()
                lse = torch.logsumexp(raw, dim=-1)
                logp = raw - lse.unsqueeze(-1)
                H = -(logp.exp() * logp).sum(dim=-1)

                cons_t = torch.tensor([r.consumed for r in rows], device=device)
                cpos_t = torch.tensor([base_chars + len(r.sent) for r in rows],
                                      device=device)
                masked, bits_enc = con(cons_t, cpos_t, raw)

                scores = antirepeat_rows(masked.clone(),
                                         [tokens[offset:] + r.toks for r in rows],
                                         freq_penalty=0.5, window=64,
                                         no_repeat_ngram=3)
                scores = scores + bit_bonus * bits_enc.to(scores.dtype)
                if eos_t is not None:               # never stop mid-payload
                    mid = (cons_t < nbits)
                    if mid.any():
                        blk = scores[mid]
                        blk[:, eos_t] = float("-inf")
                        scores[mid] = blk

                choice = sample_rows(scores, temperature, top_p, gen)
                ex = (lse - raw.gather(1, choice.unsqueeze(1)).squeeze(1)) - H
                gained = bits_enc.gather(1, choice.unsqueeze(1)).squeeze(1)

                ch = choice.tolist()
                exl = ex.tolist()
                gl = gained.tolist()
                # Whether a row was finished BEFORE this step. A row that becomes
                # done *during* this step still appended a token, and that token
                # must be fed like any other or the cache ends up holding a pad
                # where the sentence's last character belongs.
                was_done = [r.done for r in rows]
                appended = []
                for i, r in enumerate(rows):
                    if was_done[i]:
                        continue
                    if ch[i] in eos_ids:
                        r.eos = r.done = r.complete = True
                        continue
                    r.ex.append(exl[i])
                    r.toks.append(ch[i])
                    r.sent += tok.decode([ch[i]])
                    r.consumed += int(gl[i])
                    appended.append(i)
                    if exl[i] > budget.hard_abort:
                        r.done = True                      # aborted: not complete
                    elif ((se._sentence_done(r.sent)
                           and se._can_end(r.sent, len(r.toks)))
                          or len(r.sent) >= max_sentence_chars):
                        r.complete = r.done = True

                # Feed each row's own token; only rows that were already finished
                # (or that just chose EOS, which they do not keep) get the pad.
                live = torch.zeros(N, dtype=torch.bool, device=device)
                if appended:
                    live[torch.tensor(appended, device=device)] = True
                feed = torch.where(live, choice,
                                   torch.full_like(choice, pad_id)).unsqueeze(1)
                logits = model(input_ids=feed, past_key_values=cache,
                               use_cache=True).logits[:, -1, :]
                # The next round continues from the distribution that FOLLOWS the
                # committed token, so this has to be captured after the forward.
                # Capturing it before is an off-by-one: the round would resume by
                # re-predicting the token it just emitted, which shows up as
                # lowercase sentence starts and doubled punctuation at the seam.
                for i in appended:
                    rows[i].logits = logits[i]
                steps += 1
        stats["steps"] += steps

        # ---- pick the winner, exactly as the CPU path ranks candidates --------
        vs = [r.verdict(budget, vocab) for r in rows]
        accepted = [i for i, v in enumerate(vs) if v["ok"] and rows[i].toks]
        def rank_fallback(i):
            v, r = vs[i], rows[i]
            return (not (r.complete or r.eos), v["gained"] == 0,
                    v["caps"] > budget.max_caps, v["coined"], v["shocks"],
                    v["worst"], -v["gained"])

        def rank_accepted(i):
            # every one of these already cleared the quality budget, so choose on
            # progress first and use quality only to break ties
            v = vs[i]
            return (-v["gained"], v["coined"], v["shocks"], v["worst"])

        if accepted:
            w = min(accepted, key=rank_accepted); stats["accepted"] += 1
        else:
            w = min(range(N), key=rank_fallback); stats["fallback"] += 1
        stats["rejected"] += N - 1

        cache.batch_select_indices(torch.tensor([w], device=device))
        surplus = steps - len(rows[w].toks)
        if surplus > 0:
            cache.crop(-surplus)
        if verbose:
            v = vs[w]
            print(f"  [{'ok ' if v['ok'] else 'FB '}] shocks={v['shocks']} "
                  f"worst={v['worst']:5.1f} coined={v['coined']} "
                  f"+{rows[w].consumed - consumed}b  {rows[w].sent.strip()[:60]!r}",
                  flush=True)

        if not rows[w].toks:
            break
        tokens.extend(rows[w].toks)
        consumed = rows[w].consumed
        logits1 = rows[w].logits.unsqueeze(0)
        if rows[w].eos:
            break

    return tok.decode(tokens[offset:]), stats
