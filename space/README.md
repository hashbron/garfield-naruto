---
title: Garfield-Naruto Encoder
emoji: 🐈
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: mit
short_description: Hide a short message inside LLM-generated cover text.
---

# Garfield-Naruto Encoder

Hides a short message inside ordinary-looking LLM prose. Every character of the
output is an encoding slot: a secret key decides, per character position, which
letters carry a `0`, which carry a `1`, and which carry nothing.

**Decoding needs only the text and the key — no model, no GPU.** The Decode tab
runs entirely in the Space process and costs no GPU quota.

## GPU quota

This Space runs on **ZeroGPU**, where GPU time is charged to **the visitor's own
account**, not to whoever published the Space. Quota is per user, per day.

| | daily ZeroGPU quota |
|---|---|
| not identified | ~2 minutes, tracked by IP and shared |
| free account | ~5 minutes |
| PRO / Team / Enterprise | 40+ minutes, and pay-as-you-go credits past that |

**There is no sign-in button, and you do not need one.** ZeroGPU identifies you
from the `x-ip-token` header that the Hugging Face proxy injects using your
huggingface.co session, so simply opening this Space in a browser where you are
signed in is what puts the time on your own account. An app-level OAuth button
(`gr.LoginButton` / `hf_oauth`) is a separate mechanism that tells the *app* who
you are without changing your quota, so one here would only be misleading.

Two situations fall back to the shared IP pool: not being signed in to
huggingface.co, and running this Space **embedded in a frame on another site** —
browsers restrict cookies in cross-site frames, so the proxy cannot see your
session. Open the Space directly to avoid both.

## Warning

This is a research prototype. It should not be used for any privacy-critical
purpose. Measured against a model-equipped adversary the scheme is **not**
indistinguishable from ordinary model output.

## Performance

Hardware: ZeroGPU runs **NVIDIA RTX Pro 6000 Blackwell** (96 GB GDDR7, 1.79 TB/s).
`large` — the default — is half a card: **48 GB, 1× quota**. `xlarge` is the full
card: 96 GB, 2× quota. This workload never needs `xlarge`; see below.

Two optimisations, both measured:

**Top-K constraint** (`STEGO_TOPK`, default 2048). The constraint is elementwise
work over `vocab × max_token_chars` — 3.1M elements per step for a 128k
vocabulary. Restricting it to the highest-logit tokens cuts that ~63×. Measured
over 884 real steps, the legal tokens inside a 2048 window still hold **99.3%**
of the post-mask probability mass (5th pct 96.9%), with a median of 406 legal
tokens and no step ever starved. Per-step constraint cost drops 33 ms → 2.1 ms;
end-to-end **1.6× faster**.

**Batched sentences** (`STEGO_BATCH`, default `auto`). The CPU path regenerates a
rejected sentence sequentially, and ~82% of sampled tokens belong to attempts
that are later discarded. Those attempts are independent, so on a GPU they run as
one batch: `batch_repeat_interleave` fans the prefix out, rows sample in
parallel, `batch_select_indices` keeps the winner's KV — which also removes the
rollback/replay logic entirely. Decode is memory-*bandwidth* bound, so extra rows
are close to free until the step turns compute-bound.

**Why batch is not memory-limited.** SmolLM3-3B is 36 layers with 4 KV heads and
head_dim 128, i.e. **72 KiB of KV per token**. Batch 64 at 2048 tokens is 9 GiB
against 48 GiB available, and the weights are ~6 GB. `autosize_batch` still
computes the fit from live `mem_get_info`, but in practice the cap binds first:
best-of-N stops paying off long before the GPU saturates, and every row costs a
tokenizer decode on the CPU each step.

`STEGO_BATCH=off` forces the sequential path; a number forces that batch size
even on CPU (useful for debugging, not for speed — CPU decode is compute-bound,
so rows cost proportionally there).

Not used: `torch.compile` is unsupported on ZeroGPU, since a fresh process is
spun up per GPU task and compilation cannot be reused. Ahead-of-time compilation
via `torch.export` + `spaces.aoti_*` is the supported route and is reported at
1.3–1.8×, but it targets a fixed-shape forward; this generator's shapes change
per step, so it is left as future work.
