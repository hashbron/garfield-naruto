"""Garfield-Naruto encoder as a ZeroGPU Space.

ZeroGPU bills GPU time to the *visitor's* account rather than to whoever
published the Space, and quota is per user per day.

There is deliberately no sign-in button. ZeroGPU identifies a visitor from the
`x-ip-token` header the Hugging Face proxy injects using their huggingface.co
session, so quota already follows whoever opens the Space. An app-level OAuth
button (`gr.LoginButton` / `hf_oauth`) is a different mechanism: it would tell
*this app* who the user is without changing their quota at all. Having one
implied otherwise, so it was removed.

Encoding needs the model, so it runs inside @spaces.GPU. Decoding needs only the
text and the key, so it stays outside and costs no quota at all.
"""
import os
import time

import gradio as gr

try:                                  # no-op locally, real allocator on Spaces
    import spaces
except ImportError:                   # pragma: no cover
    class _Shim:
        @staticmethod
        def GPU(*a, **k):
            def deco(f):
                return f
            return deco if not (a and callable(a[0])) else a[0]
    spaces = _Shim()

os.environ.setdefault("STEGO_BACKEND", "torch")
import sentence_encode as se
import payload_codec as pc

MODEL_ID = os.environ.get("STEGO_MODEL", "HuggingFaceTB/SmolLM3-3B")
DEVICE = os.environ.get("STEGO_DEVICE", "cuda")
MAX_MESSAGE_CHARS = int(os.environ.get("STEGO_MAX_CHARS", "160"))
# The website demo exposes Attempts only, so temperature is fixed here to
# sentence_encode's default rather than surfaced as a control.
TEMPERATURE = float(os.environ.get("STEGO_TEMPERATURE", "0.9"))


# ZeroGPU wants the model on cuda at module level: a CUDA emulation layer is
# active outside @spaces.GPU, and placements made at startup are far cheaper
# than moving weights inside the GPU function.
se.set_backend(se.torch_backend(device=DEVICE))
MODEL, TOK = se.backend().load(MODEL_ID)
se._VOCAB = se.load_vocab()

# Batched generation: a round produces BATCH candidate sentences in one set of
# forward passes instead of up to `attempts` sequential ones. Decode is
# memory-bandwidth bound, so the extra rows are close to free on a GPU; on CPU
# they are not, so it stays off there.
import gpu_encode as ge

_want = os.environ.get("STEGO_BATCH", "auto")
if _want == "off":
    BATCH = 0
elif _want == "auto":
    # auto-enable only on a GPU: on CPU decode is compute-bound, so extra rows
    # cost proportionally and batching is a loss
    BATCH = ge.autosize_batch(MODEL, seq_budget=1024, cap=16) \
        if DEVICE.startswith("cuda") else 0
else:
    BATCH = int(_want)          # explicit override wins, so the batched path
                                # stays exercisable off-GPU
# Warm the per-tokenizer constraint tables now, at import, rather than letting
# the first request pay ~0.7s of vocabulary indexing *inside* its GPU allocation
# — that time is billed to the visitor's quota while the GPU sits idle.
se.Encoder(TOK, [0], 1.0, None)._build(MODEL.config.vocab_size)

print(f"[stego] model={MODEL_ID} device={DEVICE} topk={os.environ.get('STEGO_TOPK','2048')} "
      f"batch={BATCH or 'off (sequential)'} | constraint tables warmed", flush=True)


def _nbits(message: str) -> int:
    try:
        return len(pc.compress_to_bits(message or ""))
    except Exception:
        return 8 * len((message or "").encode())


# Cost per sampled token: one forward pass plus the whole-vocab constraint. The
# constraint alone measures ~31 ms on CPU for a 128k vocabulary, and it runs
# while the GPU is allocated, so it counts against quota even though the GPU is
# idle for it. 50 ms/step is that plus a forward, rounded up.
# With TOPK on, the constraint costs ~2 ms/step instead of ~33 ms, so a step is
# dominated by the forward pass. Measured 1.6x end-to-end from TOPK alone.
TOPK = int(os.environ.get("STEGO_TOPK", "2048"))
SECONDS_PER_STEP = float(os.environ.get("STEGO_SECONDS_PER_STEP", "0.03"))
STEPS_PER_BIT = 3.4          # sequential path: counts rejected sentences
STEPS_PER_BIT_BATCHED = 1.5  # batched path: rejects run alongside, not after


def estimate_duration(message, topic, key, attempts):
    """Seconds of GPU to request. Cost scales with payload length, not with the
    topic. Asking for less improves queue priority for everyone, so this is
    proportional rather than a flat maximum."""
    n = _nbits(message)
    # Batching collapses the attempts into one set of forward passes, so the
    # *sequential* step count no longer scales with `attempts`.
    steps = (STEPS_PER_BIT_BATCHED * n if BATCH
             else STEPS_PER_BIT * n * (float(attempts) / 6.0))
    return float(min(240.0, 20.0 + SECONDS_PER_STEP * steps))


@spaces.GPU(duration=estimate_duration)
def encode(message, topic, key, attempts):
    message = (message or "").strip()
    if not message:
        return "", "Enter a message to hide."
    if len(message) > MAX_MESSAGE_CHARS:
        return "", (f"Message is {len(message)} characters; the cap is "
                    f"{MAX_MESSAGE_CHARS} so a single run stays inside a "
                    f"reasonable slice of your daily GPU quota.")
    if not (key or "").strip():
        return "", "A key is required. The same key is needed to decode."

    bits = pc.compress_to_bits(message)
    prompt = se.build_prompt(TOK, (topic or "the sea").strip(), think=False)
    t0 = time.time()
    seed = int(time.time()) & 0xFFFF
    if BATCH:
        text, stats = ge.generate_batched(
            MODEL, TOK, prompt, bits, key=key, batch=BATCH,
            temperature=TEMPERATURE, seed=seed, device=DEVICE)
    else:
        text, stats = se.generate(
            MODEL, TOK, prompt, bits, key=key, attempts=int(attempts),
            temperature=TEMPERATURE, seed=seed, topk=TOPK)
    dt = time.time() - t0

    ok = se.verify(text, bits, key)
    try:
        ok = ok and pc.decompress_from_bits(se.extract(text, key)) == message
    except Exception:
        ok = False
    span = se.payload_span(text, len(bits), key)
    note = (
        f"{'Verified — round-trips to the original message.' if ok else 'FAILED to verify; try another run or raise attempts.'}\n"
        f"{len(bits)} payload bits in {len(text)} characters "
        f"({len(bits) / max(len(text), 1):.3f} bits/char"
        + (f", {len(bits) / span:.3f} over the payload span" if span else "") + ")\n"
        f"sentences: {stats['accepted']} accepted, {stats['rejected']} rejected, "
        f"{stats['fallback']} fallback"
        + (f", {stats['rounds']} batched rounds of {BATCH}" if BATCH else "")
        + f" — {dt:.0f}s of GPU"
    )
    return text.strip(), note


def decode(text, key):
    """No model, no GPU — just the text and the key."""
    if not (text or "").strip():
        return "Paste the cover text."
    if not (key or "").strip():
        return "A key is required."
    try:
        return pc.decompress_from_bits(se.extract(text, key))
    except Exception as exc:
        return f"Could not decode: {exc}\n\n(Wrong key, or the cover text was edited.)"


CSS = """
#gn-out, #gn-decode-out textarea { font-family: ui-monospace, Menlo, Consolas, monospace; }
"""

with gr.Blocks(title="Garfield-Naruto Encoder") as demo:
    gr.Markdown("# Garfield-Naruto Encoder")
    gr.Markdown(
        "> **Prototype.** Not for privacy-critical use — against an adversary "
        "who has the model this is *not* indistinguishable from ordinary output."
    )

    with gr.Tab("Encode"):
        gr.Markdown("## Encode a message")
        gr.Markdown(
            "Enter your secret message below and G-N Encoder will hide it in LLM "
            "slop. Topic allows you to choose what the slop is about and Key is a "
            "random seed for the encoding (think of it like a password required "
            "for decoding)."
        )
        msg = gr.Textbox(label="Message",
                         placeholder="Enter your secret message here.")
        topic = gr.Textbox(label="Topic",
                           placeholder="Choose a topic for the LLM slop.")
        key_in = gr.Textbox(label="Key", placeholder="Pick a secret key.",
                            type="password")
        attempts = gr.Number(label="Attempts", value=6, minimum=1, maximum=10,
                             precision=0)
        go = gr.Button("Begin Encoding", variant="primary")

        out = gr.Textbox(label="Output", elem_id="gn-out", lines=10,
                         placeholder="Output will appear here…")
        note = gr.Markdown()
        go.click(encode, [msg, topic, key_in, attempts], [out, note])

if __name__ == "__main__":
    # css moved from the Blocks constructor to launch() in Gradio 6
    demo.launch(css=CSS)
