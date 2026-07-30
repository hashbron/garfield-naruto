#!/usr/bin/env python3
"""payload_codec.py — compress a text message to a hidden bitstream and back.

Uses Unishox2 (short-string compression), so a secret message costs far fewer
payload bits than its raw bytes — which means far less stego cover text. The
bitstream is self-delimiting via a compact varint header, so it can be recovered
from the (usually longer) bit sequence read out of a cover text, ignoring any
trailing "free tail" bits the model produced after the payload.

    pip install unishox2-py3

Frame layout (bytes, then packed MSB-first into bits):
    varint(original_char_count) | varint(compressed_len) | compressed_bytes
"""
from __future__ import annotations

import unishox2


# --------------------------------------------------------------------------- #
# bit <-> byte packing (MSB first)
# --------------------------------------------------------------------------- #
def bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]


def bits_to_bytes(bits: list[int]) -> bytes:
    n = len(bits) - len(bits) % 8                      # drop any partial trailing byte
    return bytes(sum(bits[i + j] << (7 - j) for j in range(8)) for i in range(0, n, 8))


# --------------------------------------------------------------------------- #
# LEB128-style varints (1 byte for lengths < 128, keeps the header tiny)
# --------------------------------------------------------------------------- #
def _write_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = data[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def compress_to_bits(message: str) -> list[int]:
    """Text -> Unishox2 -> framed, self-delimiting list of 0/1 ready for the encoder."""
    if not message:
        raise ValueError("message must be non-empty")
    compressed, original_size = unishox2.compress(message)
    frame = _write_varint(original_size) + _write_varint(len(compressed)) + bytes(compressed)
    return bytes_to_bits(frame)


# Unishox2's `original_size` argument is the size of the C output buffer, not a
# format field. Overestimating is harmless; UNDERestimating overruns the buffer and
# corrupts the heap (measured: a 143-char message decompressed with size=50 dies on
# SIGTRAP, size=5 on SIGSEGV). The header here is recovered from cover text, so a
# wrong key or a corrupted paragraph supplies arbitrary values — which makes a
# crash reachable from ordinary user input. Size the buffer defensively instead.
# A fixed generous buffer beats deriving one from the header: the header is the
# very thing that may be corrupt, so any formula over it can still under-size.
# Payloads here are at most a few hundred bytes, so 1 MiB is free insurance.
_DECOMP_BUF = 1 << 20


def decompress_from_bits(bits: list[int]) -> str:
    """Inverse of compress_to_bits; trailing stego 'free tail' bits are ignored.

    Raises ValueError on a malformed frame rather than trusting it — the frame
    comes out of the cover text, so it is untrusted input."""
    data = bits_to_bytes(bits)
    try:
        original_size, i = _read_varint(data, 0)
        comp_len, i = _read_varint(data, i)
    except IndexError:
        raise ValueError("bitstream too short to hold a payload header") from None
    compressed = bytes(data[i:i + comp_len])
    if len(compressed) < comp_len:
        raise ValueError(
            f"bitstream truncated: header wants {comp_len} compressed bytes, "
            f"only {len(compressed)} present (wrong key, or the cover text was edited)")
    if original_size > _DECOMP_BUF:
        raise ValueError(f"header declares {original_size} chars, above the "
                         f"{_DECOMP_BUF}-byte decompression limit")
    return unishox2.decompress(compressed, _DECOMP_BUF)


def ratio_report(message: str, bits: list[int]) -> str:
    """One-line summary of how much Unishox2 bought us."""
    raw_bits = len(message.encode()) * 8
    return (f"{len(message)} chars ({raw_bits} raw bits) -> {len(bits)} payload bits "
            f"({raw_bits / max(len(bits), 1):.2f}x smaller)")
