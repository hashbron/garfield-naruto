/* ==================================================================== *
 * Keyed stego decoder — mirrors sentence_encode.py's extract().
 *
 * The key schedule uses nothing but SHA-256 so Python and JavaScript derive
 * byte-identical letter roles. Unishox2 decompression is NOT implemented here:
 * unishox2.js is loaded before this file and provides it as a global.
 * ==================================================================== */

/* ---- minimal synchronous SHA-256 (WebCrypto is async, awkward per-char) ---- */
const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function sha256Bytes(bytes) {
  const ml = bytes.length;
  const withPad = new Uint8Array((((ml + 8) >> 6) + 1) << 6);
  withPad.set(bytes);
  withPad[ml] = 0x80;
  const bitLen = ml * 8;
  new DataView(withPad.buffer).setUint32(withPad.length - 4, bitLen >>> 0);
  new DataView(withPad.buffer).setUint32(withPad.length - 8, Math.floor(bitLen / 4294967296));

  const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(64);
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));

  for (let off = 0; off < withPad.length; off += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = (withPad[off + i * 4] << 24) | (withPad[off + i * 4 + 1] << 16) |
             (withPad[off + i * 4 + 2] << 8) | withPad[off + i * 4 + 3];
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + K256[i] + w[i]) >>> 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0;
      d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0;
    H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0;
    H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
  }
  const out = new Uint8Array(32);
  for (let i = 0; i < 8; i++) {
    out[i * 4] = H[i] >>> 24; out[i * 4 + 1] = (H[i] >>> 16) & 255;
    out[i * 4 + 2] = (H[i] >>> 8) & 255; out[i * 4 + 3] = H[i] & 255;
  }
  return out;
}

const utf8 = (s) => new TextEncoder().encode(s);

/* ---- scheme constants — must match sentence_encode.py exactly ---- */
const BIT0 = 0, BIT1 = 1, SKIP = 2, SKIP_FREE = 3, FORBIDDEN = 4;
const ALPHA = "abcdefghijklmnopqrstuvwxyz";
const FREE_CHARS = new Set(
  " .,;:!?'\"()-" + "0123456789" + "‘’“”–—…"
);
// English letter frequencies, in a-z order. The greedy pass below uses these to
// keep each role at ~1/3 of natural letter mass, so the constraint does not skew
// the letter distribution of the cover text.
const LFREQ = [8.17, 1.49, 2.78, 4.25, 12.70, 2.23, 2.02, 6.09, 6.97, 0.15,
               0.77, 4.03, 2.41, 6.75, 7.51, 1.93, 0.10, 5.99, 6.33, 9.06,
               2.76, 0.98, 2.36, 0.15, 1.97, 0.07];

/* ---- key schedule ---- */
function letterOrder(key, pos) {
  // 26 sort keys of 4 bytes each = 104 bytes, so 4 SHA-256 blocks.
  const stream = new Uint8Array(128);
  for (let b = 0; b < 4; b++) {
    stream.set(sha256Bytes(utf8(`${key}|${pos}|${b}`)), b * 32);
  }
  const idx = [];
  for (let i = 0; i < 26; i++) {
    const o = i * 4;
    const v = ((stream[o] << 24) | (stream[o + 1] << 16) |
               (stream[o + 2] << 8) | stream[o + 3]) >>> 0;
    idx.push([v, i]);
  }
  // ties broken by letter index so the order is fully determined
  idx.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return idx.map((p) => p[1]);
}

function buildRoles(key, pos, unkeyed) {
  // Greedy: walk the letters in the keyed order, each one joining whichever role
  // currently carries the least frequency mass. Reshuffles *membership* every
  // position, so no letter is durably tied to one role.
  const order = unkeyed
    ? Array.from({ length: 26 }, (_, i) => i)
        .sort((a, b) => (LFREQ[b] - LFREQ[a]) || (a - b))
    : letterOrder(key, pos);
  const sums = [0, 0, 0];
  const role = new Array(26);
  for (const L of order) {
    let r = 0;
    if (sums[1] < sums[r]) r = 1;
    if (sums[2] < sums[r]) r = 2;
    role[L] = r;
    sums[r] += LFREQ[L];
  }
  return role;
}

// Unkeyed fallback (key omitted): the fixed greedy split, matching _BASE_ROLE.
const BASE_ROLE = buildRoles(null, 0, true);

const roleCache = new Map();
function letterRoles(key, pos) {
  if (key === null || key === undefined || key === '') return BASE_ROLE;
  const ck = `${key}|${pos}`;
  let v = roleCache.get(ck);
  if (v === undefined) { v = buildRoles(key, pos, false); roleCache.set(ck, v); }
  return v;
}

/* ---- character classification — mirrors char_group / char_role ---- */
function letterIndex(ch) {
  const c = ch.toLowerCase();
  let i = ALPHA.indexOf(c);
  if (i >= 0) return i;
  // fold accents the way Python's unicodedata.normalize("NFKD") does
  for (const base of c.normalize("NFKD")) {
    i = ALPHA.indexOf(base);
    if (i >= 0) return i;
  }
  return -1;
}

function isAlpha(ch) {
  // Python's str.isalpha() is Unicode-aware; \p{L} is the JS equivalent.
  return /\p{L}/u.test(ch);
}

function charRole(ch, pos, key) {
  if (!isAlpha(ch)) return FREE_CHARS.has(ch) ? SKIP_FREE : FORBIDDEN;
  const i = letterIndex(ch);
  if (i < 0) return FORBIDDEN;          // non-Latin letter: never emitted
  return letterRoles(key, pos)[i];
}

/* ---- decoding ---- */
function decodeBits(text, key = null) {
  const bits = [];
  for (let pos = 0; pos < text.length; pos++) {
    const r = charRole(text[pos], pos, key);
    if (r === BIT0 || r === BIT1) bits.push(r);
  }
  return bits;
}

// Pack MSB-first, matching payload_codec.bits_to_bytes.
function bitsToBytes(bits) {
  const n = bits.length - (bits.length % 8);
  const out = new Uint8Array(n / 8);
  for (let i = 0; i < n; i += 8) {
    let b = 0;
    for (let j = 0; j < 8; j++) b = (b << 1) | bits[i + j];
    out[i / 8] = b;
  }
  return out;
}

// Unwrap the frame: varint(original_size) varint(compressed_len) body. The frame
// is self-delimiting, so the free tail the encoder appends after the payload is
// read as bits and then simply sliced off here.
function decodeFrame(text, key = null) {
  const data = bitsToBytes(decodeBits(text, key));
  let i = 0;
  const varint = () => {
    let n = 0, shift = 0;
    for (;;) {
      if (i >= data.length) throw new Error("truncated payload frame");
      const b = data[i++];
      n |= (b & 0x7f) << shift;
      if (!(b & 0x80)) return n;
      shift += 7;
    }
  };
  const originalSize = varint();
  const compressedLen = varint();
  if (data.length - i < compressedLen) {
    throw new Error(
      `frame wants ${compressedLen} compressed bytes, only ${data.length - i} present ` +
      `(wrong key, or the text was edited)`
    );
  }
  return { originalSize, compressed: data.slice(i, i + compressedLen) };
}

/**
 * Full recovery: cover text -> hidden message.
 *
 * Decompression comes from unishox2.js, loaded as a classic script before this
 * one, so its functions are globals. Note the length convention differs from
 * Python: unishox2_decompress_simple takes the *compressed* length, while the
 * Python decompress() takes the *original* size.
 *
 * Cross-language compatibility is guaranteed for ASCII 32-126. A message with
 * non-ASCII characters can decompress wrongly, because back-reference distances
 * are measured over different units in JS strings than in byte arrays — so the
 * length is checked rather than returning a plausible-looking wrong string.
 */
function decodeMessage(text, key) {
  const { originalSize, compressed } = decodeFrame(text, key);
  if (typeof unishox2_decompress_simple !== "function") {
    throw new Error("unishox2.js did not load, so the message cannot be decompressed");
  }
  const out = unishox2_decompress_simple(compressed, compressed.length);
  if (out.length !== originalSize) {
    throw new Error(
      `unishox2 length mismatch: frame declares ${originalSize} chars, ` +
      `decompressed ${out.length}`
    );
  }
  return out;
}

/* ==================================================================== *
 * Page wiring
 * ==================================================================== */

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const output = document.getElementById('output');
const statusText = document.getElementById('statusText');
const decodeBtn = document.getElementById('decodeBtn');
const decodeOutput = document.getElementById('decodeOutput');

let eventSource = null;

let first_line = true;

function appendLine(line) {
  if (first_line) {
    first_line = false;
  }
  else {
    output.textContent += line;
  }
  output.scrollTop = output.scrollHeight; // auto-scroll to bottom
}

function stopStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  first_line = true;
  statusText.textContent = 'Stopped.';
}

// The local-encode block is hidden unless server.js answers, so the deployed
// static site never shows it. Probing costs one request that is allowed to fail:
// any response at all, 404 included, means the helper is there.
const localEncode = document.getElementById('localEncode');
if (localEncode) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 1500);
  fetch('http://localhost:3000/', { signal: ac.signal })
    .then(() => { localEncode.hidden = false; })
    .catch(() => { /* no helper: the hosted encoder above is the only path */ })
    .finally(() => clearTimeout(timer));
}

// Guarded because the controls live in that optional block: without this the
// missing startBtn throws here and the decode handler below is never registered.
if (startBtn && stopBtn) {
  startBtn.addEventListener('click', () => {
    output.textContent = '';
    statusText.textContent = 'Connecting…';
    startBtn.disabled = true;
    stopBtn.disabled = false;

    // Read the two input fields.
    const topic = document.getElementById('ftopic').value;
    const bitstream = document.getElementById('fbitstream').value;
    const key = document.getElementById('fencodekey').value;

    // Send them as query parameters. encodeURIComponent keeps special
    // characters (spaces, &, etc.) from breaking the URL. This is safe:
    // the server treats them as plain string data, never as shell
    // commands, so there's nothing to inject here.
    const params = new URLSearchParams({ topic, bitstream, key });

    // EventSource opens a persistent connection and fires onmessage
    // every time the server sends a new "data: ..." event.
    eventSource = new EventSource(
      `http://localhost:3000/stream-command?${params.toString()}`
    );

    eventSource.onopen = () => {
      statusText.textContent = 'Streaming…';
    };

    eventSource.onmessage = (event) => {
      appendLine(event.data);
    };

    eventSource.onerror = () => {
      statusText.textContent =
        'Connection closed or lost (is server.js running?).';
      stopStream();
    };
  });

  stopBtn.addEventListener('click', async () => {
    // Tell the server to kill the running process, then close our connection.
    try {
      await fetch('http://localhost:3000/stop-command');
    } catch (err) {
      // Server may already be down; ignore.
    }
    stopStream();
  });
}

decodeBtn.addEventListener('click', () => {
  // Decoding runs entirely in the browser — no server needed.
  const encoded = document.getElementById('fencoded').value;
  const decode_key = document.getElementById('fdecodekey').value;

  try {
    decodeOutput.textContent = decodeMessage(encoded, decode_key);
  } catch (err) {
    // A wrong key produces arbitrary bits, so a failed decode is expected here
    // rather than exceptional; show why instead of failing silently.
    decodeOutput.textContent = `Could not decode: ${err.message}`;
  }
});
