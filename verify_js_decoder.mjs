/*
 * verify_js_decoder.mjs — proves the browser decoder agrees with the Python encoder.
 *
 *   node verify_js_decoder.mjs                    # key schedule + bit extraction only
 *   node verify_js_decoder.mjs ./unishox2.js      # also the Unishox2 message layer
 *
 * Test vectors are produced by the Python side (unishox_vectors.json,
 * stego_vectors.json), so this checks against the real codec rather than against
 * another JavaScript implementation of the same idea.
 */
import fs from "node:fs";
import * as dec from "./stego_decode.js";

const arg = process.argv[2];
let fail = 0, checks = 0;
const ok = (cond, msg) => { checks++; if (!cond) { fail++; console.log("  FAIL " + msg); } };

/* ---- 1. key schedule + extract(), against Python-generated vectors ---- */
if (fs.existsSync("./stego_vectors.json")) {
  const V = JSON.parse(fs.readFileSync("./stego_vectors.json", "utf8"));
  for (const [key, table] of Object.entries(V.roles)) {
    for (let pos = 0; pos < table.length; pos++) {
      for (let i = 0; i < 26; i++) {
        ok(dec.charRole("abcdefghijklmnopqrstuvwxyz"[i], pos, key) === table[pos][i],
           `role key=${key} pos=${pos} letter=${i}`);
      }
    }
  }
  for (const [k, pybits] of Object.entries(V.bits)) {
    const [key, idx] = k.split("|");
    const js = dec.decodeBits(V.texts[+idx], key);
    ok(js.length === pybits.length && js.every((v, i) => v === pybits[i]),
       `extract(${k})`);
  }
  console.log(`key schedule + extract : ${checks} checks, ${fail} failures`);
} else {
  console.log("key schedule + extract : SKIPPED (stego_vectors.json missing)");
}

/* ---- 2. Unishox2 message layer ---- */
if (!arg) {
  console.log("unishox2 layer         : SKIPPED (pass a path to unishox2.js)");
} else {
  const usx = await import(arg.startsWith(".") || arg.startsWith("/") ? arg : "./" + arg);
  const V = JSON.parse(fs.readFileSync("./unishox_vectors.json", "utf8"));
  let ufail = 0, uchecks = 0, skipped = 0;
  for (const v of V) {
    const isAscii = [...v.message].every((c) => c.charCodeAt(0) >= 32 && c.charCodeAt(0) <= 126);
    if (!isAscii) { skipped++; continue; }   // string API is not byte-compatible for UTF-8
    uchecks++;
    const compressed = Uint8Array.from(Buffer.from(v.compressed_hex, "hex"));
    let got;
    try {
      got = usx.unishox2_decompress_simple(compressed, compressed.length);
    } catch (e) {
      ufail++; console.log(`  FAIL decompress threw on ${JSON.stringify(v.message.slice(0, 30))}: ${e.message}`);
      continue;
    }
    if (got !== v.message) {
      ufail++;
      console.log(`  FAIL message mismatch\n    want ${JSON.stringify(v.message)}\n    got  ${JSON.stringify(got)}`);
    }
  }
  console.log(`unishox2 layer         : ${uchecks} vectors, ${ufail} failures` +
              (skipped ? ` (${skipped} non-ASCII skipped)` : ""));

  // end-to-end: cover text -> bits -> frame -> message
  if (fs.existsSync("./stego_e2e.json")) {
    const E = JSON.parse(fs.readFileSync("./stego_e2e.json", "utf8"));
    let efail = 0;
    for (const c of E) {
      try {
        const got = dec.decodeMessage(c.text, c.key, usx);
        if (got !== c.message) { efail++; console.log(`  FAIL e2e: want ${JSON.stringify(c.message)} got ${JSON.stringify(got)}`); }
      } catch (e) { efail++; console.log(`  FAIL e2e threw: ${e.message}`); }
    }
    console.log(`end-to-end             : ${E.length} cover texts, ${efail} failures`);
    fail += efail;
  }
  fail += ufail;
}

console.log(fail === 0 ? "\nALL CHECKS PASSED" : `\n${fail} FAILURES`);
process.exit(fail === 0 ? 0 : 1);
