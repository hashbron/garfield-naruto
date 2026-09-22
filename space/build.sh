#!/usr/bin/env bash
# Assemble a deployable Space folder. The encoder itself is NOT duplicated in
# git: it is copied in at build time, so the Space and the CLI can never drift.
#
#   ./space/build.sh                 -> builds space/_build
#   ./space/build.sh ../my-space     -> builds straight into a Space checkout
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="${1:-$here/_build}"
mkdir -p "$out"
cp "$here/app.py" "$here/gpu_encode.py" "$here/requirements.txt" "$here/README.md" "$out/"
cp "$root/sentence_encode.py" "$root/payload_codec.py" "$out/"
echo "built $out"
ls -1 "$out"
echo
echo "To publish:"
echo "  cd $out && git init && git remote add origin https://huggingface.co/spaces/<user>/<name>"
echo "  git add -A && git commit -m 'stego encoder' && git push -u origin main"
echo "Then set the Space hardware to ZeroGPU in its settings."
