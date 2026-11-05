#!/bin/bash
# Engine + vision pre-pass + CLAP sound tags as one "encode" step for
# batch_encode.sh:   CRV=./full_pipeline.sh PUBLISH_DIR=... ./batch_encode.sh films.tsv
# Any stage failing fails the film, so the driver keeps the original and the
# status file shows ENCODE_FAIL. Needs oMLX on :8000 and OMLX_API_KEY, in the
# environment or in $CRV_ENV_FILE (vision_pass), and the uv script env (sound_tags).
set -u
export PATH=/opt/homebrew/bin:$PATH
_site="$(dirname "$0")/site.env"; [ -f "$_site" ] && . "$_site"   # per-site hosts/paths, see site.example.env
[ -n "${OMLX_API_KEY:-}" ] || [ -z "${CRV_ENV_FILE:-}" ] || { set -a; . "$CRV_ENV_FILE"; set +a; }
mkv="$1"; shift
out=""
while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; *) shift;; esac; done
cd "$(dirname "$0")" || exit 1
./.venv/bin/python enhance_process.py "$mkv" -o "$out" || exit 1
./.venv/bin/python vision_pass.py "$out" || exit 1
uv run --script sound_tags.py "$out" "$mkv" || exit 1
# OCR on every kept frame (vision_pass reads only a shot's first frame, and
# Alien's SPECIAL ORDER 937 screen sat inside a dark static shot)
./.venv/bin/python ocr_pass.py "$out" || exit 1
grep -q "^vision:" "$out/MANIFEST.txt" && grep -q "^sound tags:" "$out/MANIFEST.txt" \
  && grep -q "^ocr:" "$out/MANIFEST.txt"
