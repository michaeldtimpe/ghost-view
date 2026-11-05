#!/bin/bash
# The second-review repairs on one finished package, in place:
#   dark_cuts   cuts the engine missed inside long dark shots (from the frames)
#   vision_pass regroups shots from the new cut column, captions the new shots
#   sdh_sounds  SDH cue descriptions replace CLAP ambience (SDH sources only)
#   voice_gate  voice placement cleared where no cue is live
#   ocr_pass    Apple Vision OCR on every kept frame -> text.tsv / text column
#   origin      MANIFEST origin: + container: from origins.tsv (slug bytes secs path)
#   cite_map    old package -> this package frame map, when the old one exists
# and sync.md carries captions + OCR whichever pass ran last.
#
#   repair_pass.sh <pkg_dir> [--origins origins.tsv] [--old <old_pkg_dir>]
# Env: CRV_VLM_BASE / CRV_VLM_MODEL for vision_pass (a server that takes
# images); OMLX_API_KEY from the environment or $CRV_ENV_FILE. Idempotent: every pass skips
# or repeats harmlessly; `repaired:` in MANIFEST marks a finished pass.
set -u
export PATH=/opt/homebrew/bin:$PATH
cd "$(dirname "$0")" || exit 1
[ -f ./site.env ] && . ./site.env   # per-site hosts/paths, see site.example.env
[ -n "${OMLX_API_KEY:-}" ] || [ -z "${CRV_ENV_FILE:-}" ] || { set -a; . "$CRV_ENV_FILE"; set +a; }
PY=./.venv/bin/python
pkg="${1%/}"; shift
origins=""; old=""
while [ $# -gt 0 ]; do case "$1" in
  --origins) origins="$2"; shift 2;;
  --old) old="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;;
esac; done
[ -f "$pkg/INDEX.tsv" ] || { echo "✗ $pkg: no INDEX.tsv" >&2; exit 1; }
slug=$(basename "$pkg"); slug=${slug%-crv}
log() { echo "[$(date +%H:%M:%S)] $slug: $*"; }

if grep -q "^repaired: " "$pkg/MANIFEST.txt"; then log "already repaired, skipping"; exit 0; fi

log "dark cuts"
$PY dark_cuts.py "$pkg" || exit 1
log "vision (regroup + caption new shots)"
$PY vision_pass.py "$pkg" || exit 1
# vision_pass stops on the first server error and asks for a rerun; a package
# with uncaptioned shots must not be stamped repaired
if grep -q "stopping; rerun" "$pkg/run.log" 2>/dev/null; then :; fi
log "sdh sounds"
$PY sdh_sounds.py "$pkg" || exit 1
log "voice gate"
$PY voice_gate.py "$pkg" || exit 1
log "ocr"
$PY ocr_pass.py "$pkg" || exit 1
if [ -n "$origins" ]; then
  line=$(awk -F'\t' -v s="$slug" '$1==s' "$origins" | head -1)
  if [ -n "$line" ]; then
    bytes=$(printf '%s' "$line" | cut -f2); secs=$(printf '%s' "$line" | cut -f3)
    remote=$(printf '%s' "$line" | cut -f4)
    dur=$($PY -c "import sys; s=float(sys.argv[1]); print('%02d:%02d:%02d' % (s//3600, s%3600//60, s%60))" "$secs")
    log "origin"
    $PY origin_patch.py "$pkg" --origin "$remote" --duration "$dur" --bytes "$bytes" \
        --name "$(basename "$remote")" || exit 1
  else
    log "no origin row for $slug in $origins"
  fi
fi
if [ -n "$old" ] && [ -d "$old/frames" ]; then
  log "cite map from $(basename "$old")"
  $PY cite_map.py "$old" "$pkg" || exit 1
fi
$PY -c "import pkg, sys; pkg.rebuild_sync(sys.argv[1])" "$pkg" || exit 1
# all captions in? vision_pass writes N shots / M captioned; a gap means the
# server stopped it (no stamp, so the next run finishes the job)
read -r shots caps < <($PY - "$pkg" <<'EOF'
import sys, re
m = re.search(r"(?m)^vision: .*?(\d+) shots in shots.tsv, (\d+) frames captioned", open(sys.argv[1] + "/MANIFEST.txt").read())
print(m.group(1), m.group(2)) if m else print(0, 0)
EOF
)
# a row per shot in shots.tsv (a caption the model garbled still has a row)
ncap=$(awk -F'\t' 'NR>1 && $1!=""' "$pkg/shots.tsv" | wc -l | tr -d ' ')
if [ "$ncap" -lt "$shots" ]; then
  log "✗ $ncap of $shots shots captioned; not stamped (rerun once the vision server is up)"; exit 1
fi
$PY -c "import pkg, sys, datetime; pkg.manifest_set(sys.argv[1], 'repaired', datetime.date.today().isoformat() + ' second-review pass (dark cuts, SDH sounds, voice gate, OCR, origin, cite map)', after='engine')" "$pkg"
log "done"
