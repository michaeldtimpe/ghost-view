#!/bin/bash
# repair_pass.sh over every package under the shared-media phases, with the
# superseded package (crv-redo/old/phase-N/<slug>-crv[.pilot]) as the cite-map
# source and crv-redo/origins.tsv for the origin lines. Sequential; a package
# that fails is logged and the loop goes on. Re-runnable: stamped packages skip.
#
#   repair_all.sh [phase-dir ...]      (default: all phase-* dirs with crv packages)
# Env (or site.env): VLM_HOST (default localhost) -- the oMLX server that takes
# images; CRV_MEDIA / CRV_REDO -- package and redo roots. OMLX_API_KEY comes from
# the environment, else from $CRV_ENV_FILE on VLM_HOST (over ssh when remote),
# and is never printed.
set -u
export PATH=/opt/homebrew/bin:$PATH
cd "$(dirname "$0")" || exit 1
[ -f ./site.env ] && . ./site.env   # per-site hosts/paths, see site.example.env
MEDIA=${CRV_MEDIA:-$HOME/crv/shared-media}
REDO=${CRV_REDO:-$HOME/crv/crv-redo}
LOG=$REDO/repair.log
VLM_HOST=${VLM_HOST:-localhost}
export CRV_VLM_BASE="http://$VLM_HOST:8000/v1"
if [ -z "${OMLX_API_KEY:-}" ] && [ -n "${CRV_ENV_FILE:-}" ]; then
  # the key file path as seen on VLM_HOST; ~ is left for that host's shell
  _kf=${CRV_ENV_FILE/#$HOME/\~}
  if [ "$VLM_HOST" = localhost ]; then _rd=(sh -c); else _rd=(ssh "$VLM_HOST"); fi
  OMLX_API_KEY=$("${_rd[@]}" "grep '^OMLX_API_KEY=' $_kf | head -1 | cut -d= -f2-" | tr -d '"'"'"'')
fi
[ -n "$OMLX_API_KEY" ] || { echo "no OMLX_API_KEY on $VLM_HOST" >&2; exit 1; }
export OMLX_API_KEY
curl -s -m 10 -H "Authorization: Bearer $OMLX_API_KEY" "$CRV_VLM_BASE/models" | grep -q '"data"' \
  || { echo "vision server $CRV_VLM_BASE not answering" >&2; exit 1; }

if [ $# -gt 0 ]; then phases=("$@"); else
  phases=(); for d in "$MEDIA"/phase-*/; do /bin/ls -d "$d"*-crv >/dev/null 2>&1 && phases+=("${d%/}"); done
fi
ok=0; bad=0; skip=0
for ph in "${phases[@]}"; do
  pn=$(basename "$ph")
  for pkg in "$ph"/*-crv; do
    [ -f "$pkg/INDEX.tsv" ] || continue
    slug=$(basename "$pkg" -crv)
    if grep -q "^repaired: " "$pkg/MANIFEST.txt"; then skip=$((skip+1)); continue; fi
    old=""
    for cand in "$REDO/old/$pn/$slug-crv.pilot" "$REDO/old/$pn/$slug-crv"; do
      [ -d "$cand/frames" ] && { old="$cand"; break; }
    done
    args=(--origins "$REDO/origins.tsv"); [ -n "$old" ] && args+=(--old "$old")
    echo "[$(date +%H:%M)] $pn/$slug start (old: ${old:-none})" >> "$LOG"
    if ./repair_pass.sh "$pkg" "${args[@]}" >> "$LOG" 2>&1; then
      ok=$((ok+1)); echo "[$(date +%H:%M)] $pn/$slug OK" >> "$LOG"
    else
      bad=$((bad+1)); echo "[$(date +%H:%M)] $pn/$slug FAILED" >> "$LOG"
    fi
  done
done
echo "[$(date +%H:%M)] ALL DONE: $ok repaired, $bad failed, $skip already stamped" | tee -a "$LOG"
