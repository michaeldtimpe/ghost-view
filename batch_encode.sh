#!/bin/bash
# Copy each film from the NAS over ssh and encode with the enhanced crv pipeline
# (deduped frames + sync index + mechanical audio layer). Double-buffered:
# the NEXT film (that isn't already done) downloads while the CURRENT film
# encodes, so the network fills the gap the CPU leaves. Enhanced crv works
# in place (no source.mp4 copy), so cleanup is just removing the downloaded
# original once its encode succeeds.
#
# Usage: ./batch_encode.sh films.tsv
#   films.tsv: name<TAB>remote_path<TAB>slug, one film per line.
#   '#' comments and blank lines are ignored. The list is required; the old
#   built-in list was a July 2026 session's and only ever re-ran it by accident.
#
# Env (or site.env): CRV (engine command), NAS (ssh host, default `nas`),
#   NAS_LAN (preferred alias for the same box when it answers), MIN_FREE_GB,
#   PUBLISH_DIR -- when set, each finished folder is moved to
#   $PUBLISH_DIR/<slug>-crv (the shared-media/phase-N convention) and a film
#   already published there counts as complete. Unset: output stays in
#   ./crv-<slug>.
set -u
cd "$(dirname "$0")" || exit 1
# Engine: the repo's enhance_process.py through its venv. CRV overrides with a
# command string (word-split), e.g. a stub in tests. It used to default to
# ./.venv/bin/crv, which on a fresh venv is stock claude-real-video 0.2 --
# 150 bare frames, no INDEX.tsv -- so every film came back ENCODE_FAIL.
if [ -n "${CRV:-}" ]; then read -r -a ENGINE <<< "$CRV"; else ENGINE=(./.venv/bin/python ./enhance_process.py); fi
# If the NAS's usual alias goes through a VPN, a bulk copy on the home LAN
# pays for WireGuard + utun on top of ssh. Prefer NAS_LAN when it answers.
_nas_env=${NAS:-}
[ -f ./site.env ] && . ./site.env   # per-site hosts/paths, see site.example.env
if [ -n "$_nas_env" ]; then NAS=$_nas_env
elif [ -n "${NAS_LAN:-}" ] && nc -z -w 2 "$NAS_LAN" 22 >/dev/null 2>&1; then NAS=$NAS_LAN
else NAS=${NAS:-nas}; fi
ORIG=originals
mkdir -p "$ORIG"
STATUS=batch_status.tsv
: > "$STATUS"

FILMS_TSV="${1:-}"
PUBLISH_DIR="${PUBLISH_DIR:-}"

if [ -z "$FILMS_TSV" ]; then
  echo "usage: $0 films.tsv   (name<TAB>remote_path<TAB>slug per line; see films.example.tsv)" >&2
  exit 2
fi
if [ ! -f "$FILMS_TSV" ]; then
  echo "film list not found: $FILMS_TSV" >&2
  exit 1
fi
# name|remote_path|slug
FILMS=()
{
  bad=0
  while IFS=$'\t' read -r fname fremote fslug extra; do
    # a short row would leave the slug empty (rm -rf "crv-"; originals/.mkv), and
    # '|' would shear the name|remote|slug packing below. Refuse the whole list
    # before touching anything rather than half-run it.
    if [ -z "$fname" ] || [ -z "$fremote" ] || [ -z "$fslug" ] || [ -n "$extra" ] \
       || [[ ! "$fslug" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$fname$fremote" == *"|"* ]]; then
      echo "bad row in $FILMS_TSV (want name<TAB>remote_path<TAB>slug): $fname	$fremote	$fslug" >&2
      bad=1
      continue
    fi
    FILMS+=("$fname|$fremote|$fslug")
  done < <(grep -vE '^[[:space:]]*(#|$)' "$FILMS_TSV" | tr -d '\r')   # CRLF-safe
  [ "$bad" -eq 0 ] || exit 1
}

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# headroom to leave on the originals volume after a download, in GB
MIN_FREE_GB=${MIN_FREE_GB:-5}

# bytes free on the filesystem holding $1
free_bytes(){
  df -Pk "$1" 2>/dev/null | awk 'NR==2{printf "%.0f", $4*1024}'
}

# does $dest's container look complete (not just correctly sized)?
verify_container(){
  local f="$1" dur
  dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null)
  case "$dur" in
    ''|*[!0-9.]*) return 1 ;;
  esac
  awk -v d="$dur" 'BEGIN{exit !(d>0)}'
}

# download REMOTE -> DEST, verifying byte size against the NAS and, once sizes
# match, confirming the container is actually playable (not a truncated/corrupt
# stream). Resumes a smaller partial file with `tail`; restarts if the local
# file is somehow larger than the remote. Up to 3 attempts.
download(){
  local remote="$1" dest="$2"
  local rs; rs=$(ssh "$NAS" "stat -c%s \"$remote\"" 2>/dev/null)
  if [ -z "$rs" ]; then
    log "  cannot stat $(basename "$dest") on NAS (ssh failed?)"
    return 1
  fi
  local ls; ls=$(stat -f%z "$dest" 2>/dev/null)
  # Two films can sit on disk at once (the one encoding + the prefetch), and a
  # UHD remux is 60 GB; running the volume out mid-copy fails the download and
  # can take the encode's frame writes down with it.
  local need=$(( rs - ${ls:-0} )) free; free=$(free_bytes "$(dirname "$dest")")
  if [ -n "$free" ] && [ "$need" -gt 0 ] \
     && [ "$free" -lt $(( need + MIN_FREE_GB * 1000000000 )) ]; then
    log "  not enough disk for $(basename "$dest"): need $((need/1000000)) MB + ${MIN_FREE_GB} GB headroom, have $((free/1000000)) MB"
    return 1
  fi
  if [ -n "$ls" ] && [ "$rs" = "$ls" ]; then
    if verify_container "$dest"; then
      log "  $(basename "$dest") already present ($rs bytes)"
      return 0
    fi
    log "  $(basename "$dest") right size but container check failed; re-downloading"
    rm -f "$dest"
  fi
  local a
  for a in 1 2 3; do
    ls=$(stat -f%z "$dest" 2>/dev/null)
    if [ -n "$ls" ] && [ "$ls" -gt 0 ] && [ "$ls" -lt "$rs" ]; then
      log "  resuming $(basename "$dest") from byte $ls (remote=$rs)"
      ssh "$NAS" "tail -c +$((ls+1)) \"$remote\"" >> "$dest" 2>/dev/null
    else
      if [ -n "$ls" ] && [ "$ls" -gt "$rs" ]; then
        log "  local $(basename "$dest") larger than remote; restarting"
      fi
      ssh "$NAS" "cat \"$remote\"" > "$dest" 2>/dev/null
    fi
    ls=$(stat -f%z "$dest" 2>/dev/null)
    if [ "$rs" = "$ls" ]; then
      if verify_container "$dest"; then return 0; fi
      log "  retry $(basename "$dest") (container check failed after size match)"
      rm -f "$dest"
      continue
    fi
    log "  retry $(basename "$dest") (remote=$rs local=$ls)"
  done
  return 1
}

n=${#FILMS[@]}

# where a finished film lives: the publish dir when set, else in place
published_dir(){
  local slug="$1"
  if [ -n "$PUBLISH_DIR" ]; then echo "$PUBLISH_DIR/$slug-crv"; else echo "crv-$slug"; fi
}

# is this film already fully processed (index present here or where it was
# published, original cleaned up)?
is_complete(){
  local slug="$1"
  { [ -f "crv-$slug/INDEX.tsv" ] || [ -f "$(published_dir "$slug")/INDEX.tsv" ]; } \
    && [ ! -f "$ORIG/$slug.mkv" ]
}

# first index >= $1 whose film is not already complete, or -1 if none.
find_next_incomplete(){
  local start="$1" j jslug
  for ((j=start; j<n; j++)); do
    IFS='|' read -r _ _ jslug <<< "${FILMS[$j]}"
    if ! is_complete "$jslug"; then
      echo "$j"; return 0
    fi
  done
  echo -1
  return 1
}

pf_pid=""
pf_idx=-1

if [ "$n" -eq 0 ]; then
  log "no films to process."
  exit 0
fi

# Ctrl-C / kill: stop the background prefetch too (and the ssh it is running),
# or it keeps streaming a 60 GB file into originals/ after we are gone.
cleanup(){
  if [ -n "$pf_pid" ]; then
    pkill -P "$pf_pid" 2>/dev/null
    kill "$pf_pid" 2>/dev/null
  fi
}
trap 'cleanup; exit 130' INT TERM

# prime: download the first not-already-complete film synchronously
prime_idx=$(find_next_incomplete 0)
prime_rc=0
if [ "$prime_idx" -ge 0 ]; then
  IFS='|' read -r NAME0 REMOTE0 SLUG0 <<< "${FILMS[$prime_idx]}"
  log "downloading ${NAME0}..."
  download "$REMOTE0" "$ORIG/$SLUG0.mkv"; prime_rc=$?
fi

for ((i=0; i<n; i++)); do
  IFS='|' read -r NAME REMOTE SLUG <<< "${FILMS[$i]}"
  MKV="$ORIG/$SLUG.mkv"
  OUT="crv-$SLUG"
  log "===== $NAME ====="

  # already fully done? decide BEFORE touching the network for this film.
  if is_complete "$SLUG"; then
    log "$NAME already complete, skipping."
    done_dir=$(published_dir "$SLUG"); [ -d "$done_dir" ] || done_dir="$OUT"
    echo -e "$NAME\tDONE(skip)\t$(ls "$done_dir/frames" 2>/dev/null | wc -l | tr -d ' ')\t0" >> "$STATUS"
    continue
  fi

  # figure out this film's download result: prime, a pending prefetch, or
  # (if neither applies, e.g. after a previous film's failure) synchronous.
  if [ "$i" -eq "$prime_idx" ]; then
    dl_rc=$prime_rc
  elif [ "$i" -eq "$pf_idx" ]; then
    wait "$pf_pid"; dl_rc=$?
    pf_pid=""; pf_idx=-1
  else
    log "downloading ${NAME}..."
    download "$REMOTE" "$MKV"; dl_rc=$?
  fi

  if [ "$dl_rc" -ne 0 ]; then
    log "$NAME DOWNLOAD FAILED"; echo -e "$NAME\tDOWNLOAD_FAIL\t0\t0" >> "$STATUS"; rm -f "$MKV"
    continue
  fi

  # kick off prefetch of the next film that still needs downloading, so it
  # downloads while THIS one encodes
  nxt=$(find_next_incomplete $((i+1)))
  if [ "$nxt" -ge 0 ]; then
    IFS='|' read -r NNAME NREMOTE NSLUG <<< "${FILMS[$nxt]}"
    log "prefetching next: $NNAME (in background)"
    ( download "$NREMOTE" "$ORIG/$NSLUG.mkv" ) &
    pf_pid=$!
    pf_idx=$nxt
  fi

  # encode (enhanced: frames + sync index + audio layer), fresh output dir
  rm -rf "$OUT"
  log "$NAME encoding (enhanced: sync index + audio layer)..."
  t0=$(date +%s)
  # CRV_ORIGIN: the NAS path goes into MANIFEST (origin:), so which cut was
  # encoded is checkable later -- the local file is only a slug
  CRV_ORIGIN="$REMOTE" "${ENGINE[@]}" "$MKV" -o "$OUT" > "$OUT.run.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  FRAMES=$(ls "$OUT/frames" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$rc" -ne 0 ] || [ ! -f "$OUT/INDEX.tsv" ] || [ "$FRAMES" -eq 0 ]; then
    log "$NAME ENCODE FAILED (rc=$rc, ${secs}s). See $OUT.run.log"
    echo -e "$NAME\tENCODE_FAIL\t$FRAMES\t$secs" >> "$STATUS"
    continue
  fi
  if [ -n "$PUBLISH_DIR" ]; then
    # move the whole folder together (frames + index + manifest), never half
    DEST=$(published_dir "$SLUG")
    mkdir -p "$PUBLISH_DIR"
    if [ -e "$DEST" ]; then
      log "$NAME publish target exists, leaving output in $OUT: $DEST"
      echo -e "$NAME\tPUBLISH_FAIL\t$FRAMES\t$secs" >> "$STATUS"
      continue
    fi
    if ! mv "$OUT" "$DEST" || [ ! -f "$DEST/INDEX.tsv" ]; then
      log "$NAME PUBLISH FAILED: could not move $OUT to $DEST (original kept)"
      echo -e "$NAME\tPUBLISH_FAIL\t$FRAMES\t$secs" >> "$STATUS"
      continue
    fi
    mv -f "$OUT.run.log" "$DEST/run.log" 2>/dev/null
    log "$NAME published -> $DEST"
  fi
  rm -f "$MKV"   # remove the downloaded original; nothing else to clean
  SND=$(($(wc -l < "$(published_dir "$SLUG")/sounds.tsv" 2>/dev/null || echo 1)-1)); [ "$SND" -lt 0 ] && SND=0
  log "$NAME done: $FRAMES frames, sync index built, $SND distinct sounds, ${secs}s. Original removed."
  echo -e "$NAME\tDONE\t$FRAMES\t$secs" >> "$STATUS"
done

# make sure any trailing prefetch finished
[ -n "$pf_pid" ] && wait "$pf_pid" 2>/dev/null
log "ALL FILMS PROCESSED."
