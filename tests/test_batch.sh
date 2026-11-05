#!/bin/bash
# Test harness for batch_encode.sh. Runs each scenario in its own throwaway
# temp dir (under the scratchpad, not /tmp) with stub ssh/ffprobe/crv on
# PATH, then asserts batch_status.tsv contents and filesystem state.
#
# Usage: bash tests/test_batch.sh
set -u

# Throwaway work dirs live under one mktemp root (was a hardcoded path from an
# old session, which broke on any other machine or CI).
SCRATCH_BASE=$(mktemp -d "${TMPDIR:-/tmp}/crv-batch-test.XXXXXX")
trap 'rm -rf "$SCRATCH_BASE"' EXIT
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_SRC="$REPO_DIR/batch_encode.sh"

mkdir -p "$SCRATCH_BASE"

overall_fail=0
pass_count=0
fail_count=0
scenario_failed=0

# ---------- stub builders ----------

make_stubs(){
  local bindir="$1"
  mkdir -p "$bindir"

  cat > "$bindir/ssh" <<'EOF'
#!/bin/bash
# fake ssh: [-o opt]... host command. The host goes to SSH_HOSTS_LOG; `true`
# (the NAS_LAN probe) succeeds only for the host named in FAKE_LAN_UP.
while [ "${1:-}" = "-o" ]; do shift 2; done
host="$1"; shift
cmd="$*"
printf '%s\n' "$host" >> "${SSH_HOSTS_LOG:-/dev/null}"
if [ "$cmd" = "true" ]; then [ "$host" = "${FAKE_LAN_UP:-}" ]; exit $?; fi
printf '%s\n' "$cmd" >> "$SSH_LOG"

stat_re='^stat -c%s "(.*)"$'
cat_re='^cat "(.*)"$'
tail_re='^tail -c \+([0-9]+) "(.*)"$'

if [[ "$cmd" =~ $stat_re ]]; then
  f="$FAKE_NAS${BASH_REMATCH[1]}"
  [ -f "$f" ] || exit 1
  stat -f%z "$f"
  exit 0
elif [[ "$cmd" =~ $cat_re ]]; then
  f="$FAKE_NAS${BASH_REMATCH[1]}"
  [ -f "$f" ] || exit 1
  cat "$f"
  exit 0
elif [[ "$cmd" =~ $tail_re ]]; then
  f="$FAKE_NAS${BASH_REMATCH[2]}"
  [ -f "$f" ] || exit 1
  tail -c +"${BASH_REMATCH[1]}" "$f"
  exit 0
else
  echo "fake-ssh: unrecognized command: $cmd" >&2
  exit 1
fi
EOF
  chmod +x "$bindir/ssh"

  cat > "$bindir/ffprobe" <<'EOF'
#!/bin/bash
# fake ffprobe: prints a fixed duration, unless the target path matches a
# marker (used to simulate a corrupt/truncated container).
for last; do :; done
if [ -n "${FAKE_FFPROBE_FAIL_MARKER:-}" ] && [[ "$last" == *"$FAKE_FFPROBE_FAIL_MARKER"* ]]; then
  exit 1
fi
echo "10.0"
EOF
  chmod +x "$bindir/ffprobe"

  cat > "$bindir/df" <<'EOF'
#!/bin/bash
# fake df -Pk: FAKE_DF_FREE_KB available (default: plenty)
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "/dev/fake 999999999 1 ${FAKE_DF_FREE_KB:-999999999} 1% /"
EOF
  chmod +x "$bindir/df"

  cat > "$bindir/crv" <<'EOF'
#!/bin/bash
# fake crv: writes the expected output layout, or fails for a slug listed
# in FAKE_CRV_FAIL_SLUGS (space-separated).
shift  # drop the mkv path arg
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
slug="${out#crv-}"
for bad in ${FAKE_CRV_FAIL_SLUGS:-}; do
  if [ "$bad" = "$slug" ]; then
    echo "fake-crv: forced failure for $slug" >&2
    exit 1
  fi
done
mkdir -p "$out/frames"
printf 'a\tb\n' > "$out/INDEX.tsv"
: > "$out/frames/frame_001.jpg"
{ printf 'sound\tstart\tend\n'; printf 's1\t0\t1\n'; } > "$out/sounds.tsv"
EOF
  chmod +x "$bindir/crv"
}

# ---------- assertion helpers ----------

fail(){
  echo "    FAIL: $*"
  scenario_failed=1
}

assert_status_line(){
  # assert_status_line STATUS_FILE NAME STATE [FRAMES]
  local f="$1" name="$2" state="$3" frames="${4:-}"
  local line
  line=$(awk -F'\t' -v n="$name" '$1==n{print}' "$f" 2>/dev/null)
  if [ -z "$line" ]; then
    fail "no status line for '$name' in $f"
    return
  fi
  local got_state got_frames
  got_state=$(printf '%s' "$line" | awk -F'\t' '{print $2}')
  got_frames=$(printf '%s' "$line" | awk -F'\t' '{print $3}')
  if [ "$got_state" != "$state" ]; then
    fail "$name: expected state '$state', got '$got_state'"
  fi
  if [ -n "$frames" ] && [ "$got_frames" != "$frames" ]; then
    fail "$name: expected frames '$frames', got '$got_frames'"
  fi
}

assert_exists(){ [ -e "$1" ] || fail "expected $1 to exist"; }
assert_absent(){ [ -e "$1" ] && fail "expected $1 to NOT exist"; return 0; }
assert_ssh_log_no_match(){ if grep -qF "$2" "$1" 2>/dev/null; then fail "ssh log unexpectedly contains: $2"; fi; }
assert_ssh_log_match(){ if ! grep -qF "$2" "$1" 2>/dev/null; then fail "ssh log missing expected call: $2"; fi; }

run_scenario(){
  local name="$1"; shift
  scenario_failed=0
  echo "--- scenario: $name ---"
  "$@"
  if [ "$scenario_failed" -eq 0 ]; then
    echo "PASS: $name"
    pass_count=$((pass_count+1))
  else
    echo "FAIL: $name"
    fail_count=$((fail_count+1))
    overall_fail=1
  fi
}

# ---------- common scenario setup ----------

setup_common(){
  # sets: WORK BIN FAKE_NAS_DIR SSH_LOG TSV
  WORK=$(mktemp -d "$SCRATCH_BASE/batch-test.XXXXXX")
  cp "$SCRIPT_SRC" "$WORK/batch_encode.sh"
  chmod +x "$WORK/batch_encode.sh"
  BIN="$WORK/bin"
  make_stubs "$BIN"
  FAKE_NAS_DIR="$WORK/fake-nas"
  mkdir -p "$FAKE_NAS_DIR"
  SSH_LOG="$WORK/ssh.log"
  : > "$SSH_LOG"

  yes "abcdefghij0123456789" | head -c 30000 > "$FAKE_NAS_DIR/film1.mkv"
  yes "the-second-filmZZZZZ" | head -c 25000 > "$FAKE_NAS_DIR/film2.mkv"
  yes "third-film-contentQQ" | head -c 22000 > "$FAKE_NAS_DIR/film3.mkv"

  TSV="$WORK/films.tsv"
  cat > "$TSV" <<'EOF'
# test film list
FilmOne	/film1.mkv	film1
FilmTwo	/film2.mkv	film2
FilmThree	/film3.mkv	film3
EOF
}

run_batch(){
  # run_batch [EXTRA_ENV=val ...] -- runs batch_encode.sh films.tsv in $WORK
  ( cd "$WORK" && env "$@" \
      CRV="$BIN/crv" \
      NAS="testnas" \
      FAKE_NAS="$FAKE_NAS_DIR" \
      SSH_LOG="$SSH_LOG" \
      PATH="$BIN:$PATH" \
      bash ./batch_encode.sh films.tsv > "$WORK/run.log" 2>&1 )
}

# ---------- scenarios ----------

scenario_happy_path(){
  setup_common
  run_batch
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_status_line "$WORK/batch_status.tsv" FilmTwo DONE 1
  assert_status_line "$WORK/batch_status.tsv" FilmThree DONE 1
  assert_absent "$WORK/originals/film1.mkv"
  assert_absent "$WORK/originals/film2.mkv"
  assert_absent "$WORK/originals/film3.mkv"
  assert_exists "$WORK/crv-film1/INDEX.tsv"
  assert_exists "$WORK/crv-film2/INDEX.tsv"
  assert_exists "$WORK/crv-film3/INDEX.tsv"
}

scenario_download_fail_cascade(){
  setup_common
  rm -f "$FAKE_NAS_DIR/film2.mkv"   # film2 missing on the NAS
  run_batch
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_status_line "$WORK/batch_status.tsv" FilmTwo DOWNLOAD_FAIL 0
  assert_status_line "$WORK/batch_status.tsv" FilmThree DONE 1
  assert_absent "$WORK/originals/film2.mkv"
}

scenario_skip_already_complete(){
  setup_common
  mkdir -p "$WORK/crv-film1/frames"
  : > "$WORK/crv-film1/INDEX.tsv"
  : > "$WORK/crv-film1/frames/frame_001.jpg"
  run_batch
  assert_status_line "$WORK/batch_status.tsv" FilmOne "DONE(skip)" 1
  assert_status_line "$WORK/batch_status.tsv" FilmTwo DONE 1
  assert_status_line "$WORK/batch_status.tsv" FilmThree DONE 1
  assert_ssh_log_no_match "$SSH_LOG" '"/film1.mkv"'
  assert_absent "$WORK/originals/film1.mkv"
}

scenario_resume_partial(){
  setup_common
  mkdir -p "$WORK/originals"
  head -c 12000 "$FAKE_NAS_DIR/film1.mkv" > "$WORK/originals/film1.mkv"
  # force film1's encode to fail so the (now-complete) original is kept on
  # disk for a byte-for-byte comparison against the NAS source.
  run_batch FAKE_CRV_FAIL_SLUGS=film1
  assert_status_line "$WORK/batch_status.tsv" FilmOne ENCODE_FAIL 0
  assert_status_line "$WORK/batch_status.tsv" FilmTwo DONE 1
  assert_status_line "$WORK/batch_status.tsv" FilmThree DONE 1
  assert_exists "$WORK/originals/film1.mkv"
  if [ -e "$WORK/originals/film1.mkv" ]; then
    cmp -s "$WORK/originals/film1.mkv" "$FAKE_NAS_DIR/film1.mkv" || fail "resumed film1.mkv does not match source bytes"
  fi
  assert_ssh_log_match "$SSH_LOG" 'tail -c +12001 "/film1.mkv"'
}

scenario_encode_fail(){
  setup_common
  run_batch FAKE_CRV_FAIL_SLUGS=film2
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_status_line "$WORK/batch_status.tsv" FilmTwo ENCODE_FAIL 0
  assert_status_line "$WORK/batch_status.tsv" FilmThree DONE 1
  assert_exists "$WORK/originals/film2.mkv"
}

scenario_empty_list(){
  setup_common
  printf '# nothing here\n\n' > "$TSV"
  run_batch
  local rc=$?
  [ "$rc" -eq 0 ] || fail "empty list: expected exit 0, got $rc"
  grep -q "no films to process" "$WORK/run.log" || fail "empty list: no 'no films' log line"
  grep -q "unbound variable" "$WORK/run.log" && fail "empty list: unbound variable crash"
  [ -s "$SSH_LOG" ] && fail "empty list: ssh was called"
  return 0
}

scenario_bad_row_refuses_list(){
  setup_common
  printf 'FilmOne\t/film1.mkv\tfilm1\nNoSlug\t/film2.mkv\n' > "$TSV"
  run_batch
  local rc=$?
  [ "$rc" -ne 0 ] || fail "bad row: expected non-zero exit"
  grep -q "bad row" "$WORK/run.log" || fail "bad row: not reported"
  [ -s "$SSH_LOG" ] && fail "bad row: ssh was called before validation"
  assert_absent "$WORK/crv-"
  return 0
}

scenario_crlf_list(){
  setup_common
  printf '# windows list\r\nFilmOne\t/film1.mkv\tfilm1\r\nFilmTwo\t/film2.mkv\tfilm2\r\n' > "$TSV"
  run_batch
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_status_line "$WORK/batch_status.tsv" FilmTwo DONE 1
  assert_exists "$WORK/crv-film1/INDEX.tsv"
  assert_absent "$WORK/crv-film1"$'\r'
}

scenario_low_disk(){
  setup_common
  run_batch FAKE_DF_FREE_KB=1000 MIN_FREE_GB=1
  assert_status_line "$WORK/batch_status.tsv" FilmOne DOWNLOAD_FAIL 0
  assert_status_line "$WORK/batch_status.tsv" FilmTwo DOWNLOAD_FAIL 0
  grep -q "not enough disk" "$WORK/run.log" || fail "low disk: not reported"
  assert_ssh_log_no_match "$SSH_LOG" 'cat "/film1.mkv"'
}

scenario_publish_dir(){
  setup_common
  run_batch PUBLISH_DIR="$WORK/pub/phase-9"
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_exists "$WORK/pub/phase-9/film1-crv/INDEX.tsv"
  assert_exists "$WORK/pub/phase-9/film1-crv/run.log"
  assert_absent "$WORK/crv-film1"
  assert_absent "$WORK/originals/film1.mkv"
  # a second run sees the published copy as complete and never touches the NAS
  : > "$SSH_LOG"
  run_batch PUBLISH_DIR="$WORK/pub/phase-9"
  assert_status_line "$WORK/batch_status.tsv" FilmOne "DONE(skip)" 1
  assert_ssh_log_no_match "$SSH_LOG" '"/film1.mkv"'
  # an occupied target is refused, output and original are kept
  rm -rf "$WORK/originals" "$WORK/crv-film2"
  mkdir -p "$WORK/pub/phase-9/film2-crv"; rm -f "$WORK/pub/phase-9/film2-crv/INDEX.tsv"
  run_batch PUBLISH_DIR="$WORK/pub/phase-9"
  assert_status_line "$WORK/batch_status.tsv" FilmTwo PUBLISH_FAIL 1
  assert_exists "$WORK/crv-film2/INDEX.tsv"
  assert_exists "$WORK/originals/film2.mkv"
}

scenario_no_list_arg(){
  setup_common
  ( cd "$WORK" && CRV="$BIN/crv" PATH="$BIN:$PATH" bash ./batch_encode.sh > "$WORK/run.log" 2>&1 )
  local rc=$?
  [ "$rc" -eq 2 ] || fail "no list: expected exit 2, got $rc"
  grep -q "usage:" "$WORK/run.log" || fail "no list: no usage line"
  [ -s "$SSH_LOG" ] && fail "no list: ssh was called"
  return 0
}

scenario_status_has_seconds(){
  setup_common
  run_batch
  awk -F'\t' '$2=="DONE" && $4 !~ /^[0-9]+$/ {bad=1} END{exit bad}' "$WORK/batch_status.tsv" \
    || fail "status: DONE rows lack an integer seconds column"
}

run_batch_nas(){
  # run_batch_nas [ENV=val ...] -- like run_batch but NAS is not forced, so the
  # script picks it (site.env, NAS_LAN probe, default). Hosts go to hosts.log.
  : > "$WORK/hosts.log"
  ( cd "$WORK" && env -u NAS -u NAS_LAN "$@" \
      CRV="$BIN/crv" \
      FAKE_NAS="$FAKE_NAS_DIR" \
      SSH_LOG="$SSH_LOG" \
      SSH_HOSTS_LOG="$WORK/hosts.log" \
      PATH="$BIN:$PATH" \
      bash ./batch_encode.sh films.tsv > "$WORK/run.log" 2>&1 )
}

assert_transfers_via(){
  # every non-probe ssh call went to host $1
  local other
  other=$(grep -vxF "$1" "$WORK/hosts.log" | grep -vxF "${2:-$1}" | head -1)
  grep -qxF "$1" "$WORK/hosts.log" || fail "no ssh call went to $1 (hosts: $(sort -u "$WORK/hosts.log" | tr '\n' ' '))"
  [ -z "$other" ] || fail "unexpected ssh host $other"
}

scenario_nas_lan_alias_up(){
  # an ssh-config alias that DNS can't resolve must still be picked when it answers
  setup_common
  run_batch_nas NAS_LAN=lan-alias-only FAKE_LAN_UP=lan-alias-only
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_transfers_via lan-alias-only
}

scenario_nas_lan_down_site_env(){
  # LAN alias does not answer: fall back to NAS from site.env
  setup_common
  printf 'NAS=${NAS:-wan-host}\nNAS_LAN=${NAS_LAN:-lan-alias-only}\n' > "$WORK/site.env"
  run_batch_nas FAKE_LAN_UP=nobody
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_transfers_via wan-host lan-alias-only
  grep -qxF lan-alias-only "$WORK/hosts.log" || fail "NAS_LAN from site.env was never probed"
}

scenario_nas_env_wins(){
  # NAS in the environment beats site.env and skips the probe
  setup_common
  printf 'NAS=${NAS:-wan-host}\nNAS_LAN=${NAS_LAN:-lan-alias-only}\n' > "$WORK/site.env"
  run_batch_nas NAS=explicit-host FAKE_LAN_UP=lan-alias-only
  assert_status_line "$WORK/batch_status.tsv" FilmOne DONE 1
  assert_transfers_via explicit-host
}

# ---------- syntax checks ----------

echo "=== bash -n ==="
if bash -n "$SCRIPT_SRC"; then echo "OK: batch_encode.sh syntax"; else echo "FAIL: batch_encode.sh syntax"; overall_fail=1; fi
if bash -n "${BASH_SOURCE[0]}"; then echo "OK: test_batch.sh syntax"; else echo "FAIL: test_batch.sh syntax"; overall_fail=1; fi
if command -v shellcheck >/dev/null 2>&1; then
  echo "=== shellcheck ==="
  shellcheck "$SCRIPT_SRC" || echo "NOTE: shellcheck reported issues above (non-fatal)"
else
  echo "NOTE: shellcheck not installed, skipping"
fi

# ---------- run scenarios ----------

run_scenario "1 happy path (3 films)" scenario_happy_path
run_scenario "2 download fail cascade" scenario_download_fail_cascade
run_scenario "3 skip already complete" scenario_skip_already_complete
run_scenario "4 resume partial download" scenario_resume_partial
run_scenario "5 encode fail mid-batch" scenario_encode_fail
run_scenario "6 empty film list" scenario_empty_list
run_scenario "7 malformed row refuses the list" scenario_bad_row_refuses_list
run_scenario "8 CRLF film list" scenario_crlf_list
run_scenario "9 not enough disk" scenario_low_disk
run_scenario "10 publish dir" scenario_publish_dir
run_scenario "11 no list argument" scenario_no_list_arg
run_scenario "12 status has seconds" scenario_status_has_seconds
run_scenario "13 NAS_LAN ssh alias answers" scenario_nas_lan_alias_up
run_scenario "14 NAS_LAN down, site.env NAS" scenario_nas_lan_down_site_env
run_scenario "15 NAS env beats site.env" scenario_nas_env_wins

echo "=== SUMMARY: $pass_count passed, $fail_count failed ==="
exit $overall_fail
