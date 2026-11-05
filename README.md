# ghost-view — let an LLM actually *watch* a film

Started from [claude-real-video](https://github.com/HUANGCHIHHUNGLeo/claude-real-video)
by LeoAido (MIT; see `NOTICE`). Turns a movie into something an LLM
can read as a **cut**, not a shuffled flip-book. On top of scene-aware,
deduplicated frames it routes the channels a still image can't carry into text:

1. **Sync index** — every kept frame co-located with its timestamp and the
   dialogue line live at that moment (no more guessing which line goes with which
   image, no drift).
2. **Mechanical audio layer** — per beat: loudness, building/releasing, a local
   tempo estimate, brightness, and timbre-clustered labels so distinct non-speech
   sounds read as distinct (three different screams ≠ one `[SCREECH]`).
3. **Motion / camera grammar** — per shot: `static | pan | tilt | push-in |
   pull-out | handheld | whip-*` + a local cut-rate. A slow push-in, a static
   hold and a whip-pan stop looking like identical frames. Phase correlation
   is sub-pixel, so a push-in as slow as ~0.7%/s registers.
4. **Guaranteed dialogue** — if the source has no readable subtitles, dialogue is
   recovered automatically (see fallback chain below), so nothing ships mute.

All local, deterministic, no cloud model.

## Output (per film, in `crv-<slug>/`)
| file | what it is |
|---|---|
| `frames/` | deduped, uncapped scene frames (`frame_00001.jpg`…), 960 px wide, JPEG q3 since format 8 (640 px at default quality before: OCR truncated words, faces were small) |
| `INDEX.tsv` | canonical map: `frame · time · dialogue · motion · cuts_min · loudness · trend · tempo_bpm · brightness · sound · impact · field · voice · pan · cut · held_s · text` (+ `shot` after `vision_pass.py`, `ambience` after `sound_tags.py`); `text` = on-screen text read by Apple Vision (format 8); `held_s` = seconds this frame stands for before the next kept one |
| `sync.md` | human-readable, read top-to-bottom as a cut; the last column lists only what changed since the previous row |
| `text.tsv` | the kept frames that carry on-screen text: `frame · time · text` |
| `dialogue.srt` | the cues verbatim with their timecodes |
| `dialogue.tsv` | every cue with the kept frame it was spoken over: `n · start · end · frame · text` |
| `sheets/` + `sheets.tsv` | contact sheets, 30 frames each (6×5 tiles, 320 px, frame number and time burned in) and the sheet → frame-range index |
| `sounds.tsv` | distinct non-speech sounds (Sound-A/B/C + count/seconds/pitch/texture/placement) and LFE rumbles (Rumble-1/2/…, ≥2 s of sustained LFE excess, e.g. a 4-minute landing) |
| `MANIFEST.txt` | summary: where the dialogue came from, HDR handling, which layers ran or failed, stage timings, and the engine version + output format |
| `shots.tsv` | after `vision_pass.py`: one row per shot — `shot · start · end · frames · setting · people · scale · action · light · text` |
| `events.tsv` | every sound event and rumble with its time, duration, placement (+ `tag` after `sound_tags.py`) |

Frame names are zero-padded to 5 digits — films run 2–7k frames, and 3 digits
sorts wrong lexicographically past 999. `INDEX.tsv` is the canonical map either way.

`MANIFEST.txt` ends with `engine: <git describe> (output format N)`, so you can
tell which library entries predate a fix. Folders without that line are older
than format 3.

| format | what changed |
|---|---|
| 1 | 3-digit frame names (before 2026-09-21) |
| 2 | 5-digit frame names |
| 3 | push-in/pull-out were swapped and a pure zoom read `static` — fixed; sub-pixel motion; HDR sources tonemapped (were washed out); non-speech sounds split by a short pause now merge; VobSub/DVB subtitles OCR'd (2026-09-23) |
| 4 | scene-change score computed on the 640px frame instead of the full-size one (same `--scene` threshold, slightly different picks on a 2160p source); motion pass decodes in software (2026-11-01) |
| 5 | spatial audio layer (`impact · field · voice · pan` columns, sound placement); loudness on BS.1770 momentary LUFS instead of mono RMS dB; `cut` column marks hard cuts (2026-11-01) |
| 6 | zooms need both axes to diverge consistently and pans need most samples to agree (a locked-off film no longer reads push-in/pull-out/pan: Zone of Interest static 138 → 633); cuts need an absolute jump (was 88% of frames on static films); `events.tsv` (2026-11-01) |
| 7 | `dialogue` carries every cue since the previous frame (was the line live *at* the frame: 514 of 1688 cues reached INDEX on Obsession); a frame is kept at least every `--max-gap` 10 s inside held shots; `held_s`; `dialogue.srt` + `dialogue.tsv`; TSV fields never carry tab/newline/ASCII quote; `sheets/`; MANIFEST `dialogue coverage:` line (2026-11-01) |

Films encoded from HDR sources before format 3 (e.g. the 2160p Arrival and
Khan) have grey, low-contrast frames — median peak luma 128–160 of 255 vs
~250 for SDR — and are worth re-running.

## Spatial audio layer (format 5)
86% of the library is 5.1 (451 of 527 /cinema titles; 6 are 7.1, 52 stereo,
15 mono), and a 5.1 bed is already annotated by the mixer: dialogue lives in
FC, the LFE track is literally "feel it", the surrounds carry envelopment.
The spatial layer reads that instead of folding it to mono. Every channel
comes off the single decode at 8 kHz (`_spatial8k.s16le`, memmapped in
chunks) and gives four per-beat columns plus a placement per sound event:

| column | values | from |
|---|---|---|
| `impact` | `hit` (transient >10 dB over the window's own LFE level) · `hard` (LFE > median+4·MAD) · `low` (> median+2·MAD) · blank | LFE channel |
| `field` | `frontal` (fronts exceed rears by >15 dB) · `wide` (>6 dB) · `enveloping` | front vs surround level |
| `voice` | `centre` · `off-L` · `off-R` · `rear` · blank | where the 300–3400 Hz band sits; see below |
| `pan` | `L→R` · `R→L` · `front→rear` · `rear→front` · blank | >8 dB level movement across the beat, both ends above −45 dB |
| `sound` | `Sound-B@rear-L`, `Rumble-2@front-wide` | dominant channel of the event (`centre`, `front-L/R`, `rear-L/R`, `front-wide`, `all-around`); the *shortest* live sound wins the column, so a long rumble never masks the events inside it |

`voice` *places* speech, it does not detect it: a beat with a dialogue cue is
always placed (centre unless another channel is >6 dB above FC); a beat
without one is placed only when its speech band is in the film's top fifth.
Measured on Alien (1979) 5.1: 98% of subtitled beats placed, 12% of them
off-centre (the bridge radio traffic, exactly where the mix puts it), 17% of
unsubtitled beats placed. An energy threshold alone had missed 59% of
subtitled beats, which is why the cue gate exists.

`loudness` and `trend` are now on BS.1770 momentary LUFS from ffmpeg's
`ebur128` (a separate audio-only pass, ~8 s per film): silent < −50, quiet
< −36, loud > −24. Alien's percentiles 5/25/50/75/90 are −52/−40/−33/−28/−23.
Stereo sources get `voice` (L/R balance) and `pan` only; mono and unknown
layouts get none, and `MANIFEST.txt` says which (`spatial: …`). An untagged
6-channel track is assumed `5.1(side)` and marked `5.1(side)?`. Atmos
objects are not recoverable: E-AC-3 JOC and TrueHD decode to the bed, and
the bed is what this reads. Thresholds are relative to the film's own
distribution, so a quiet mix and a loud one read the same way.

## Vision pre-pass (`vision_pass.py`, optional, needs oMLX)
Claude still takes video only as images, so what it understands is bounded by
how the frames are framed. The pre-pass lets a local VLM look first:
```bash
set -a; . "$CRV_ENV_FILE"; set +a              # defines OMLX_API_KEY, never a flag
./.venv/bin/python vision_pass.py crv-movie     # --model, --base, --every 20, --limit N, --dry-run
```
It groups the kept frames into shots on the engine's `cut` column (older
folders: pixel jumps, which over-split long takes), captions each shot's
first frame plus one every 20 s inside a long take through oMLX's
OpenAI-style endpoint, and writes `shots.tsv`, a `shot` column in
`INDEX.tsv`, a `[sees] …` caption on each shot's first `sync.md` row, and a
`vision:` line in `MANIFEST.txt`. The captions live in `shots.tsv`; the
`sync.md` writer reads that sidecar itself (`attach_sidecars`), so a later
pass that rewrites `sync.md` (sound tags, redialogue, OCR) keeps them. Until
2026-11-02 the sound-tags pass dropped every caption from `sync.md`. Captions are compact JSON (setting, people
count, shot scale, action, light, readable on-screen text), parsed and
cleaned; a reply that is not JSON lands in `action` as raw text. On-screen
text comes from **Apple Vision OCR** on every captioned frame when the helper
is available; the VLM's own reading only stands when Vision finds nothing (a
VLM guesses words — "Lufkin Air School" for a letter reading "Luther Art
School" — Vision reads letters). Resumable:
captions append to `_vision.jsonl` and a rerun skips what it has.

Probed on an M5 Max (8 Alien frames, 2026-11-01): **Qwen3.6-35B-A3B-4bit**, the
default, answers in 0.3–0.5 s a frame once loaded (prompt cached);
Qwen3.8-27B-8bit is a little richer at 4 s; Qwen2.5-VL-7B is comparable to
the MoE and wraps JSON in code fences. The Qwen3.6/3.8 models think by
default and spend a small `max_tokens` on reasoning, returning no content —
the pass sends `chat_template_kwargs.enable_thinking=false`. A 2 h film is
400–1000 shots, so a few minutes on the M5 Max.

Measured twice (The Zone of Interest, Opus, ten structural questions, graded
by hand against the film; the graded runs quote the film's dialogue, so they
stay out of the public repo). Run 1: **8.0/20
from the plain INDEX, 16.5/20 with shots.tsv.** The pre-pass answered what
the index cannot see at all (what adjoins the garden, the museum ending,
on-screen text) and made the model more willing to say "not determinable"
where the data lacks a cause. Run 2, after the motion fix (format 6) and
`sound_tags.py`: **13.0/20 plain, 18.5/20 with shots.tsv** — the camera
style question and the off-screen-sound question both went to full marks.
`bench/vlm_bench.py` reruns it on any folder.

**Reader rule for dark shots:** the VLM reads backlit fog and blue as water
(Alien's planet surface became "underwater", "divers in vintage suits", "a
submarine"; the Space Jockey "soldiers confront a massive alien creature").
A caption on a `light=dark` shot is a second opinion, to be checked against
the contact sheet, not a fact. Since 2026-11-02 the prompt says so itself
("darkness, fog, haze, grain or blue light are not water; name the place as
the story shows it"): on six Alien planet shots that fixed five; a gamma-lifted
image alone fixed none; the sixth (a diver-like figure under a bright light)
still reads as underwater.

## Sound tags (`sound_tags.py`, optional, needs a uv script env)
The engine's Sound-A/B/C are timbre clusters and say nothing about *what* a
sound is, and its salience detector misses the faint off-screen soundtrack
under dialogue. `sound_tags.py` names sounds with zero-shot CLAP, gated by a
supervised AudioSet classifier:
```bash
uv run --script sound_tags.py crv-movie /path/to/movie.mkv   # --model, --window 5, --min-prob 0.15, --min-prom 0.1, --limit N, --no-ast
```
It is a PEP 723 script (torch + transformers on MPS, Python 3.12, kept out
of the engine venv; first run installs them and fetches
`laion/larger_clap_general`, ~2 GB, and `MIT/ast-finetuned-audioset-10-10-0.4593`,
~350 MB, into the Hugging Face cache). It embeds the ±5 s window around
every kept frame (both models are trained on 10 s clips) and scores it
against 72 film-sound prompts, writing an `ambience` column to `INDEX.tsv`,
a `tag` to every `events.tsv` row and to `sounds.tsv`, `hear=…` in
`sync.md`, and a `sound tags:` MANIFEST line.

Three things make it work on a film mix. On 5.1/7.1 sources the audio is the
mix *without the centre channel* (`pan=mono|c0=0.25*FL+0.25*FR+0.25*SL+0.25*SR`): the
off-screen soundtrack lives in L/R and the surrounds, dialogue in C, so
CLAP hears the ambience instead of the words (stereo uses L−R, mono the full
mix). Labels are scored by prominence over the film's own median per
label, with any label confident on a quarter of the film reported once as
the "bed" rather than on every frame. And **AST decides what kind of window
it is**, because CLAP's softmax must always pick something: a window AST
calls quiet gets no tag; one it calls music gets only music labels (the
vocabulary has homes for score texture: drone score, string pad, bass
pulse, ambient score, plus AST's own mood, e.g. scary music / opera); an
effects window keeps the film vocabulary, with AST's event in front when it
is confident (knock, gunshot, crickets), and a mechanical label CLAP likes
(helicopter, explosion, alarm) is dropped unless AST gives it at least a
weak vote. Measured on Hokum (2026, a horror film that is wall-to-wall low
synth drone in dark hotel rooms): CLAP alone tagged helicopter on 344
frames, heartbeat 309, alarm 145; gated, those are gone and the film reads
as drone score / string pad / scary music, with the quiet rooms blank.
Measured on The Zone of Interest: the camp drone reads as rumble /
industrial drone, the garden as birds, the river as river / water
splashing, the Berlin party as applause / cheering; faint gunshots and
screams register only sometimes. Events (`events.tsv`) are tagged on the
full mix over their own span.

**SDH sources lead.** When the cue track is SDH (3% of cues or 30 cues
carrying a bracketed sound description: `(chains rattling)`, `[ALARM
WAILING]`; speaker labels like `[PARKER]` are not descriptions), those
descriptions drive the `ambience` column and the event tags, and CLAP is
demoted to AST-confirmed events. On Alien, CLAP had written "cattle" on 786
of 4086 rows for the Nostromo hum and church organ / cheering / applause /
crying where there were none, while the cues already said chains, rain, fire,
explosion. "cattle" is out of the vocabulary. The MANIFEST `sound tags:` line
says when the cues led. `sdh_sounds.py` applies the same rule to a finished
package without the audio (CLAP labels dropped outright there, since which
ones had an AST vote was never stored).

## Dialogue fallback chain (automatic, in priority order)
1. **Sidecar** `.srt`/`.vtt` next to the source — `movie.srt`, then tagged
   names (`movie.en.srt`, `movie.eng.sdh.srt`, `movie.fr.srt`, …) ranked like
   embedded streams (English over foreign, full over forced, plain over SDH) —
   or an **embedded text** subtitle stream (subrip/ass/mov_text/webvtt). Used
   verbatim, timecodes preserved.
2. **Apple Vision OCR** of embedded **image-based** subtitles — the real
   theatrical lines, accurate on stylized/italic fonts where tesseract fails.
   - **PGS** (Blu-ray): decoded with pgsrip (`transcribe_pgs_vision`).
   - **VobSub / DVB / XSUB**, or PGS that pgsrip can't read: ffmpeg composites
     the stream onto a white canvas at 20 fps, keeps only frames where the
     picture changes, and OCRs those (`transcribe_bitmap_vision`); timing is
     exact to 50 ms.

   Both use the `vision_ocr` binary, built from `vision_ocr.swift` on first use
   and rebuilt when the source is newer. Results are put in reading order (top
   line first), and OCR batches run in parallel. A failed OCR run reports the
   actual exception in `MANIFEST.txt`'s subtitle note, not a bare "produced
   nothing".
3. **Whisper** (`whisper.cpp`, `small.en`) transcribing the audio — last resort
   when there are no subtitles at all. Run with `-mc 0` so the decoder never
   carries its own output forward as the next window's prompt: with the default
   context it can lock onto one sentence over music or dead air and emit it for
   hours. Measured on a 2h46m film, `large-v3` collapsed 11 minutes in and
   repeated a single line for the remaining 2h34m; `small.en` lost 25 min across
   18 loops. A loop reads as fluent prose downstream, so nothing else catches it.

`transcribe_whisper` scores its own output with `loop_damage()` and appends a
warning to the dialogue note (visible on stdout and in `MANIFEST.txt`) when
loops cover ≥5% of cues or ≥1 min. Only repeated *speech* counts — a sustained
`♪` or `(TRAIN WHISTLE BLOWING)` is normal annotation, not damage.

**Loop repair is automatic.** When damage is detected, each looped span is
re-cut on silence (±5s of padding) and re-transcribed in its own whisper
process — a fresh decoder state usually breaks the fixed point. Only the
damaged spans are re-run, not the film. If a retry still loops the original is
kept, because a second bad take is not an improvement and swapping silently
would hide it. The note reports how many spans were redone.

`--chunk-mins N` transcribes the whole file in N-minute segments up front, each
in its own process, so a loop cannot spread past one segment. Boundaries snap
to detected silence within ±90s of each mark rather than landing on the raw
offset — cutting on the offset slices a word in half at every seam and then
needs overlap-and-dedup to repair. Worth using on a film that stays bad after
automatic repair; the single-pass default is faster and usually enough.

**Silero VAD is opt-in (`--vad`), not the default.** whisper-cli can gate the
decoder with a voice-activity model, and it does kill every loop. Measured on
a 3h feature against the film's own subtitles (1739 speech cues, 13283 words;
M5 Max, 2026-11-01), that gating also drops real speech:

| config | word recall | looped cues | hallucinated min | secs |
|---|---|---|---|---|
| small.en (default) | 0.786 | 0.3% | 29 | 97 |
| small.en + VAD | 0.494 | 0 | 81 | 49 |
| small.en + VAD, tuned (thr 0.2, pad 800 ms) | 0.606 | 0 | 66 | 58 |
| large-v3-turbo | 0.828 | 5.8% (9.7 min lost) | 69 | 93 |
| large-v3-turbo + VAD | 0.505 | 0 | 70 | 44 |

Word recall = share of reference words found in whisper cues overlapping the
reference cue (±2s). `-mc 0` plus the loop repair already hold small.en's
loops near zero, so VAD buys nothing it needs and costs a third of the
dialogue. large-v3-turbo hears more words but loops and hallucinates more;
not worth it unattended.

`--no-transcribe` disables 2 & 3.

A row's `dialogue` is every cue that started since the previous kept frame
plus the one live at the frame (`lines_since`), joined with ` / ` in order,
so reading the rows top to bottom gives the whole script with each line once.
It used to be only the line live *at* the frame, and a short cue that
started and ended between two kept frames (a second or more apart) appeared
nowhere. The live line is the cue that spans the timestamp (latest-starting,
if cues overlap), else the nearest within 0.3 s. The cues are also written
verbatim to `dialogue.srt`, and `dialogue.tsv` ties each cue to the kept
frame it was spoken over (frame number first, so a line can be looked up by
frame count). `MANIFEST.txt` carries `dialogue coverage: N/N cues carried in
INDEX.tsv`; anything short of N/N is marked INCOMPLETE and the e2e test
requires N/N. No TSV field contains a tab, newline or ASCII
double quote (`"` becomes `”`): bare quotes in a subtitle line made csv-style
readers swallow the following fields. Readers using pandas should still pass
`quoting=csv.QUOTE_NONE` for older folders. Subtitle parsing
keeps numeric-only lines (a line that's just "1984"), strips ASS
`{\an8}`-style tags, and accepts VTT's `mm:ss.mmm` alongside srt's
`hh:mm:ss,mmm`.

## Run one film
```bash
./.venv/bin/python enhance_process.py "/path/to/movie.mkv" -o crv-movie
```
That is the only entry point. There is no `crv` launcher any more: the one the
venv ships is stock `claude-real-video`, which caps at 150 bare frames and
writes no `INDEX.tsv`, and a rebuilt venv silently fell back to it (the old
`deploy.sh` symlink hack is gone for the same reason). Uncapped by default
(every deduped frame kept), and **a frame is kept at least every `--max-gap`
seconds (10) even inside a held shot**: dedup used to collapse a static
two-minute shot to one row, so time stopped there and everything sampled per
frame (dialogue, loudness, voice, ambience) was sampled once for two
minutes. `held_s` says how long each frame stands for, and sync.md marks
`held Ns`. Works **in place** — no multi-GB `source.mp4` copy
is made, so cleanup is just deleting the source when done. The engine exits 1,
instead of printing "✓ Done", when the source is missing, when ffmpeg fails
during extraction, or when the frames stop more than a minute short of the
container's duration (a truncated file). Previously a partial extraction
shipped as a finished film, and `batch_encode.sh` then deleted the original.

**HDR.** PQ (HDR10, Dolby Vision profile 8) and HLG sources are detected from
the stream's transfer characteristic and tonemapped to SDR BT.709 in numpy
(`tonemap_hdr`): 203-nit reference white maps to ~241/255 (BT.2408), linear
below a knee, then a shoulder that keeps highlight detail up to a 1000-nit
master. Homebrew ffmpeg has neither `zscale` nor `libplacebo`, which is why
this isn't an ffmpeg filter. Dolby Vision **profile 5** (no HDR10 base layer)
can't be decoded correctly by anything here; the run warns, and `MANIFEST.txt`
says so.

**One decode per film.** The frame extraction's ffmpeg also writes the motion
pass's 3 fps 128×72 gray stream and the audio layer's 16 kHz mono PCM through
a `split` (`_extract_cmd`); the two analyses then run on those scratch files
in background threads while dedup and the dialogue chain proceed. Until
2026-11-01 each analysis decoded the whole film again on its own, and the
motion pass tried VideoToolbox first, which copies every 1080p frame back
from the GPU only for `fps=3` to drop seven of eight. Measured on Alien
(1979), 1080p h264, 1h56, embedded srt (`timings:` line in `MANIFEST.txt`):

| step | M5 Max (18 cores) | M1 Max |
|---|---|---|
| baseline: three decodes, motion via VideoToolbox | 448s (motion 448s, everything else inside it) | VT motion decode alone 745s |
| motion pass in software | — | 362s (extract 346s ∥ motion 301s) |
| + scene select on the 640px frame | 102s (extract 95s ∥ motion 93s) | 340s (extract 324s ∥ motion 320s) |
| + single decode | **75s** (extract 68s, motion 4s, audio 1s) | **218s** (extract 201s, motion 8s, audio 2s) |

The audio layer is byte-identical across these runs. Motion labels flip on
~10% of beats between runs whose gray streams differ by 0.02/255 — symmetric
pairs (handheld↔pull-out etc.) spread evenly over the film — which is a
boundary-sensitivity of the classifier, not a decode difference; treat a
single beat's tag as ±1 class.

Audio framing uses a strided view instead of a materialized index (was 1.84 GB
at 2h of audio, 2.76 GB at 3h); motion streams frame-by-frame instead of
holding a float32 stack (was 1.2 GB at 3h). `line_at`, `sound_at` and `dedup`
use a running-max early exit, bisect and numpy respectively.

`--hwdec` decodes for frame extraction with VideoToolbox (same timestamps;
retries in software on failure). Off by default and, on the numbers above,
leave it off: GPU decode plus the copy-back to the CPU filters is slower than
software decode for everything this pipeline does.

Flags: `--scene 0.30` `--fps-floor 1.0` (seconds between guaranteed frames —
upstream crv semantics, not fps) `--dedup-threshold 8` `--dedup-window 4`
`--no-audio` `--no-motion` `--no-transcribe` `--chunk-mins 0` `--hwdec`.

If a Homebrew Python upgrade breaks the venv ("bad interpreter"), re-point `.venv/bin/python3.14` at the stable
`/opt/homebrew/opt/python@3.14/bin/python3.14` and fix `home`/`executable` in
`.venv/pyvenv.cfg`.

## Copy from a NAS over ssh
Synology NAS boxes ship with the SFTP subsystem disabled, so `scp`/`rsync` fail.
Stream over SSH instead and **verify by byte size**:
```bash
ssh nas "stat -c%s '/path/to/<Movie>/<file>.mkv'"      # remote size
ssh nas "cat '/path/to/<Movie>/<file>.mkv'" > movie.mkv # copy
stat -f%z movie.mkv                                    # must match
```
`batch_encode.sh films.tsv` drives a list of films from a TSV
(`name<TAB>remote_path<TAB>slug`, `#` comments; see `films.example.tsv`). The
list is required (the old built-in one was a July 2026 session's). With
`PUBLISH_DIR=<media-root>/shared-media/phase-N` each finished folder is
moved there as `<slug>-crv`, whole, and its run log goes inside as `run.log`;
an occupied target is refused (`PUBLISH_FAIL`, output and original kept).
Already-complete films (`INDEX.tsv` present here or in the publish dir,
original gone) are skipped before any download. `batch_status.tsv` rows are
`name · state · frames · encode seconds`. **Copy-ahead**
downloads the next film while the current one encodes; a failed download no
longer cascades into the films after it, and a partial download resumes
(`tail -c +N` over ssh) instead of restarting. Each copy is verified by byte
size **and** an `ffprobe` duration check before encoding, then the original is
deleted. `CRV=./full_pipeline.sh` runs all three stages (engine, `vision_pass.py`,
`sound_tags.py`) as the encode step, so each published folder is complete
and the source is still present for the sound tags (2026-11-01: eight films,
6–13 min each on an M5 Max).
`CRV` (engine command), `NAS`, `NAS_LAN`, `MIN_FREE_GB` and `PUBLISH_DIR` env vars
override the defaults; hosts and paths for your site go in `site.env` (copy
`site.example.env`, gitignored), which the shell scripts source when present.
`NAS` defaults to `nas`; when `NAS_LAN` is set and accepts an ssh login (probed
with `ssh -o BatchMode=yes true`, so ssh-config aliases work) it is used
instead (a LAN address for a box whose usual name goes through a VPN). Measured
on 2026-11-01 against a Synology DS1517+: a single `ssh nas cat` stream runs
~30 MB/s and three concurrent streams share ~60 MB/s, whichever cipher and
whichever route; the limit is the NAS's ssh, not the 1 Gbps link. Budget
~35 s per GB for the first film; after that the prefetch hides the copy
behind the 12–15 min a film spends in the engine, captions and sound tags.

The list is validated before anything runs: a row without all three fields,
a slug outside `[A-Za-z0-9._-]`, or a `|` in a name refuses the whole list.
CRLF (Windows-saved) lists are fine, and an empty list exits cleanly. Each
download first checks free space on `originals/` for the remaining bytes plus
`MIN_FREE_GB` (default 5) of headroom. Ctrl-C also stops the background
prefetch and its ssh. Films live in
folders named `Title (Year) {imdb-…}`.

## Patch dialogue into a finished folder (no re-encode)
`redialogue.py <out_dir> <media|.srt|.vtt> [--chunk-mins N] [--lang en]` reuses
the existing frames + their timestamps and only fills the dialogue column:
```bash
# from a subtitle/OCR file:
python redialogue.py crv-movie /path/to/subs.srt
# or transcribe audio with Whisper:
python redialogue.py crv-movie /path/to/movie.mkv
```
`--chunk-mins` re-transcribes a film that's still bad after automatic repair,
without re-encoding. Inputs are validated before transcription starts,
`INDEX.tsv` is written atomically, and `sync.md` comes from the same writer a
fresh run uses.

For a PGS film whose source is remote, extract the sub stream and OCR it:
```bash
ssh nas "cat '<remote>.mkv'" | ffmpeg -i - -map 0:s:0 -c copy /tmp/subs.en.sup
python pgs_to_pngs.py /tmp/subs.en.sup /tmp/png     # render subtitle images + timing.tsv
./vision_ocr /tmp/png/*.png > /tmp/ocr.tsv          # Apple Vision OCR
python build_srt.py /tmp/png/timing.tsv /tmp/ocr.tsv /tmp/subs.srt
python redialogue.py crv-movie /tmp/subs.srt
```
`pgs_to_pngs.py` uses the engine's renderer and timing, so a subtitle with no
end of its own is held until the next one (max 4s), as in the automatic path.
It used to write `end = start`, a zero-length cue that no frame ever matched.

## In-place repairs (second review, 2026-11-02)
Passes that fix a finished package from its own frames and tables, no
source needed (except captions for new shots, which need the vision server).
All use `pkg.py` (atomic INDEX/MANIFEST writes, `sync.md` rebuilt from the
tables) and are idempotent.

| pass | what it does | MANIFEST line |
|---|---|---|
| `ocr_pass.py <pkg>` | (packages before format 8) Apple Vision OCR on **every** kept frame → `text.tsv`, INDEX `text` column, `text=…` in sync.md when it changes (vision_pass OCRs only a shot's first frame; Alien's SPECIAL ORDER 937 screen sat in a dark static shot). 4086 frames in 50 s with 4 workers. | `ocr:` |
| `dark_cuts.py <pkg>` | cuts the engine missed inside shots over 60 s: consecutive kept frames darker than 64/255 are contrast-normalised, motion-aligned by phase correlation and scored; threshold calibrated on the film's own engine cuts. Then `vision_pass.py` regroups shots and captions the new ones. Alien: 191 cuts in the 15 long dark shots (423 s tour → shots again); of 12 candidate pairs checked by eye, 9 were cuts and 3 camera moves. | `dark cuts:` |
| `sdh_sounds.py <pkg>` | SDH cue descriptions replace the CLAP ambience/event tags (see Sound tags) | `sound tags:` |
| `voice_gate.py <pkg>` | clears `voice` where no cue is live (the placer fired on strings in Alien's final log and end credits: 409 of 2025 placements) | `voice:` |
| `origin_patch.py <pkg> --origin … --duration … --bytes …` | which file was encoded: NAS path, container duration and size (MANIFEST used to record only the slug; the engine now writes these itself from `CRV_ORIGIN`, which `batch_encode.sh` sets) | `origin:`, `container:` |
| `cite_map.py <old_pkg> <new_pkg>` | old frame numbers → new frames by image match (24×14 thumbnails inside the letterbox, Pearson correlation; time-windowed when both packages have times). Alien's format-1 package (2500 frames, no times): median score 0.905, 31% below 0.8 — treat those as unmapped, they are mostly dark frames. | `cite map:` |
| `repair_pass.sh <pkg> [--origins origins.tsv] [--old <old_pkg>]` | all of the above in order, stamps `repaired:` only when every shot has a caption record | `repaired:` |
| `repair_all.sh [phase-dir …]` | `repair_pass.sh` over `shared-media/phase-*`, old packages from `crv-redo/old/`, under `$CRV_MEDIA` / `$CRV_REDO`, vision server on `$VLM_HOST` (needs an oMLX build that takes image input) | — |

Engine-side, the same review changed: the cut detector scales the gray diff
by luminance (`cut_mask(fdiff, lum)`, `lum_scale`), voice placement is gated
on a live cue whenever a cue track exists, `write_sync_md` carries captions
and OCR from the sidecars, `sdh_cues` flags an SDH track in the `subtitles:`
line. Format 8 closes the last gap: text that types in over a static dark
shot used to fall between kept frames (dedup + the 10 s floor; Alien's CREW
EXPENDABLE line was never extracted). The engine now OCRs **every extracted
frame before dedup** (Apple Vision, ~12 ms a frame) and a frame whose reading
carries a word the last kept frame's did not is kept, at most one a second;
the kept frames' text goes straight to the `text` column and `text.tsv`, so
`ocr_pass.py` is only for packages made before format 8 (`--no-ocr` turns
the stage off).

## Tests
```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest tests -q   # 88 tests, ~30s; synthetic clips via ffmpeg
bash tests/test_batch.sh                # batch driver against stubbed ssh/ffprobe/df/crv
```
- `test_pure.py`: subtitle parsing, loop detection, stream and sidecar
  ranking, `line_at`/`sound_at` (checked against brute force), motion labels,
  sound-event merging, silence pairing, PGS end resolution, tonemap reference
  points, MANIFEST contents.
- `test_media.py`: camera moves on a synthetic still (pan/tilt/push-in/pull-out
  /static), PQ grey patches at known luminances end to end, a truncated source
  exiting 1, a hand-built VobSub track (`tests/vobsub_fixture.py`) OCR'd into
  `INDEX.tsv`, Vision reading order, and a synthetic 5.1 clip with known
  placements (centre voice, LFE burst, rear noise, off-L voice) read back
  from `INDEX.tsv`. Vision tests skip off macOS.
- `test_e2e.py`: a full run incl. `redialogue`.
- `test_batch.sh`: happy path, download-fail, skip-complete, resume,
  encode-fail, empty list, malformed row, CRLF list, low disk, publish dir
  (move, skip-on-rerun, occupied target), missing list argument, seconds column.

CI (`.github/workflows/test.yml`) runs both suites on macOS for every push.

## Dependencies
- **ffmpeg / ffprobe** ≥ 5.1 (`brew install ffmpeg`) — decode, frames, audio,
  sub extract. Needs `-fps_mode vfr`; ffmpeg 9 dropped the older `-vsync`,
  which used to make extraction silently yield zero frames.
- **Python** (venv): `pip install -r requirements.txt` — numpy, scipy, pillow,
  pgsrip, opencv-python, pinned to the versions the tests last passed on.
- **Whisper**: `brew install whisper-cpp` (the binary is `whisper-cli`) + a
  model in `models/` (next to the script, or `~/Downloads/agent-tools/models`).
  Picked in order: `$CRV_WHISPER_MODEL`, `models/ggml-small.en.bin`, any
  `ggml-small*`, then whatever else is there (the Silero VAD file is skipped) —
  it used to take the alphabetically first file, which meant `large-v3`, the
  model that collapses (see fallback chain above).
- **Apple Vision OCR**: macOS + `swiftc` (Xcode CLT). `vision_ocr` auto-builds
  from `vision_ocr.swift` on first use; or set `CRV_VISION_OCR`.
- **tesseract** (`brew install tesseract`) — only if you use pgsrip's own OCR;
  Vision is preferred and far more accurate.

## Files
| file | role |
|---|---|
| `enhance_process.py` | the engine, the one entry point |
| `vision_ocr.swift` | Apple Vision OCR helper (compiled to `vision_ocr`) |
| `pgs_to_pngs.py` | render PGS subtitle images (manual OCR path) |
| `build_srt.py` | join OCR text + PGS timing → SRT |
| `redialogue.py` | patch dialogue into a finished folder |
| `vision_pass.py` | local-VLM shot captions for a finished folder (oMLX) |
| `sound_tags.py` | zero-shot CLAP sound names for a finished folder, AST-gated, SDH-led (uv script, torch on MPS) |
| `pkg.py` | package helpers for the in-place passes (INDEX/MANIFEST/sync.md) |
| `ocr_pass.py`, `dark_cuts.py`, `sdh_sounds.py`, `voice_gate.py`, `cite_map.py`, `origin_patch.py` | in-place repairs (see above) |
| `repair_pass.sh`, `repair_all.sh` | the repairs in order, one package / every phase |
| `bench/` | `vlm_bench.py` (INDEX vs INDEX+shots.tsv through `claude -p`) |
| `batch_encode.sh` | NAS copy-ahead batch driver |
| `full_pipeline.sh` | engine + vision pass + sound tags + OCR as one encode step for the driver |
| `films.example.tsv` | example film list for `batch_encode.sh` |
| `tests/` | pytest suite (`test_pure`, `test_media`, `test_e2e`, VobSub fixture) + `test_batch.sh` |
| `requirements*.txt` | pinned runtime deps; `-dev` adds pytest |
| `.github/workflows/test.yml` | CI: both suites on macOS |
