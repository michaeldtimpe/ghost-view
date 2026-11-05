#!/usr/bin/env python3
"""Enhanced crv pipeline.

Same scene-aware, deduplicated frames as `crv`, plus two things that make a
frame dump read like a cut instead of a shuffled flip-book:

  (1) SYNC INDEX  — every kept frame co-located with its real timestamp and the
      dialogue line that was live at that moment (from the video's own timecoded
      subtitles). No more guessing which line goes with which image.

  (2) MECHANICAL AUDIO LAYER — for each kept frame (a "beat"): loudness, whether
      the sound is building / releasing / steady, a local tempo estimate, and
      brightness; plus timbre-clustered labels so distinct non-speech sounds
      (three different screams) read as three DISTINCT things, not one [SCREECH].
      Pure numpy/scipy DSP — no external model.

Works on the file in place (no multi-gig source.mp4 copy). Everything the LLM
reads ends up in INDEX.tsv (canonical), sync.md (human), sounds.tsv, MANIFEST.txt.
"""
from __future__ import annotations
import argparse, glob, os, re, subprocess, sys, time


def run(cmd, **kw):
    # errors="replace": ffmpeg/ffprobe stderr is not guaranteed UTF-8 (filenames,
    # stream titles) and a decode error here would kill the run.
    kw.setdefault("errors", "replace")
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffprobe_val(video, entries, stream=None):
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "default=nw=1:nk=1", video]
    return run(cmd).stdout.strip()


def duration(v):
    try: return float(ffprobe_val(v, "format=duration"))
    except ValueError: return 0.0

def fps(v):
    raw = ffprobe_val(v, "stream=avg_frame_rate", "v:0")
    try:
        n, d = raw.split("/"); return float(n) / float(d) if float(d) else 25.0
    except (ValueError, ZeroDivisionError): return 25.0

def has_audio(v):
    return bool(ffprobe_val(v, "stream=codec_type", "a"))


# ffmpeg's s16le output follows the stream's channel layout order; these are
# the layouts the spatial layer knows how to name. Anything else is treated
# as "no spatial layer" rather than guessed.
CHANNEL_NAMES = {
    "mono": ["M"], "stereo": ["L", "R"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
    "7.1(wide-side)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
}


def audio_info(v):
    """First audio stream: {codec, channels, layout, names} -- names is the
    channel order of a raw decode, or None when the layout is unknown."""
    import json
    out = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
               "stream=codec_name,channels,channel_layout", "-of", "json", v]).stdout
    try:
        st = (json.loads(out).get("streams") or [{}])[0]
    except (ValueError, TypeError):
        st = {}
    nch = int(st.get("channels") or 0)
    # an untagged 6-channel track (PCM remux, some DTS) is taken as 5.1(side);
    # the '?' is kept so MANIFEST shows it was assumed, not read
    layout = st.get("channel_layout") or {1: "mono", 2: "stereo", 6: "5.1(side)?"}.get(nch, "")
    names = CHANNEL_NAMES.get(layout.rstrip("?"))
    if names is not None and len(names) != nch:
        names = None
    return {"codec": st.get("codec_name", ""), "channels": nch, "layout": layout, "names": names}


def tsv_safe(text):
    """A TSV field a naive reader can trust: no tab, newline or ASCII double
    quote. Bare quotes in a subtitle line ("1984") made csv-style readers
    (pandas, csv.reader) swallow the following fields; typographic quotes
    read the same and no parser treats them as quoting."""
    text = re.sub(r"[\t\r\n]+", " ", str(text or ""))
    return text.replace('"', "\u201d").strip()


def hms(t):
    t = max(0.0, float(t)); h = int(t // 3600); m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------- frames + timestamps
HDR_TRANSFERS = {"smpte2084": "pq", "arib-std-b67": "hlg"}


def video_color(v):
    """Colour facts about v:0 that decide how frames are rendered:
    {transfer, primaries, width, height, hdr: 'pq'|'hlg'|None, dv_profile,
    dv_compat}. dv_* come from a Dolby Vision configuration record, if any."""
    import json
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,color_transfer,color_primaries:stream_side_data="
               "dv_profile,dv_bl_signal_compatibility_id", "-of", "json", v]).stdout
    try:
        st = (json.loads(out).get("streams") or [{}])[0]
    except (ValueError, TypeError):
        st = {}
    dv = next((sd for sd in st.get("side_data_list", []) or [] if "dv_profile" in sd), {})
    trc = st.get("color_transfer", "") or ""
    return {"transfer": trc, "primaries": st.get("color_primaries", "") or "",
            "width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
            "hdr": HDR_TRANSFERS.get(trc),
            "dv_profile": dv.get("dv_profile"),
            "dv_compat": dv.get("dv_bl_signal_compatibility_id")}


# PQ (SMPTE ST 2084) constants
_M1, _M2 = 2610 / 16384, 2523 / 4096 * 128
_C1, _C2, _C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
# linear BT.2020 -> BT.709 primaries
_BT2020_TO_709 = ((1.6605, -0.5876, -0.0728),
                  (-0.1246, 1.1329, -0.0083),
                  (-0.0182, -0.1006, 1.1187))


def tonemap_hdr(rgb16, transfer, ref_white=203.0, knee=0.75, peak=1000.0):
    """HDR R'G'B' (HxWx3 uint16, BT.2020, full range) -> SDR sRGB uint8.

    Without this a PQ/HLG source is written as if it were SDR: grey whites,
    lifted blacks, drained colour (the 2160p outputs peaked at luma 128-160 of
    255). Homebrew ffmpeg has neither zscale nor libplacebo, so it's done here.
    ref_white 203 nits = SDR 100% (BT.2408). Below `knee` the mapping is
    linear; above it an extended-Reinhard shoulder (slope 1 at the knee) that
    reaches 1.0 only at `peak` nits, so highlights up to a typical 1000-nit
    master keep their gradation instead of clipping. (A tanh shoulder was
    tried first: everything over ~400 nits came out 255.)"""
    import numpy as np
    x = rgb16.astype(np.float32) / 65535.0
    if transfer == "pq":
        p = np.power(x, 1 / _M2)
        nits = 10000.0 * np.power(np.maximum(p - _C1, 0) / (_C2 - _C3 * p), 1 / _M1)
    elif transfer == "hlg":
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        e = np.where(x <= 0.5, x * x / 3.0, (np.exp((x - c) / a) + b) / 12.0)
        ys = e @ np.array([0.2627, 0.6780, 0.0593], np.float32)
        nits = peak * np.power(np.maximum(ys, 1e-6), 0.2)[..., None] * e   # OOTF, gamma 1.2
    else:
        raise ValueError(f"not an HDR transfer: {transfer}")
    lin = np.maximum((nits / ref_white) @ np.array(_BT2020_TO_709, np.float32).T, 0)
    y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sx = np.maximum(y - knee, 0) / (1 - knee)
    smax = max((peak / ref_white - knee) / (1 - knee), 1e-3)
    yt = np.where(y <= knee, y, knee + (1 - knee) * sx * (1 + sx / smax ** 2) / (1 + sx))
    lin = np.clip(lin * (yt / np.maximum(y, 1e-6))[..., None], 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (srgb * 255 + 0.5).astype(np.uint8)


def _select_vf(scene, every_n):
    """Scene-change select + showinfo. The caller puts its scale *before* this:
    the scene score is then computed on the 640px frame instead of the full
    1080p/2160p one, which is the same decision on a much smaller picture
    (output format 4; format 3 scored at full size, after select)."""
    return f"select='gt(scene,{scene})+not(mod(n,{every_n}))',showinfo"


# the motion pass's stream: 3 fps, 128x72 gray (see _motion_series)
MOTION_FPS, MOTION_W, MOTION_H = 3, 128, 72
AUDIO_SR = 16000
SPATIAL_SR = 8000     # the spatial layer's channels: nothing in it needs the top octaves


def _extract_cmd(video, hw, scale, vf, frames_out, side=None, frames_fmt=None):
    """ffmpeg argv for the single decode pass. `side` may name files for the
    motion pass's gray stream ("motion") and the audio layer's 16k mono PCM
    ("audio"); they are split off the same decode. Before this each analysis
    decoded the whole film again on its own -- three 60 GB reads, and on an M1 Max
    two concurrent software decodes were the entire wall time of a run
    (extract 346s + motion 301s overlapped, total 362s)."""
    side = side or {}
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if hw: cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", video]
    fo = vf + (f",format={frames_fmt}" if frames_fmt else "")
    if side:
        graph = f"[0:v]{scale}"
        if side.get("motion"):
            graph += (f",split=2[f][m];[f]{fo}[fo];"
                      f"[m]fps={MOTION_FPS},scale={MOTION_W}:{MOTION_H},format=gray[mo]")
        else:
            graph += f",{fo}[fo]"
        cmd += ["-filter_complex", graph, "-map", "[fo]"]
    else:
        cmd += ["-vf", f"{scale},{fo}"]
    # -fps_mode vfr, not -vsync: ffmpeg 9 removed the old alias. Output options
    # bind to the output that follows them.
    cmd += ["-fps_mode", "vfr"] + frames_out
    if side.get("motion"):
        cmd += ["-map", "[mo]", "-f", "rawvideo", side["motion"]]
    if side.get("audio"):
        cmd += ["-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "s16le", side["audio"]]
    if side.get("spatial"):
        # every channel, native order (see CHANNEL_NAMES), for the spatial layer
        cmd += ["-map", "0:a:0", "-ar", str(SPATIAL_SR), "-f", "s16le", side["spatial"]]
    return cmd


# kept-frame width and JPEG quality. 640 px at ffmpeg's default quality
# (~22 KB a frame) truncated OCR ("EXPENDA") and left faces small for a
# reader; 960 px at q3 is ~4x the bytes (a feature ~350 MB) and legible.
FRAME_W, FRAME_Q = 960, 3


def _extract_sdr(video, frames_dir, vf, hw, side=None):
    p = run(_extract_cmd(video, hw, f"scale={FRAME_W}:-2", vf,
                         ["-q:v", str(FRAME_Q), os.path.join(frames_dir, "raw_%05d.jpg")], side))
    return p.returncode, p.stderr


def _extract_hdr(video, frames_dir, vf, hw, color, side=None):
    """Same selection, but frames come out as 16-bit RGB on a pipe and are
    tonemapped here one at a time -- 16-bit files for a whole film would be
    ~14 GB of scratch."""
    import threading
    import numpy as np
    from PIL import Image
    W = FRAME_W
    iw, ih = color["width"] or 1920, color["height"] or 1080
    H = max(2, int(round(W * ih / iw / 2)) * 2)
    cmd = _extract_cmd(video, hw, f"scale={W}:{H}:in_color_matrix=bt2020", vf,
                       ["-f", "rawvideo", "-"], side, frames_fmt="rgb48le")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    # drain stderr concurrently: showinfo writes a line per frame, and a full
    # pipe would stall ffmpeg while we block reading stdout.
    t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    t.start()
    nb, k = W * H * 3 * 2, 0
    try:
        while True:
            buf = p.stdout.read(nb)
            if len(buf) < nb:
                break
            k += 1
            rgb = np.frombuffer(buf, dtype="<u2").reshape(H, W, 3)
            Image.fromarray(tonemap_hdr(rgb, color["hdr"])).save(
                os.path.join(frames_dir, f"raw_{k:05d}.jpg"), quality=85)
    finally:
        p.stdout.close()
        rc = p.wait()
        t.join()
    return rc, (err[0] if err else b"").decode("utf-8", "replace")


def extract_frames_ts(video, frames_dir, scene, fps_floor, color=None, hwdec=False,
                      side=None):
    """One chronological pass. `showinfo` logs each kept frame's pts_time; the
    k-th log line is raw_{k}.jpg. Returns [(raw_path, ts_seconds), ...], or []
    if ffmpeg failed -- a partial set of frames is not a result. `side`: see
    _extract_cmd; the motion/audio files are written by the same decode.

    hwdec decodes with VideoToolbox first (same timestamps), retrying in
    software on a non-zero exit. Off by default: on a 20s 4K HEVC test clip
    VT decode alone was 1.8x faster, but downloading 4K frames for the
    software select filter ate the gain (4.5s vs 4.0s). Worth an A/B on a real
    high-bitrate film before turning it on."""
    os.makedirs(frames_dir, exist_ok=True)
    color = color if color is not None else video_color(video)
    vf = _select_vf(scene, max(1, round(fps(video) * fps_floor)))
    for hw in ((True, False) if hwdec else (False,)):
        # A crashed earlier run (or the failed hw attempt) leaves frames behind;
        # those would shift the k-th file away from the k-th showinfo line and
        # misalign every frame<->timestamp pair.
        for pat in ("raw_*.jpg", "tmp_*.jpg", "frame_*.jpg"):
            for stale in glob.glob(os.path.join(frames_dir, pat)):
                try: os.remove(stale)
                except OSError: pass
        if color.get("hdr"):
            rc, log = _extract_hdr(video, frames_dir, vf, hw, color, side)
        else:
            rc, log = _extract_sdr(video, frames_dir, vf, hw, side)
        if rc == 0:
            break
        what = "hardware-decoded" if hw else "software"
        print(f"{'warning' if hw else 'error'}: {what} frame extraction failed "
              f"(exit {rc}){'; retrying in software' if hw else ''}", file=sys.stderr)
        if not hw:
            for ln in log.splitlines()[-12:]:
                print(f"  {ln}", file=sys.stderr)
            return []
    ts = [float(m) for m in re.findall(r"pts_time:(-?[0-9.]+)", log)]
    raws = sorted(glob.glob(os.path.join(frames_dir, "raw_*.jpg")))
    # showinfo count should equal file count; zip defensively to the shorter.
    n = min(len(ts), len(raws))
    return list(zip(raws[:n], ts[:n]))


def ocr_words(text):
    """The words an OCR reading carries, for the dedup keep rule: upper-cased
    runs of 3+ letters/digits. Short fragments and punctuation are noise."""
    return {w.upper() for w in re.findall(r"[A-Za-z0-9]{3,}", text or "")}


def dedup(frames_ts, frames_dir, threshold=8.0, window=4, max_gap=10.0,
          ocr=None, kept_text=None, text_gap=1.0):
    """Drop near-duplicate frames by real pixel difference against a sliding
    window of the last `window` kept frames (crv's algorithm), carrying each
    survivor's timestamp through the rename. Uncapped — every distinct frame is
    kept. max_gap: a frame is kept regardless of similarity once this many
    seconds have passed since the last kept one (the next fps-floor candidate,
    so up to max_gap + fps_floor), so a held shot still ticks:
    without it a static two-minute shot collapsed to one row and the
    dialogue, loudness, voice and ambience sampled per frame were sampled
    once for two minutes (Obsession 2026: 514 of 1688 cues reached INDEX).
    ocr: {frame basename without extension: OCR text} for the extracted
    frames; a frame whose reading carries a word the last kept frame's did
    not is kept too (at most one every text_gap seconds -- 1 s, the fps
    floor: at 2 s Alien's CREW EXPENDABLE, on screen for two seconds after
    the previous keep, was rate-limited away), so text that types in over a
    static shot is not collapsed with the blank screen before it.
    kept_text, if given, is filled with {new frame name: text}.
    Returns [(frame_name, ts), ...]."""
    from PIL import Image
    import numpy as np
    def sig(path, size=16):
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB").resize((size, size)), dtype=np.int16)
    def pct(a, b, tol=25):
        # share of pixels whose worst channel moved by more than tol
        return 100.0 * float((np.abs(a - b).max(axis=2) > tol).mean())
    def text_of(path):
        return (ocr or {}).get(os.path.splitext(os.path.basename(path))[0], "")

    kept, recent = [], []
    last_t = None
    last_words = set()
    for path, t in frames_ts:
        h = sig(path)
        d = min((pct(h, k) for k in recent), default=None)
        tick = last_t is not None and max_gap > 0 and t - last_t >= max_gap
        words = ocr_words(text_of(path)) if ocr else set()
        new_text = bool(words - last_words) and (last_t is None or t - last_t >= text_gap)
        if d is None or d > threshold or tick or new_text:
            last_t = t
            kept.append((path, t)); recent.append(h)
            if len(recent) > window: recent.pop(0)
            if ocr:
                last_words = words
        else:
            os.remove(path)
    # rename survivors to frame_00001.jpg... preserving order + timestamp.
    # 5 digits, not 3: a feature keeps thousands of frames, and frame_1000.jpg
    # would sort before frame_999.jpg everywhere a glob or a shell lists them.
    kept.sort(key=lambda x: x[1])
    for i, (path, t) in enumerate(kept, 1):
        os.rename(path, os.path.join(frames_dir, f"tmp_{i:05d}.jpg"))
    out = []
    for i, (path, t) in enumerate(kept, 1):
        name = f"frame_{i:05d}.jpg"
        os.rename(os.path.join(frames_dir, f"tmp_{i:05d}.jpg"), os.path.join(frames_dir, name))
        out.append((name, t))
        if kept_text is not None and ocr:
            kept_text[name] = text_of(path)
    return out


# ---------------------------------------------------------------- subtitles -> timed cues
def parse_srt(path):
    """Parse .srt/.vtt into [(start,end,text)] with timecodes preserved."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    cues = []
    # hours are optional: WebVTT writes mm:ss.mmm for anything under an hour.
    TC = r"(?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3}"
    for block in re.split(r"\n\s*\n", raw):
        times = re.search(rf"({TC})\s*-->\s*({TC})", block)
        if not times: continue
        def secs(x):
            parts = x.replace(",", ".").split(":")
            if len(parts) == 2: parts = ["0"] + parts
            h, m, s = parts; return int(h)*3600+int(m)*60+float(s)
        txt, seen_tc = [], False
        for ln in block.splitlines():
            ln = ln.strip().lstrip("﻿")
            if "-->" in ln: seen_tc = True; continue
            # a digit-only line is the cue *index* only before the timecode;
            # after it, "1984" is dialogue and must survive.
            if not ln or ln.startswith("WEBVTT") or (ln.isdigit() and not seen_tc): continue
            ln = re.sub(r"<[^>]+>", "", ln)          # HTML/VTT tags
            txt.append(re.sub(r"\{\\[^}]*\}", "", ln))  # ASS override blocks {\an8}
        text = " ".join(txt).strip()
        if text: cues.append((secs(times.group(1)), secs(times.group(2)), text))
    return cues


def _is_marker(t):
    """Non-speech annotation: bare music glyphs, or a wholly bracketed direction.
    Whisper repeats these legitimately across a sustained cue, so loop scoring
    and loop repair must both ignore them."""
    t = t.strip()
    # Strip every bracketed group, not just an outer pair: "(SIGHS) I know."
    # leaves real speech behind and is a line, not an annotation.
    bare = re.sub(r"[\[(][^\])]*[\])]", " ", t)
    return (not t
            or not re.search(r"[A-Za-z]", t.strip("♪#*~ "))
            or not re.search(r"[A-Za-z]", bare))


_SDH_SPEAKER = re.compile(r"^[A-Z][A-Za-z.'\- ]{0,24}$")


def sdh_groups(text):
    """The sound descriptions in one cue's text: bracketed or parenthesised
    groups that describe a sound, not a speaker. "(CHAINS RATTLING)" and
    "[music playing]" count; "[PARKER]" / "(Ripley)" (a name, title case or
    caps, no verb) do not. Returned lowercased, inner whitespace collapsed."""
    out = []
    for g in re.findall(r"[\[(]([^\])]*)[\])]", text or ""):
        g = re.sub(r"\s+", " ", g).strip(" :,-")
        if not g or not re.search(r"[A-Za-z]", g):
            continue
        words = g.split()
        # a speaker label: up to three capitalised/caps words with no verb-ish
        # ending anywhere -- [PARKER], (Ripley), [COMPUTER VOICE]. "[NOISES
        # STOP]", "[RATTLING]", "(sighs)" and "(over radio)" all stay.
        if (len(words) <= 3 and all(w[:1].isupper() for w in words)
                and not any(re.search(r"(ing|s|ed)$", w.lower()) for w in words)):
            continue
        out.append(g.lower())
    return out


def sdh_cues(cues):
    """[(start, end, description)] for every sound description in the cue
    track, and whether the track reads as SDH at all (3% of cues carrying one,
    or 30 in total): on such a source the captioner's own words for the sound
    beat any audio tagger (Alien's 'chains', 'rain', 'fire', 'explosion' were in
    the cues while CLAP wrote 'cattle' on 786 rows for the Nostromo hum)."""
    out = []
    n = 0
    for s, e, t in cues:
        gs = sdh_groups(t)
        if gs:
            n += 1
            out.extend((s, e, g) for g in gs)
    is_sdh = bool(cues) and (n >= 30 or n >= 0.03 * len(cues))
    return out, is_sdh


def _runs(cues, min_run=3, max_period=3):
    """Yield (i, j) inclusive index ranges of looped *speech* cues: the same cue
    >=min_run times in a row, or a short cycle of 2-3 cues repeated at least
    three times over (The Endless: "I can't help it." / "Just leave before I
    get used to it." alternating every second for 20 s). Single source of
    truth for loop scoring and loop repair."""
    i = 0
    while i < len(cues):
        best_j, best_p = i, 1
        for p in range(1, max_period + 1):
            j = i + p - 1
            while j + 1 < len(cues) and cues[j + 1][2].strip() == cues[j + 1 - p][2].strip():
                j += 1
            if j - i + 1 >= max(min_run, 3 * p) and j > best_j:
                best_j, best_p = j, p
        if best_j > i and any(not _is_marker(cues[k][2]) for k in range(i, i + best_p)):
            yield i, best_j
            i = best_j + 1
        else:
            i += 1


def _loop_spans(cues, min_run=3):
    """[(start, end)] of each run of repeated speech -- the spans worth re-doing."""
    return [(cues[i][0], cues[j][1]) for i, j in _runs(cues, min_run)]


def _silence_seams(wav, noise="-30dB", dur=0.6):
    """Midpoints of detected silences -- safe places to cut without slicing a word."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", wav, "-af",
                        f"silencedetect=noise={noise}:d={dur}", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    return _seams_from_log(r.stderr)


def _seams_from_log(log):
    """Pair silencedetect's start/end events in the order they occur. Zipping
    two independent lists misaligns every later pair if one side has an extra
    or unparsed entry (a negative start near t=0, a trailing start at EOF)."""
    seams, start = [], None
    for kind, val in re.findall(r"silence_(start|end): (-?[0-9.]+)", log):
        if kind == "start":
            start = max(0.0, float(val))
        elif start is not None:
            seams.append((start + float(val)) / 2)
            start = None
    return seams


def _snap(t, seams, window=90.0, after=None):
    """Nearest silence to t, else t. Cutting on the raw offset slices a word in
    half at every boundary; snapping to a pause avoids needing overlap+dedup."""
    near = [s for s in seams if abs(s - t) <= window and (after is None or s > after + 60)]
    return min(near, key=lambda s: abs(s - t)) if near else t


def _threads():
    return min(8, os.cpu_count() or 4)


def _model_roots():
    here = os.path.dirname(os.path.realpath(__file__))
    return (os.path.join(here, "models"), os.path.expanduser("~/Downloads/agent-tools/models"))


def vad_model():
    """Path of a Silero VAD model for whisper-cli (--vad), or "" if none.
    $CRV_VAD_MODEL first, then ggml-silero*.bin in the model roots."""
    import glob as _g
    env = os.environ.get("CRV_VAD_MODEL", "")
    if env and os.path.exists(env):
        return env
    for root in _model_roots():
        hits = sorted(_g.glob(os.path.join(root, "ggml-silero*.bin")))
        if hits:
            return hits[0]
    return ""


def whisper_cmd(exe, model, wav, pref, lang, threads, vad=""):
    """whisper-cli argv shared by every call site. -mc 0: never carry decoded
    text forward as the next window's prompt (the loop mechanism). vad: a
    Silero model path; with it whisper decodes only detected speech, so music
    and dead air -- where the fixed point forms -- never reach the decoder."""
    cmd = [exe, "-m", model, "-f", wav, "-osrt", "-of", pref, "-l", lang,
           "-t", str(threads), "-mc", "0", "-np"]
    if vad:
        cmd += ["--vad", "-vm", vad]
    return cmd


def _whisper_span(exe, model, wav, out_dir, lang, start, end, tag, threads=None, vad=""):
    """Transcribe [start,end) in its own process, returned on the full timeline.

    A fresh process means a fresh decoder state, so a loop inside one span cannot
    propagate past its boundary -- that containment is the whole point."""
    piece = os.path.join(out_dir, f"_wspan{tag}.wav")
    pref = os.path.join(out_dir, f"_wspan{tag}")
    # Input seeking: exact on PCM wav, and far cheaper than decoding to the cut.
    # -t, not -to: as an *input* option -to is measured from the file start, so
    # pairing it with -ss silently produces the wrong length.
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start:.3f}", "-t", f"{max(0.0, end - start):.3f}", "-i", wav,
                    "-ac", "1", "-ar", "16000", piece], check=False)
    cues = []
    if os.path.exists(piece):
        subprocess.run(whisper_cmd(exe, model, piece, pref, lang, threads or _threads(), vad),
                       check=False)
        srt = pref + ".srt"
        if os.path.exists(srt):
            cues = [(s + start, e + start, t) for s, e, t in parse_srt(srt)]
        for tmp in (piece, srt):
            try: os.remove(tmp)
            except OSError: pass
    return cues


def _repair_loops(exe, model, wav, out_dir, lang, cues, seams, pad=5.0, vad=""):
    """Re-transcribe just the looped spans, in isolation, and splice them back.

    Keeps the original span if the retry still loops -- a second bad take is not
    an improvement, and silently swapping one for the other would hide that."""
    fixed = 0
    for k, (s, e) in enumerate(_loop_spans(cues), 1):
        a = _snap(max(0.0, s - pad), seams, window=30.0)
        b = _snap(e + pad, seams, window=30.0)
        new = _whisper_span(exe, model, wav, out_dir, lang, a, b, f"fix{k:02d}", vad=vad)
        if not new:
            continue
        nfrac, _, _, nruns = loop_damage(new)
        if nruns and nfrac >= 0.30:
            continue
        cues = [c for c in cues if c[1] <= a or c[0] >= b] + new
        cues.sort(key=lambda c: c[0])
        fixed += 1
    return cues, fixed


def loop_damage(cues, min_run=3):
    """Detect decoder loops -- runs of identical consecutive cues.

    -mc 0 prevents these, but a loop is indistinguishable from real dialogue
    downstream: it reads as fluent text and silently replaces whole stretches of
    a film. Cheap to check, so check rather than trust.

    Only repeated *speech* counts. Whisper legitimately emits the same non-speech
    annotation over a long music cue or a sustained sound, so scoring those would
    fire on healthy transcripts -- and a detector that cries wolf gets ignored.

    Returns (fraction_of_cues, seconds_lost, worst_text, n_runs).
    """
    if not cues:
        return 0.0, 0.0, "", 0
    runs = n_looped = 0
    lost = 0.0
    worst, worst_n = "", 0
    for i, j in _runs(cues, min_run):
        n = j - i + 1
        runs += 1
        n_looped += n
        lost += cues[j][1] - cues[i][0]
        if n > worst_n:
            worst_n, worst = n, cues[i][2].strip()
    return n_looped / len(cues), lost, worst, runs


def transcribe_whisper(media, out_dir, lang="en", chunk_mins=0.0, vad=""):
    """Fallback dialogue when the source has no readable subtitles: transcribe the
    audio with whisper.cpp (whisper-cli) into a timecoded transcript. Returns
    (cues, note). Degrades to ([], reason) if whisper/model/audio unavailable.

    chunk_mins > 0 transcribes in silence-aligned segments from the start, each in
    its own process, so a loop can never spread beyond one segment. Left at 0 the
    file is done in a single pass and only damaged spans are re-cut and re-run.

    vad: "" (default) decodes the whole track; "auto" gates with the installed
    Silero model (see vad_model); a path uses that model. Off by default on
    the evidence: on a 3h feature scored against its own subtitles, VAD took
    small.en's word recall from 0.79 to 0.49 and large-v3-turbo's from 0.83 to
    0.51 while hallucinated minutes went up (2026-11-01, Apocalypse Now). It
    removed every decoder loop, but -mc 0 already held those to 0.3% of cues;
    the loop repair handles the rest. Opt in with --vad for an experiment."""
    import shutil, glob as _g
    exe = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    if not exe:
        return [], "no subtitles; whisper-cli not installed"
    model = os.environ.get("CRV_WHISPER_MODEL", "")
    if not model or not os.path.exists(model):
        cands = []
        for root in _model_roots():
            # the Silero VAD model is ggml-*.bin too, and is not a whisper model
            cands += sorted(f for f in _g.glob(os.path.join(root, "ggml-*.bin"))
                            if "silero" not in os.path.basename(f) and "vad" not in os.path.basename(f))
        # Alphabetical order hands us large-v3, which is exactly the model
        # documented to collapse into hour-long loops. Prefer small.en.
        def _rank(p):
            b = os.path.basename(p)
            return (0 if b == "ggml-small.en.bin" else 1 if b.startswith("ggml-small") else 2, b)
        cands.sort(key=_rank)
        model = cands[0] if cands else ""
    if not model:
        return [], "no subtitles; no whisper model found"
    vad = vad_model() if vad == "auto" else (vad or "")
    wav = os.path.join(out_dir, "_whisper16k.wav")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", media,
                    "-vn", "-ac", "1", "-ar", "16000", wav], check=False)
    if not os.path.exists(wav):
        return [], "no subtitles; audio extract failed"
    dur = duration(wav) or 0.0
    seams, how = None, "single pass"
    tmps = [wav]

    # finally, not a trailing loop: a crash mid-transcribe otherwise leaves a
    # several-hundred-MB 16k wav (and its srt) behind in the output dir.
    try:
        if chunk_mins and dur > chunk_mins * 60:
            seams = _silence_seams(wav)
            bounds, t = [0.0], chunk_mins * 60
            while t < dur - 60:
                bounds.append(_snap(t, seams, after=bounds[-1]))
                t = bounds[-1] + chunk_mins * 60
            bounds.append(dur)
            cues = []
            for i, (s, e) in enumerate(zip(bounds, bounds[1:]), 1):
                cues += _whisper_span(exe, model, wav, out_dir, lang, s, e, f"{i:03d}", vad=vad)
            how = f"{len(bounds)-1} segments"
        else:
            pref = os.path.join(out_dir, "_whisper")
            # -mc 0: do not carry decoded text forward as the next window's prompt.
            # With the default context, whisper can lock onto a sentence over music or
            # dead air and emit it for hours -- large-v3 died 11 minutes into a 2h46m
            # film this way, small.en lost 25 minutes across 18 separate loops.
            subprocess.run(whisper_cmd(exe, model, wav, pref, lang, _threads(), vad),
                           check=False)
            srt = pref + ".srt"
            cues = parse_srt(srt) if os.path.exists(srt) else []
            tmps.append(srt)

        # Loops that survived anyway: re-cut just those spans on silence and redo
        # them in isolation. Bounded work -- only the damage is re-run, not the film.
        repaired = 0
        frac, lost, worst, runs = loop_damage(cues)
        if cues and runs and (frac >= 0.05 or lost >= 60):
            if seams is None:
                seams = _silence_seams(wav)
            cues, repaired = _repair_loops(exe, model, wav, out_dir, lang, cues, seams, vad=vad)
            frac, lost, worst, runs = loop_damage(cues)
    finally:
        for tmp in tmps:
            try: os.remove(tmp)
            except OSError: pass

    if cues:
        note = (f"Whisper transcript ({os.path.basename(model)}, "
                f"{'VAD ' + os.path.basename(vad) + ', ' if vad else ''}{how}, "
                f"{len(cues)} lines)")
        if repaired:
            note += f"; re-ran {repaired} looped span(s) in isolation"
        if runs and (frac >= 0.05 or lost >= 60):
            note += (f" -- WARNING: {runs} decoder loop(s) span {frac:.0%} of cues "
                     f"and {lost/60:.1f} min, repeating {worst[:48]!r}. That audio "
                     f"is lost, not transcribed; re-run with another model or a "
                     f"smaller --chunk-mins.")
        return cues, note
    return [], "no subtitles; whisper produced nothing"


def _vision_ocr_bin():
    """Locate the Apple Vision OCR helper binary, building it from
    vision_ocr.swift on first use if only the source is present (macOS + swiftc),
    and rebuilding it when the source is newer than the binary -- otherwise a
    fix to the .swift never reaches a machine that already has a build.
    Set CRV_VISION_OCR to override. Returns a path or None."""
    import shutil
    env = os.environ.get("CRV_VISION_OCR")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.realpath(__file__))
    roots = [here, os.path.expanduser("~/Downloads/agent-tools/tools/crv"), os.getcwd()]
    swiftc = shutil.which("swiftc")
    for r in roots:
        b, src = os.path.join(r, "vision_ocr"), os.path.join(r, "vision_ocr.swift")
        stale = (os.path.exists(b) and os.path.exists(src)
                 and os.path.getmtime(src) > os.path.getmtime(b))
        if os.path.exists(b) and not (stale and swiftc):
            return b
        if swiftc and os.path.exists(src):
            subprocess.run([swiftc, "-O", src, "-o", b], capture_output=True,
                           text=True, errors="replace")
            if os.path.exists(b):
                return b
    return shutil.which("vision_ocr")


def vision_ocr(binp, paths, batch=200, workers=None, failed=None,
               timeout_base=60.0, timeout_per=1.0, tries=3):
    """OCR image files with the Vision helper -> {basename-without-ext: text}.
    Batched to stay under the argv limit (an SDH track is thousands of files),
    and the batches run in parallel -- one process at a time left the
    Neural Engine idle between images.

    Every helper call has a watchdog (timeout_base + timeout_per a frame; a
    normal batch reads ~40 frames a second). Apple Vision occasionally hangs
    inside perform() at 0% CPU: on an M5 Max (2026-11-02) one batch of RRR's caption
    frames sat for 1 h 40 min and stalled the whole film, and the same 84
    frames then read in 2-3 s three times running. So a timed-out batch is
    retried, then split in half, down to a single frame; a frame that still
    times out after `tries` attempts is left unread, reported on stderr and
    appended to `failed` (a list) so the caller can say so."""
    from concurrent.futures import ThreadPoolExecutor
    def run_once(chunk):
        r = subprocess.run([binp, *chunk], capture_output=True, text=True, errors="replace",
                           timeout=timeout_base + timeout_per * len(chunk))
        got = {}
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            idx = os.path.splitext(os.path.basename(parts[0]))[0]
            got[idx] = parts[1].strip() if len(parts) > 1 else ""
        return got
    lost = []
    def one(chunk):
        for _ in range(tries if len(chunk) == 1 else 2):
            try:
                return run_once(chunk)
            except subprocess.TimeoutExpired:
                continue
        if len(chunk) == 1:
            lost.append(chunk[0])
            return {}
        mid = len(chunk) // 2
        got = one(chunk[:mid]); got.update(one(chunk[mid:]))
        return got
    chunks = [paths[a:a + batch] for a in range(0, len(paths), batch)]
    text = {}
    with ThreadPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 2)) as ex:
        for got in ex.map(one, chunks):
            text.update(got)
    if lost:
        print(f"  ⚠ Apple Vision OCR timed out {tries}x on {len(lost)} image(s), left unread: "
              + ", ".join(os.path.basename(p) for p in lost[:5]) + (" …" if len(lost) > 5 else ""),
              file=sys.stderr, flush=True)
        if failed is not None:
            failed.extend(lost)
    return text


def pgs_render(sup, png_dir):
    """Render each PGS subtitle in `sup` to an upscaled PNG (black text on
    white, good for OCR). Returns [(key, png_path, start_s, end_s|None)] in
    display order. Shared by the engine and pgs_to_pngs.py."""
    import cv2
    from pgsrip.media_path import MediaPath
    from pgsrip.pgs import PgsReader
    from pgsrip.media import PgsSubtitleItem
    os.makedirs(png_dir, exist_ok=True)
    mp = MediaPath(sup)
    with open(sup, "rb") as f:
        items = PgsSubtitleItem.create_items(mp, list(PgsReader.decode(f.read(), mp)))
    def to_sec(t):
        h, m, rest = str(t).replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    out = []
    for it in items:
        if it.image is None or it.start is None:
            continue
        img = it.image.data                                   # binary: black text on white
        h, w = img.shape[:2]
        big = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        key = f"{it.index:05d}"
        p = os.path.join(png_dir, key + ".png")
        cv2.imwrite(p, big)
        out.append((key, p, to_sec(it.start), to_sec(it.end) if it.end else None))
    return out


def resolve_ends(entries, hold=4.0):
    """[(key, start, end|None)] -> {key: (start, end)}. A missing end holds the
    line until the next one starts, capped at `hold` seconds. A zero-length
    cue is never live at any beat, so the line would simply vanish."""
    entries = sorted(entries, key=lambda e: e[1])
    out = {}
    for k, (key, st, en) in enumerate(entries):
        if en is None or en <= st:
            nxt = entries[k + 1][1] if k + 1 < len(entries) else st + hold
            en = min(max(nxt, st), st + hold)
            if en <= st:                     # next cue starts at the same instant
                en = st + min(hold, 0.5)
        out[key] = (st, en)
    return out


def srt_ts(t):
    """Seconds -> SRT timestamp HH:MM:SS,mmm."""
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _pgs_sup_to_cues(sup, out_dir, binp):
    """Render each PGS subtitle to a PNG, OCR with Apple Vision, and join with the
    PGS timing. Returns [(start_s, end_s, text)]. The accurate path for image subs
    (stylized/italic fonts where tesseract fails)."""
    import shutil
    png_dir = os.path.join(out_dir, "_pgs_png")
    try:
        items = pgs_render(sup, png_dir)
        if not items:
            return []
        text = vision_ocr(binp, [p for _, p, _, _ in items])
        timing = resolve_ends([(k, st, en) for k, _, st, en in items])
        return sorted((timing[k][0], timing[k][1], text[k])
                      for k, _, _, _ in items if text.get(k))
    finally:
        shutil.rmtree(png_dir, ignore_errors=True)


def transcribe_pgs_vision(video, out_dir, stream="0:s:0"):
    """OCR embedded image-based (PGS) subtitles with Apple Vision — recovers the
    real, exact dialogue where tesseract mangles the font. Returns (cues, note)."""
    binp = _vision_ocr_bin()
    if not binp:
        return [], "image subs; Apple Vision OCR helper unavailable"
    try:
        import cv2  # noqa: F401 (pgsrip needs it)
        import pgsrip  # noqa: F401
    except Exception:
        return [], "image subs; pgsrip/opencv unavailable"
    sup = os.path.join(out_dir, "_subs.en.sup")               # .en tag: pgsrip needs a language
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video,
                    "-map", stream, "-c", "copy", sup], check=False)
    if not os.path.exists(sup) or os.path.getsize(sup) == 0:
        return [], "image subs; PGS extract failed"
    err = ""
    try:
        cues = _pgs_sup_to_cues(sup, out_dir, binp)
    except Exception as e:  # noqa: BLE001
        cues = []
        # a bare swallow made every OCR failure look like an empty subtitle track
        err = f" [{type(e).__name__}: {e}"[:120] + "]"
    finally:
        try: os.remove(sup)
        except OSError: pass
    if cues:
        return cues, f"Apple Vision OCR of PGS subtitles ({len(cues)} lines)"
    return [], f"image subs; Vision OCR produced nothing{err}"


TEXT_SUB_CODECS = ("subrip", "ass", "ssa", "mov_text", "webvtt", "text")
BITMAP_SUB_CODECS = ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub")


def _change_cues(changes, end, hold=6.0):
    """[(t, text)] at each moment the subtitle picture changed -> cues. A
    non-blank picture lasts until the next change (capped at `hold`, in case a
    clear was missed); consecutive identical texts -- a fade, a re-render --
    fold into one cue."""
    cues = []
    for k, (t, txt) in enumerate(changes):
        nxt = changes[k + 1][0] if k + 1 < len(changes) else end
        if not txt:
            continue
        e = min(nxt, t + hold)
        if cues and cues[-1][2] == txt and abs(cues[-1][1] - t) < 1e-3:
            cues[-1] = (cues[-1][0], e, txt)
        else:
            cues.append((t, e, txt))
    return cues


def transcribe_bitmap_vision(video, out_dir, sub, rate=20):
    """OCR any bitmap subtitle stream (VobSub, DVB, PGS, XSUB) with Apple
    Vision. ffmpeg composites the stream onto a white canvas at `rate` fps
    (negated, so DVD white-on-clear becomes black-on-white), keeps only the
    frames where the picture changes, and each kept frame is OCR'd; timing
    is exact to 1/rate s. pgsrip only reads PGS, so this is the path for
    everything else. Returns (cues, note)."""
    import shutil
    binp = _vision_ocr_bin()
    if not binp:
        return [], f"image subs ({sub['codec']}); Apple Vision OCR helper unavailable"
    dur = duration(video)
    w, h = sub.get("width") or 0, sub.get("height") or 0
    if not (w and h):
        c = video_color(video); w, h = c["width"] or 720, c["height"] or 480
    png_dir = os.path.join(out_dir, "_bitmap_png")
    shutil.rmtree(png_dir, ignore_errors=True)
    os.makedirs(png_dir)
    try:
        p = run(["ffmpeg", "-hide_banner", "-i", video, "-filter_complex",
                 f"color=c=white:s={w}x{h}:r={rate}:d={dur:.3f}[bg];"
                 f"[0:s:{sub['idx']}]negate=negate_alpha=0[s];"
                 f"[bg][s]overlay=eof_action=pass,format=gray,"
                 f"select='gt(scene,0.0001)+eq(n,0)',showinfo",
                 "-an", "-fps_mode", "vfr", os.path.join(png_dir, "%06d.png")])
        if p.returncode != 0:
            tail = p.stderr.strip().splitlines()[-1:] or [""]
            return [], f"image subs ({sub['codec']}); render failed [{tail[0][:80]}]"
        ts = [float(m) for m in re.findall(r"pts_time:(-?[0-9.]+)", p.stderr)]
        pngs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
        n = min(len(ts), len(pngs))
        text = vision_ocr(binp, pngs[:n])
        changes = [(ts[k], text.get(os.path.splitext(os.path.basename(pngs[k]))[0], ""))
                   for k in range(n)]
        cues = _change_cues(changes, dur)
    finally:
        shutil.rmtree(png_dir, ignore_errors=True)
    if cues:
        return cues, f"Apple Vision OCR of {sub['codec']} subtitles ({len(cues)} lines)"
    return [], f"image subs ({sub['codec']}); Vision OCR produced nothing"


def _sub_streams(video):
    """Subtitle streams in order as [{idx, codec, lang, forced, title}], where idx
    is the subtitle-relative index for `-map 0:s:idx`."""
    import json
    out = run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
               "stream=codec_name,width,height:stream_disposition=forced:"
               "stream_tags=language,title",
               "-of", "json", video]).stdout
    try:
        streams = json.loads(out).get("streams", [])
    except (ValueError, TypeError):
        return []
    res = []
    for i, s in enumerate(streams):
        tags = s.get("tags", {}) or {}
        disp = s.get("disposition", {}) or {}
        res.append({"idx": i, "codec": s.get("codec_name", ""),
                    "width": int(s.get("width") or 0), "height": int(s.get("height") or 0),
                    "lang": (tags.get("language", "") or "").lower(),
                    "forced": int(disp.get("forced", 0) or 0) == 1,
                    "title": (tags.get("title", "") or "").lower()})
    return res


def _forcedish(s):
    return s["forced"] or "forced" in s["title"]


def _sub_rank(s):
    """Lower is better. Prefer full (non-forced) English, then full any-language,
    then forced/foreign as last resort. Mild deprioritization of SDH within a tier."""
    eng = s["lang"] in ("eng", "en", "")
    if not _forcedish(s):
        tier = 0 if s["lang"] in ("eng", "en") else 1 if eng else 2
    else:
        tier = 3 if s["lang"] in ("eng", "en") else 4
    return (tier, 1 if "sdh" in s["title"] else 0, s["idx"])


def _pick_text_sub(streams):
    cands = [s for s in streams if s["codec"] in TEXT_SUB_CODECS]
    return min(cands, key=_sub_rank) if cands else None


def _pick_pgs_sub(streams):
    cands = [s for s in streams if s["codec"] == "hdmv_pgs_subtitle"]
    return min(cands, key=_sub_rank) if cands else None


def _pick_bitmap_sub(streams):
    cands = [s for s in streams if s["codec"] in BITMAP_SUB_CODECS]
    return min(cands, key=_sub_rank) if cands else None


_SIDECAR_SDH = ("sdh", "cc", "hi")


def _sidecars(src):
    """Sidecar subtitle files for src, best first: movie.srt / movie.vtt, then
    tagged names (movie.en.srt, movie.eng.forced.srt, movie.fr.sdh.vtt...)
    ranked with the same rules as embedded streams."""
    base = os.path.splitext(src)[0]
    exact = [base + ext for ext in (".srt", ".vtt") if os.path.exists(base + ext)]
    tagged = []
    for path in glob.glob(glob.escape(base) + ".*"):
        stem, ext = os.path.splitext(path)
        if ext.lower() not in (".srt", ".vtt") or path in exact:
            continue
        tags = [t for t in stem[len(base) + 1:].lower().split(".") if t]
        lang = next((t for t in tags if t.isalpha() and 2 <= len(t) <= 3
                     and t not in _SIDECAR_SDH), "")
        if "english" in tags: lang = "eng"
        tagged.append(({"idx": len(tagged), "lang": lang, "forced": "forced" in tags,
                        "title": "sdh" if set(tags) & set(_SIDECAR_SDH) else ""}, path))
    tagged.sort(key=lambda sp: (_sub_rank(sp[0]), os.path.basename(sp[1])))
    return exact + [p for _, p in tagged]


def get_cues(src, video, out_dir):
    """Return (cues, note). cues = [(start,end,text)] from a sidecar .srt/.vtt or
    an embedded text subtitle stream. Image subs (PGS/VobSub) can't be read as
    text -> ([], reason)."""
    # 1) sidecar next to the original source
    if not src.startswith(("http://", "https://")):
        for path in _sidecars(src):
            c = parse_srt(path)
            if c: return c, f"sidecar {os.path.basename(path)}"
    # 2) embedded stream — pick the best TEXT stream: skip forced/signs-only
    #    tracks, prefer a full English track (else any full non-forced track).
    streams = _sub_streams(video)
    best = _pick_text_sub(streams)
    if best is not None:
        raw = os.path.join(out_dir, "_embedded.srt")
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video,
             "-map", f"0:s:{best['idx']}", raw])
        if os.path.exists(raw):
            c = parse_srt(raw)
            try: os.remove(raw)
            except OSError: pass
            if c:
                extra = []
                if best["lang"] and best["lang"] not in ("eng", "en"): extra.append(best["lang"])
                if "sdh" in best["title"]: extra.append("SDH")
                tag = f" [{', '.join(extra)}]" if extra else ""
                return c, f"embedded {best['codec']} subtitle stream{tag}"
    if streams:
        # name the codec that actually blocked us, not stream 0 (which may well
        # be a text track we rejected as forced/signs-only).
        img = next((s for s in streams if s["codec"] not in TEXT_SUB_CODECS), streams[0])
        return [], f"subtitles present but image-based ({img['codec']}) — not text-readable"
    return [], "no subtitle track"


_LA_CACHE = [None, None, None, None]     # [cues_ref, starts, ordered, running max end]


def _cue_starts(cues):
    """Cached sorted index for `cues`, so repeated line_at() calls over the same
    list don't re-sort it. Callers (redialogue.py included) may pass any cue list
    in any order, so sorting has to happen here rather than be assumed.

    The cache holds a reference to the list itself, not just its id -- keeping it
    alive is what makes the identity check safe."""
    if _LA_CACHE[0] is not cues or len(_LA_CACHE[2]) != len(cues):
        ordered = sorted(cues, key=lambda c: (c[0], c[1]))
        maxend, m = [], float("-inf")
        for c in ordered:
            m = max(m, c[1]); maxend.append(m)
        _LA_CACHE[:] = [cues, [c[0] for c in ordered], ordered, maxend]
    return _LA_CACHE[1], _LA_CACHE[2], _LA_CACHE[3]


def line_at(t, cues, pad=0.3):
    """Dialogue live at time t: the cue that actually spans t (the latest-starting
    one, when they overlap), else the nearest cue within `pad`, else ''.

    First-match-with-pad returned the *previous* line at every cue boundary --
    the padded tail of the outgoing cue beat the incoming one that really covers
    the moment, so a frame on a cut got the line before it."""
    if not cues:
        return ""
    import bisect
    starts, ordered, maxend = _cue_starts(cues)
    i = bisect.bisect_right(starts, t)
    # spanning cue: scan back from the last cue that starts at/before t. Once
    # no cue at or before k ends at/after t, none can span it -- exact, and in
    # a gap between lines it stops at once instead of walking back an hour.
    for k in range(i - 1, -1, -1):
        if maxend[k] < t:
            break
        s, e, txt = ordered[k]
        if s <= t <= e:
            return txt
    # else nearest within pad, on either side
    best, bd = "", pad + 1e-9
    for k in (i - 1, i):
        if 0 <= k < len(ordered):
            s, e, txt = ordered[k]
            d = (s - t) if t < s else (t - e if t > e else 0.0)
            if d <= bd + 1e-6:     # a tie in the gap goes to the incoming line
                best, bd = txt, d
    return best


# ---------------------------------------------------------------- mechanical audio layer
def _audio_features(video, sr=16000, hop=512, win=1024, pcm_path=None):
    """Heavy, beat-independent half of analyze_audio: boil the soundtrack down
    to per-frame series. pcm_path: s16le mono PCM at `sr` written by the frame
    extraction's decode (the normal path); without it the track is decoded
    here. Returns a dict, or None."""
    import numpy as np
    if pcm_path:
        x = np.fromfile(pcm_path, dtype=np.int16).astype(np.float32)
    else:
        # decode the soundtrack to raw mono PCM on stdout (binary; global opts first)
        p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-i", video, "-vn", "-ac", "1", "-ar", str(sr),
                            "-f", "s16le", "-"], capture_output=True)
        raw = p.stdout
        del p  # don't hold the raw bytes alongside the float32 copy
        if not raw:
            return None
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        del raw
    if not x.size:
        return None
    x /= 32768.0  # in place: same elementwise op, one array instead of two
    if x.size < win * 2:
        return None

    n_frames = 1 + (len(x) - win) // hop
    # a fancy-index matrix here is n_frames x win int64 — 1.8 GB for a 2h film,
    # which defeats the chunk loop below. The strided view costs nothing.
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    if len(frames) != n_frames:
        return None
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)

    rms = np.empty(n_frames, np.float32)
    centroid = np.empty(n_frames, np.float32)
    flatness = np.empty(n_frames, np.float32)
    CH = 8000  # frames per chunk, bounds memory
    for a in range(0, n_frames, CH):
        b = min(a + CH, n_frames)
        fr = frames[a:b]
        rms[a:b] = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
        mag = np.abs(np.fft.rfft(fr * window, axis=1)) + 1e-10
        centroid[a:b] = (mag * freqs).sum(1) / mag.sum(1)
        flatness[a:b] = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    ftimes = (np.arange(n_frames) * hop + win / 2) / sr
    db = 20 * np.log10(rms + 1e-6)

    # onset envelope for tempo (positive spectral-flux proxy from rms)
    onset = np.maximum(0, np.diff(db, prepend=db[0]))
    return {"sr": sr, "hop": hop, "n": n_frames, "ftimes": ftimes, "db": db,
            "centroid": centroid, "flatness": flatness, "onset": onset}


def _lufs_series(video):
    """BS.1770 momentary loudness (400 ms window, every 100 ms) from ffmpeg's
    ebur128, on the source's own channel layout. Returns (times, M) with
    silence clamped to -70 LUFS, or None. ~8 s for a 2 h film: audio-only."""
    import numpy as np, tempfile
    fd, path = tempfile.mkstemp(suffix=".lufs")
    os.close(fd)
    try:
        r = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video, "-vn", "-af",
                 f"ebur128=metadata=1,ametadata=mode=print:key=lavfi.r128.M:file={path}",
                 "-f", "null", "-"])
        if r.returncode != 0:
            return None
        txt = open(path, encoding="utf-8", errors="replace").read()
    finally:
        try: os.remove(path)
        except OSError: pass
    ts = re.findall(r"pts_time:(-?[0-9.]+)", txt)
    ms = re.findall(r"lavfi\.r128\.M=(-?[0-9.]+|-inf|nan)", txt)
    n = min(len(ts), len(ms))
    if n < 2:
        return None
    M = np.array([float(m) if m not in ("-inf", "nan") else -120.0 for m in ms[:n]], np.float32)
    return np.array([float(t) for t in ts[:n]], np.float32), np.maximum(M, -70.0)


SPEECH_BAND = (300.0, 3400.0)


def _spatial_features(pcm_path, names, sr=SPATIAL_SR, hop=400, win=800):
    """Per-channel series from the multichannel decode (s16le, channel order
    `names`), 50 ms frames: broadband level and speech-band level per channel,
    in dB. Read through a memmap in chunks, so a 6-channel 2 h film (690 MB on
    disk) never becomes a float copy. Returns a dict, or None."""
    import numpy as np
    from scipy.signal import butter, sosfilt, sosfilt_zi
    nch = len(names)
    try:
        raw = np.memmap(pcm_path, dtype=np.int16, mode="r")
    except (OSError, ValueError):
        return None
    S = raw.size // nch
    if S < win * 2:
        return None
    raw = raw[:S * nch].reshape(S, nch)
    n = 1 + (S - win) // hop
    db = np.empty((nch, n), np.float32)
    sp = np.empty((nch, n), np.float32)
    sos = butter(4, SPEECH_BAND, btype="band", fs=sr, output="sos")
    zi = [sosfilt_zi(sos) * 0.0 for _ in range(nch)]
    CH = 4000                                  # frames per chunk
    for a in range(0, n, CH):
        b = min(a + CH, n)
        s0, s1 = a * hop, (b - 1) * hop + win
        x = raw[s0:s1].astype(np.float32) / 32768.0          # samples x ch
        for c in range(nch):
            v = np.lib.stride_tricks.sliding_window_view(x[:, c], win)[::hop]
            db[c, a:b] = 20 * np.log10(np.sqrt((v * v).mean(1)) + 1e-6)
            # the band filter runs on the chunk's *new* samples only, carrying
            # state, so chunk seams do not double-filter the window overlap
            seg = x[:(b - a) * hop, c] if b < n else x[:, c]
            y, zi[c] = sosfilt(sos, seg, zi=zi[c])
            if b < n:
                # frames near the chunk end need samples past the new ones
                tail = x[(b - a) * hop:, c]
                yt, _ = sosfilt(sos, tail, zi=zi[c])
                y = np.concatenate([y, yt])
            v = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
            sp[c, a:b] = 20 * np.log10(np.sqrt((v * v).mean(1)) + 1e-6)
    ftimes = (np.arange(n) * hop + win / 2) / sr
    return {"names": names, "ftimes": ftimes, "db": db, "sp": sp, "hop": hop, "sr": sr}


def _audio_all(video, pcm_path, spatial_path, names):
    """The audio layer's background work: mono features, BS.1770 loudness and,
    when the source has a known multichannel layout, the spatial series."""
    feat = _audio_features(video, AUDIO_SR, 512, 1024, pcm_path)
    lufs = _lufs_series(video) if feat else None
    spat = None
    if feat and spatial_path and names and len(names) >= 2:
        spat = _spatial_features(spatial_path, names)
    return {"feat": feat, "lufs": lufs, "spat": spat}


# loudness classes on momentary LUFS. Alien (1979) 5.1: percentiles 5/25/50/
# 75/90/95 = -52/-40/-33/-28/-23/-21; theatrical dialogue sits near -30.
LUFS_SILENT, LUFS_QUIET, LUFS_LOUD = -50.0, -36.0, -24.0

_FRONT = ("FL", "FR", "FC", "FLC", "FRC", "L", "R")
_REAR = ("SL", "SR", "BL", "BR")


def _placement(names, levels):
    """Where a sound sits, from per-channel mean levels (dB, LFE excluded):
    centre / front-L / front-R / rear-L / rear-R / front-wide / all-around."""
    import numpy as np
    order = np.argsort(levels)[::-1]
    top, second = names[order[0]], names[order[1]] if len(order) > 1 else None
    gap = levels[order[0]] - (levels[order[1]] if len(order) > 1 else -999)
    side = {"FL": "front-L", "L": "front-L", "FLC": "front-L", "FR": "front-R", "R": "front-R",
            "FRC": "front-R", "FC": "centre", "SL": "rear-L", "BL": "rear-L",
            "SR": "rear-R", "BR": "rear-R"}
    if gap >= 3.0 or second is None:
        return side.get(top, "")
    if top in _FRONT and second in _FRONT:
        return "front-wide"
    return "all-around"


def _spatial_beats(spat, lufs, beats, sounds, cues):
    """Per-beat impact / field / voice / pan from the spatial series, and a
    placement for every sound event. Returns {ts: {...}}; sounds are updated
    in place. Thresholds are relative to the film's own distribution (LFE
    median/MAD, speech-band percentile) so a quiet mix and a loud one read
    the same way.

    voice *places* speech, it does not detect it: a beat with a dialogue cue
    is always placed (the cues are the better detector -- on Alien an energy
    threshold alone missed 59% of subtitled beats and fired on a third of the
    silent ones); a beat without one is placed only when its speech band is
    in the film's top fifth, for untranscribed voices."""
    import numpy as np
    names = spat["names"]; ft = spat["ftimes"]; db = spat["db"]; sp = spat["sp"]
    ix = {nm: i for i, nm in enumerate(names)}
    non_lfe = [i for i, nm in enumerate(names) if nm != "LFE"]
    fronts = [ix[nm] for nm in names if nm in _FRONT]
    rears = [ix[nm] for nm in names if nm in _REAR]
    lfe = ix.get("LFE"); fc = ix.get("FC")
    L = ix.get("FL", ix.get("L")); R = ix.get("FR", ix.get("R"))
    # speech present when the loudest channel's speech band is in the film's
    # upper third (dialogue covers ~a third of a feature) and clear of the
    # noise floor; the floor clause keeps a near-silent clip from reading as
    # wall-to-wall voice
    spmax = sp[non_lfe].max(0)
    sp_thr = max(float(np.percentile(spmax, 80)), -50.0)
    if lfe is not None:
        lmed = float(np.median(db[lfe])); lmad = float(np.median(np.abs(db[lfe] - lmed))) + 1e-6
    out = {}
    def win(t, pad=0.6):
        i0 = int(np.searchsorted(ft, t - pad)); i1 = max(int(np.searchsorted(ft, t + pad)), i0 + 1)
        return i0, min(i1, len(ft))
    for _, t in beats:
        i0, i1 = win(t)
        if i0 >= len(ft):
            out[t] = {}; continue
        d = {}
        quiet = float(db[non_lfe, i0:i1].max()) < -60
        # impact: the LFE channel is the "feel it" track; a transient on top of
        # the window's own level is a hit, a sustained excess is a rumble
        if lfe is not None and not quiet:
            seg = db[lfe, i0:i1]; mx = float(seg.max())
            if mx > lmed + 2 * lmad and mx - float(seg.mean()) > 10:
                d["impact"] = "hit"
            elif mx > lmed + 4 * lmad:
                d["impact"] = "hard"
            elif mx > lmed + 2 * lmad:
                d["impact"] = "low"
        # field: front stage vs envelopment
        if rears and fronts and not quiet:
            fr = float(np.median(db[fronts, i0:i1].max(0) - db[rears, i0:i1].max(0)))
            d["field"] = "frontal" if fr > 15 else ("wide" if fr > 6 else "enveloping")
        # voice: where the speech-band energy sits, when someone speaks. With
        # a cue track the cue is the only gate: the energy rule alone placed
        # voice off-L/off-R on 158 cue-less beats of Alien (strings in the
        # final log and the end credits read as speech). Without any cues
        # (no subs, no transcript) the energy rule is all there is.
        m = sp[:, i0:i1].mean(1)
        mx = float(m[non_lfe].max())
        if (line_at(t, cues) and mx > -60) if cues else (mx >= sp_thr):
            dom = names[non_lfe[int(np.argmax(m[non_lfe]))]]
            if fc is not None:
                if m[fc] >= mx - 6: d["voice"] = "centre"
                elif dom in ("FL", "FLC"): d["voice"] = "off-L"
                elif dom in ("FR", "FRC"): d["voice"] = "off-R"
                elif dom in _REAR: d["voice"] = "rear"
            elif L is not None and R is not None:
                bal = float(m[L] - m[R])
                d["voice"] = "centre" if abs(bal) < 3 else ("off-L" if bal > 0 else "off-R")
        # pan: energy moving across the field inside the window
        # pan: a real level (both ends above -45 dB) moving by >8 dB across
        # the window; 6 dB without the floor fired on a quarter of all beats
        if L is not None and R is not None and not quiet and i1 - i0 >= 6:
            k = (i1 - i0) // 3
            lv0 = float(db[non_lfe, i0:i0 + k].max()); lv1 = float(db[non_lfe, i1 - k:i1].max())
            if lv0 > -45 and lv1 > -45:
                bal0 = float((db[L, i0:i0 + k] - db[R, i0:i0 + k]).mean())
                bal1 = float((db[L, i1 - k:i1] - db[R, i1 - k:i1]).mean())
                if bal0 - bal1 > 8: d["pan"] = "L→R"
                elif bal1 - bal0 > 8: d["pan"] = "R→L"
                elif rears and fronts:
                    fr0 = float((db[fronts, i0:i0 + k].max(0) - db[rears, i0:i0 + k].max(0)).mean())
                    fr1 = float((db[fronts, i1 - k:i1].max(0) - db[rears, i1 - k:i1].max(0)).mean())
                    if fr0 - fr1 > 8: d["pan"] = "front→rear"
                    elif fr1 - fr0 > 8: d["pan"] = "rear→front"
        out[t] = d
    # LFE rumbles: sustained excess on the LFE track, as events of their own.
    # The mono detector's 0.3-8 s window cannot hold a 4-minute landing or a
    # 54 s self-destruct; per-beat `impact` tags each beat, this names the span.
    if lfe is not None:
        hot = db[lfe] > lmed + 4 * lmad
        idx = np.flatnonzero(hot)
        gap = int(1.0 / (spat["hop"] / spat["sr"]))          # merge pauses < 1 s
        k = 1
        if idx.size:
            brk = np.flatnonzero(np.diff(idx) > gap)
            for a, b in zip(np.r_[idx[0], idx[brk + 1]], np.r_[idx[brk], idx[-1]]):
                dur = float(ft[b] - ft[a]) + spat["hop"] / spat["sr"]
                if dur < 2.0:
                    continue                                   # a hit, not a rumble
                sounds.append({"label": f"Rumble-{k}", "t": float(ft[a]), "dur": round(dur, 1),
                               "pitch": "low", "texture": "rumble", "rumble": True})
                k += 1
    for snd in sounds:
        i0, i1 = win(snd["t"] + snd["dur"] / 2, max(snd["dur"] / 2, 0.3))
        if i0 < len(ft):
            lv = db[non_lfe, i0:i1].mean(1)
            snd["placement"] = _placement([names[i] for i in non_lfe], lv)
    sounds.sort(key=lambda s: s["t"])
    return out


def _audio_beats(feat, beats, cues, lufs=None, spat=None):
    """Cheap, beat-dependent half: look each beat up in the series, then pick the
    salient non-speech events out of them — that part needs the speech mask, so
    it has to wait for `cues`. lufs: (times, momentary LUFS) for loudness and
    trend, else the mono RMS dB is used; spat: _spatial_features output for the
    impact/field/voice/pan columns and sound placement. Returns (per_beat:
    {ts:info}, sounds)."""
    if isinstance(feat, dict) and "feat" in feat:           # _audio_all bundle
        feat, lufs, spat = feat["feat"], feat.get("lufs"), feat.get("spat")
    if not feat:
        return {}, []
    import numpy as np
    sr = feat["sr"]; hop = feat["hop"]; n_frames = feat["n"]
    ftimes = feat["ftimes"]; db = feat["db"]; onset = feat["onset"]
    centroid = feat["centroid"]; flatness = feat["flatness"]

    def frames_between(t0, t1):
        i0 = np.searchsorted(ftimes, t0); i1 = np.searchsorted(ftimes, t1)
        return i0, max(i1, i0 + 1)

    def local_tempo(t):
        i0, i1 = frames_between(t - 4, t + 4)
        seg = onset[i0:i1]
        if seg.size < 16 or seg.std() < 1e-4: return None
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[seg.size - 1:]
        lo = int(0.30 / (hop / sr)); hi = int(1.20 / (hop / sr))  # 50-200 BPM
        if hi >= ac.size: hi = ac.size - 1
        if hi <= lo: return None
        lag = lo + int(np.argmax(ac[lo:hi]))
        if lag <= 0: return None
        return round(60.0 / (lag * hop / sr))

    def bright(c):
        return "dark" if c < 1200 else ("bright" if c > 3200 else "mid")

    per_beat, prev_db = {}, None
    lt, lm = (lufs if lufs is not None else (None, None))
    for name, t in beats:
        i0, i1 = frames_between(t - 0.6, t + 0.6)
        seg_db = float(db[i0:i1].mean()); seg_c = float(centroid[i0:i1].mean())
        if lt is not None:
            j0 = int(np.searchsorted(lt, t - 0.6)); j1 = max(int(np.searchsorted(lt, t + 0.6)), j0 + 1)
            seg_l = float(lm[j0:j1].mean()) if j0 < len(lt) else -70.0
            loud = ("silent" if seg_l < LUFS_SILENT else "quiet" if seg_l < LUFS_QUIET
                    else "loud" if seg_l > LUFS_LOUD else "mid")
            level = seg_l
        else:
            loud = "silent" if seg_db < -55 else ("quiet" if seg_db < -35 else ("loud" if seg_db > -18 else "mid"))
            level = seg_db
        if prev_db is None: trend = "—"
        elif level - prev_db > 3: trend = "building"
        elif prev_db - level > 3: trend = "releasing"
        else: trend = "steady"
        prev_db = level
        per_beat[t] = {"loud": loud, "trend": trend,
                       "tempo": local_tempo(t), "bright": bright(seg_c)}

    # -------- distinct non-speech sound events ------------------------------
    speech = np.zeros(n_frames, bool)
    for s, e, _ in cues:
        i0, i1 = frames_between(s, e); speech[i0:i1] = True
    thr = np.median(db) + 1.4826 * np.median(np.abs(db - np.median(db))) * 1.5
    salient = (db > thr) & (~speech)
    # contiguous runs, merge gaps < 0.25s, keep 0.3s..8s
    # Two salient frames belong to one event when at most `gap` frames apart.
    # The old scan only looked exactly `gap` ahead, so a short burst after a
    # short pause ended the event mid-gap and both halves fell under 0.3s.
    events = []
    gap = int(0.25 / (hop / sr))
    idx = np.flatnonzero(salient)
    if idx.size:
        brk = np.flatnonzero(np.diff(idx) > gap)
        for a, b in zip(np.r_[idx[0], idx[brk + 1]], np.r_[idx[brk], idx[-1]]):
            i, j = int(a), int(b) + 1                    # [i, j) like before
            dur = ftimes[min(j, n_frames-1)] - ftimes[i]
            if 0.3 <= dur <= 8.0:
                events.append((i, j))

    sounds = []
    if events:
        feats = []
        for i0, i1 in events:
            feats.append([centroid[i0:i1].mean(), flatness[i0:i1].mean(),
                          db[i0:i1].max(), ftimes[i1 if i1 < n_frames else -1] - ftimes[i0]])
        feats = np.array(feats, np.float32)
        fn = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
        # cluster distinct timbres
        labels = np.zeros(len(events), int)
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist
            if len(events) > 1:
                # condensed distances explicitly: with exactly two events the
                # 2x2 observation matrix is mistaken for a distance matrix
                labels = fcluster(linkage(pdist(fn[:, :2]), "ward"), t=1.4, criterion="distance")
            else:
                labels = np.array([1])
        except Exception:
            labels = np.arange(1, len(events) + 1)
        # order labels by first appearance -> Sound-A, Sound-B, ... Sound-AA.
        # chr(ord("A")+n) ran off the end of the alphabet into punctuation on any
        # film with more than 26 distinct timbres.
        def _alpha(n):
            s = ""
            while True:
                n, r = divmod(n, 26)
                s = chr(ord("A") + r) + s
                if n == 0: return s
                n -= 1
        remap = {}
        for lab in labels:
            if lab not in remap:
                remap[lab] = _alpha(len(remap))
        for (i0, i1), lab, f in zip(events, labels, feats):
            c, flat, mx, dur = f
            sounds.append({
                "label": f"Sound-{remap[lab]}", "t": float(ftimes[i0]),
                "dur": round(float(dur), 1),
                "pitch": "low" if c < 1200 else ("high" if c > 3500 else "mid"),
                "texture": "tonal" if flat < 0.08 else ("noisy" if flat > 0.35 else "rough"),
            })
        sounds.sort(key=lambda s: s["t"])
    if spat:
        for t, d in _spatial_beats(spat, lufs, beats, sounds, cues).items():
            per_beat.setdefault(t, {}).update(d)
    return per_beat, sounds


def analyze_audio(video, beats, cues, sr=16000, hop=512, win=1024):
    """Per-beat loudness / trend / tempo / brightness + timbre-clustered labels
    for salient non-speech sound events. Returns (per_beat: {ts:info}, sounds).
    Degrades to ({}, []) on any failure. Both halves back to back; main() runs
    the heavy one in the background instead."""
    return _audio_beats(_audio_features(video, sr, hop, win), beats, cues)


def lines_since(prev_t, t, cues, pad=0.3):
    """Every cue that belongs to the stretch (prev_t, t]: cues starting in
    it, plus the one live at t (line_at's rule). Joined with ' / ' in start
    order. INDEX used to carry only the line live *at* the frame, so a short
    cue that started and ended between two kept frames (a second or more
    apart) appeared nowhere; now reading the rows top to bottom gives every
    line, each once. prev_t None = from the start of the film."""
    if not cues:
        return ""
    import bisect
    starts, ordered, _ = _cue_starts(cues)
    lo = 0 if prev_t is None else bisect.bisect_right(starts, prev_t)
    hi = bisect.bisect_right(starts, t)
    picked = [c for c in ordered[lo:hi]]
    live = line_at(t, cues, pad)
    if live and not any(c[2] == live for c in picked):
        # the live line started before prev_t (or within pad after t)
        for c in ordered:
            if c[2] == live and c[0] <= t + pad and c[1] >= t - pad:
                picked.append(c); break
    picked.sort(key=lambda c: c[0])
    return " / ".join(c[2].strip() for c in picked if c[2].strip())


def write_dialogue_tsv(path, cues, beats):
    """Every cue with the kept frame it belongs to: the last frame at or before
    the cue's start (or the first frame, for a cue before any). A reader can
    go from a line to the picture it was spoken over without the index's
    sampling in the way. Columns: n · start · end · frame · text."""
    import bisect
    ts = [t for _, t in beats]
    with open(path, "w", encoding="utf-8") as f:
        f.write("n\tstart\tend\tframe\ttext\n")
        for i, (s, e, t) in enumerate(sorted(cues, key=lambda c: c[0]), 1):
            j = max(0, bisect.bisect_right(ts, s) - 1) if ts else 0
            frame = beats[j][0] if beats else ""
            f.write(f"{i}\t{hms(s)}\t{hms(e)}\t{frame}\t{tsv_safe(t)}\n")


def cue_coverage(rows, cues):
    """(carried, total): how many cues' text appears in the INDEX dialogue
    column. Must be total/total; the MANIFEST says so and the e2e test
    checks it, because the old sampling silently dropped two thirds."""
    # both sides through tsv_safe: rows arrive with the raw joined text (the
    # newline inside a two-line cue included) and are only cleaned at write
    # time, which read 57 multi-line cues as missing on a film that had none
    texts = " / ".join(tsv_safe(r.get("dialogue", "")) for r in rows)
    carried = sum(1 for _, _, t in cues if t.strip() and tsv_safe(t) in texts)
    return carried, sum(1 for _, _, t in cues if t.strip())


def write_srt(path, cues):
    """The dialogue verbatim, with its own timecodes, so the index's sampling
    never stands between a reader and the lines."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(sorted(cues, key=lambda c: c[0]), 1):
            f.write(f"{i}\n{srt_ts(s)} --> {srt_ts(e)}\n{t.strip()}\n\n")


_SA_CACHE = [None, None, None, None, 0.0]   # [sounds_ref, n, ordered, starts, max_dur]


def sound_at(t, sounds, pad=0.4):
    """Label of the shortest sound live at t (±pad), else '' -- shortest, so a
    4-minute LFE rumble does not mask the events inside it (was: earliest-
    starting, which is the same thing when nothing overlaps). A cached
    start-sorted index plus bisect bounds the scan, instead of walking every
    sound for every beat (same caching contract as line_at)."""
    import bisect
    if not sounds:
        return ""
    if _SA_CACHE[0] is not sounds or _SA_CACHE[1] != len(sounds):
        ordered = sorted(sounds, key=lambda s: s["t"])
        _SA_CACHE[:] = [sounds, len(sounds), ordered, [s["t"] for s in ordered],
                        max(s["dur"] for s in ordered)]
    ordered, starts, max_dur = _SA_CACHE[2], _SA_CACHE[3], _SA_CACHE[4]
    lo = bisect.bisect_left(starts, t - pad - max_dur)
    hi = bisect.bisect_right(starts, t + pad)
    live = [s for s in ordered[lo:hi] if s["t"] - pad <= t <= s["t"] + s["dur"] + pad]
    if not live:
        return ""
    s = min(live, key=lambda s: s["dur"])
    return f"{s['label']}@{s['placement']}" if s.get("placement") else s["label"]


# ---------------------------------------------------------------- motion (camera grammar)
# Mean left/right-half divergence (px per 1/3 s at 128px wide, sub-pixel) that
# reads as a zoom. Calibrated on synthetic clips: still, grain, handheld jitter,
# pans and tilts all average within +-0.005 (per-sample sd <= 0.017); a
# 0.74%/s push-in measures 0.068, 2.4%/s measures 0.24.
ZOOM_DIV = 0.05
H_OVER_W = MOTION_H / MOTION_W     # the vertical half-shift of an isotropic zoom
# Mean |gray diff| (0-255, 128x72, 3 fps) below which a jump is never a cut,
# and how far above both neighbours a sample must sit to be a cut when the
# neighbours are themselves large (handheld).
CUT_FLOOR = 20.0
CUT_EXCESS = 15.0


CUT_LUM_REF, CUT_LUM_FLOOR = 64.0, 8.0


def lum_scale(lum):
    """Per-sample gain that puts a dark frame pair on the same footing as a
    mid-grey one: CUT_LUM_REF / mean luminance, 1 at or above the reference,
    capped at CUT_LUM_REF / CUT_LUM_FLOOR (8x). A hard cut between two
    near-black set-ups moves the 128x72 mean by 3-8 where a lit one moves it
    by 20-250, so a fixed floor read Alien's opening tour (423 s), the egg
    chamber (212 s) and the shuttle (186 s) as single shots."""
    import numpy as np
    lum = np.asarray(lum, dtype=np.float64)
    return CUT_LUM_REF / np.clip(lum, CUT_LUM_FLOOR, CUT_LUM_REF)


def cut_mask(fdiff, lum=None):
    """Which samples of the per-sample mean |gray diff| are hard cuts: above
    the film's own baseline (median + 3 MAD) and above CUT_FLOOR, and a
    one-sample spike (more than twice both neighbours, or CUT_EXCESS above
    them), so a fast pan's run of high samples is not a string of cuts.
    `lum`: the mean luminance of each sample's frame pair; when given, the
    diff is scaled by lum_scale so dark footage is judged at the same
    threshold as lit footage. A sample passes on either the raw or the
    scaled series (scaling never removes a cut)."""
    import numpy as np
    fdiff = np.asarray(fdiff, dtype=np.float64)
    if fdiff.size < 2:
        return np.zeros(fdiff.size, bool)
    if lum is not None:
        return cut_mask(fdiff) | cut_mask(fdiff * lum_scale(lum))
    # median + 3 MAD, floored: with 8 MAD a busy handheld film (Disclosure Day
    # 2026: median 9.3, MAD 7.0) set the bar at 65 and passed 22 cuts in 25
    # minutes; at 30 it passes 140 (6/min). The spike/excess test below is
    # what separates a cut from motion; this is only a floor.
    med = np.median(fdiff); mad = np.median(np.abs(fdiff - med)) + 1e-6
    thr = max(med + 3 * mad, CUT_FLOOR)
    # neighbours padded with 0 at the ends: the first/last sample is its own
    # neighbour otherwise and can never be a spike. A spike is twice both
    # neighbours OR CUT_EXCESS above them: on handheld footage every sample is
    # already 10-15, so a 40-point cut is not "twice" its neighbours and the
    # ratio alone found 101 shots in 2h25m of Disclosure Day (2026).
    nb = np.maximum(np.r_[0.0, fdiff[:-1]], np.r_[fdiff[1:], 0.0])
    return (fdiff > thr) & ((fdiff > 2 * nb) | (fdiff - nb > CUT_EXCESS))

def _motion_series(video, fps_s=MOTION_FPS, W=MOTION_W, H=MOTION_H, gray_path=None):
    """Heavy, beat-independent half of analyze_motion. Read a steady low-res
    gray stream and get global optical flow via phase correlation, so a slow
    push-in, a static hold and a whip-pan stop looking like identical frames.
    gray_path: that stream as a raw file, written by the frame extraction's
    decode (the normal path); without it the film is decoded here. Returns
    the per-frame series (plus the cut times), or None."""
    import numpy as np
    half = W // 2
    _win = {}
    def win(shape):
        if shape not in _win:
            h, w = shape
            _win[shape] = (np.hanning(h)[:, None] * np.hanning(w)[None, :]).astype(np.float32)
        return _win[shape]

    vhalf = H // 2
    def ffts(f):
        """The five windowed FFTs a frame takes part in (whole, left/right
        halves, top/bottom halves). Cached per frame: every frame is otherwise
        transformed twice, once as b and again as a."""
        return (np.fft.rfft2(f * win((H, W))),
                np.fft.rfft2(f[:, :half] * win((H, half))),
                np.fft.rfft2(f[:, half:] * win((H, W - half))),
                np.fft.rfft2(f[:vhalf] * win((vhalf, W))),
                np.fft.rfft2(f[vhalf:] * win((H - vhalf, W))))

    def peak(A, B, shape):
        """Signed global (dx,dy) of B's frame relative to A's via phase correlation."""
        h, wd = shape
        R = A * np.conj(B); R /= np.abs(R) + 1e-8   # fresh array — caches untouched
        c = np.fft.irfft2(R, s=shape)
        iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
        # Sub-pixel: fit a parabola through the peak and its wrapped neighbours.
        # Whole-pixel peaks at 128px wide put a slow push-in (~0.2px/sample of
        # divergence) permanently below the quantization step.
        def frac(m, z, p):
            d = m - 2 * z + p
            return 0.0 if d >= 0 else max(-0.5, min(0.5, 0.5 * (m - p) / d))
        dy = iy + frac(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
        dx = ix + frac(c[iy, (ix - 1) % wd], c[iy, ix], c[iy, (ix + 1) % wd])
        if dy > h / 2: dy -= h
        if dx > wd / 2: dx -= wd
        return float(dx), float(dy)

    def decode():
        """Stream the gray frames and score each against its predecessor, so only
        one frame — not the whole film — is ever in memory. -> (rc, series).

        Software decode on purpose. This used to try VideoToolbox first, and on
        a 1080p h264 feature that was 745s against 149s in software (M1 Max,
        2026-11-01): every decoded frame is copied back from the GPU at full
        size only for fps=3 to drop seven of eight. The numpy loop that follows
        is ~7s for the whole film, so the decode is the entire cost of this pass
        and it was the whole wall time of a run (motion 448s of 448s on an M5 Max)."""
        if gray_path:
            p, src = None, open(gray_path, "rb")
        else:
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            cmd += ["-i", video, "-vf", f"fps={fps_s},scale={W}:{H},format=gray",
                    "-f", "rawvideo", "-"]
            # stderr to DEVNULL: nothing drains it while we read stdout, and a
            # full stderr pipe would wedge ffmpeg partway through the film.
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            src = p.stdout
        nb = W * H
        dxs, dys, divx, divy, fdiff, lum = [], [], [], [], [], []
        prev = prev_f = None
        n = 0
        try:
            while True:
                chunk = src.read(nb)
                if len(chunk) < nb:          # EOF, or a torn trailing frame
                    break
                cur = np.frombuffer(chunk, dtype=np.uint8).reshape(H, W).astype(np.float32)
                cur_f = ffts(cur)
                if prev is not None:
                    dx, dy = peak(prev_f[0], cur_f[0], (H, W))
                    # left/right half shift -> horizontal divergence (zoom/dolly)
                    ldx = peak(prev_f[1], cur_f[1], (H, half))[0]
                    rdx = peak(prev_f[2], cur_f[2], (H, W - half))[0]
                    tdy = peak(prev_f[3], cur_f[3], (vhalf, W))[1]
                    bdy = peak(prev_f[4], cur_f[4], (H - vhalf, W))[1]
                    # positive = the halves move apart = the image grows =
                    # push-in. (rdx - ldx had it backwards: peak() reports the
                    # camera's motion, which is opposite to the content's.)
                    # Vertical divergence too: a real zoom spreads both axes,
                    # a person walking through a static frame only one.
                    dxs.append(dx); dys.append(dy); divx.append(ldx - rdx); divy.append(tdy - bdy)
                    fdiff.append(np.mean(np.abs(cur - prev)))
                    lum.append(0.5 * (float(cur.mean()) + float(prev.mean())))
                prev, prev_f = cur, cur_f
                n += 1
        finally:
            src.close()
            rc = p.wait() if p else 0
        return rc, (n, dxs, dys, divx, divy, fdiff, lum)

    rc, ser = decode()
    n, _dxs, _dys, _divx, _divy, _fdiff, _lum = ser
    # a non-zero exit means a partial stream, and a truncated read would be
    # scored as real motion -- so it is no result, not a short one.
    if rc != 0 or n < 4:
        return None
    times = np.arange(n) / fps_s
    dxs = np.zeros(n); dys = np.zeros(n); divx = np.zeros(n); divy = np.zeros(n); fdiff = np.zeros(n)
    dxs[1:] = _dxs; dys[1:] = _dys; divx[1:] = _divx; divy[1:] = _divy; fdiff[1:] = _fdiff
    lum = np.full(n, CUT_LUM_REF); lum[1:] = _lum

    # cuts: a single-sample jump in the gray stream. Relative to the film's
    # baseline, but never below an absolute floor: on a film of locked-off
    # static shots the MAD is tiny (The Zone of Interest: median 0.83, MAD
    # 0.63 of 255, so med+8*MAD was 5.8) and ordinary in-shot motion read as
    # a cut -- 889 of 1012 kept frames, cuts_min 180. A hard cut at 128x72
    # moves the mean by 20-250; a fast pan moves it less and for several
    # samples in a row, which the spike test (more than twice both
    # neighbours) rejects.
    # samples a row; the mean |diff| of a dark pair is scaled up by its
    # luminance (cut_mask / lum_scale) so the dark set-pieces segment too.
    cut_t = times[cut_mask(fdiff, lum)]
    return {"times": times, "dxs": dxs, "dys": dys, "divx": divx, "divy": divy, "cut_t": cut_t}


def _motion_beats(ser, beats):
    """Cheap, beat-dependent half: name the camera move around each beat —
      static | pan-L/R | tilt-U/D | (whip-)pan | push-in | pull-out | handheld,
    plus a local cut-rate (cuts/min). Returns {ts: {motion, cuts}}."""
    if not ser:
        return {}
    import numpy as np
    times = ser["times"]; dxs = ser["dxs"]; dys = ser["dys"]
    divx = ser["divx"]; cut_t = ser["cut_t"]
    divy = ser.get("divy")
    if divy is None:
        divy = divx * (H_OVER_W)          # older series: assume isotropic

    def classify(t):
        m = (times >= t - 0.7) & (times <= t + 0.7)
        m[0] = False
        if not m.any():
            return "—"
        dx, dy, dv, dvy = dxs[m], dys[m], divx[m], divy[m]
        mag = np.sqrt(dx * dx + dy * dy)
        mm = float(mag.mean())
        # coherence: how much of the motion is one consistent direction
        vmean = np.hypot(dx.mean(), dy.mean())
        coh = vmean / (mm + 1e-6)
        dvm = float(dv.mean()); dvym = float(dvy.mean())
        # Zoom before static: a centred push-in has ~zero net translation, so
        # testing `mm` first labelled every pure zoom "static". A zoom must
        # spread BOTH axes the same way on most samples; people moving inside
        # a locked-off frame spread one axis, sporadically (The Zone of
        # Interest came out push-in 208 / pull-out 242 / static 138).
        agree = (np.sign(dv) == np.sign(dvm)) & (np.sign(dvy) == np.sign(dvm))
        if (abs(dvm) > ZOOM_DIV and abs(dvm) > 0.55 * mm
                and abs(dvym) > ZOOM_DIV * H_OVER_W and dvym * dvm > 0
                and float(agree.mean()) >= 0.75):
            return "push-in" if dvm > 0 else "pull-out"
        if mm < 0.7:
            return "static"
        if coh > 0.6:
            whip = "whip-" if mm > 6 else ""
            horiz = abs(dx.mean()) >= abs(dy.mean())
            axis = dx if horiz else dy
            # a camera move shifts the frame the same way on every sample; a
            # subject crossing a locked-off frame shifts a few samples, so the
            # window mean says "pan" while most samples say nothing (The Zone
            # of Interest: pan-R 173 / tilt-D 205 on a locked-off film)
            moving = np.abs(axis) >= 0.5
            if float(((np.sign(axis) == np.sign(axis.mean())) & moving).mean()) < 0.75:
                # most samples still -> a subject crossed a static frame;
                # most samples moving but not one way -> handheld
                return "static" if float(np.median(mag)) < 0.7 else "handheld"
            if horiz:
                return f"{whip}pan-{'R' if dx.mean() > 0 else 'L'}"
            return f"{whip}tilt-{'D' if dy.mean() > 0 else 'U'}"
        return "handheld"

    out = {}
    prev_t = None
    for _, t in beats:
        cuts = int(((cut_t >= t - 15) & (cut_t <= t + 15)).sum()) * 2  # -> per minute
        # a hard cut since the previous kept frame: the shot boundary itself,
        # so a reader (or vision_pass.py) can group frames into shots without
        # re-deriving it from pixels. The 3 fps series puts a cut within 1/3 s
        # of its frame, so take the half-sample slack on both sides.
        lo = -0.2 if prev_t is None else prev_t + 0.2
        cut = bool(((cut_t > lo) & (cut_t <= t + 0.2)).any())
        out[t] = {"motion": classify(t), "cuts": cuts, "cut": "cut" if cut else ""}
        prev_t = t
    return out


def analyze_motion(video, beats, fps_s=3, W=128, H=72):
    """The channel a still can't carry: per-beat camera grammar + cut-rate.
    Pure numpy — degrades to {} on any failure. Both halves back to back;
    main() runs the heavy one in the background instead."""
    return _motion_beats(_motion_series(video, fps_s, W, H), beats)


# ---------------------------------------------------------------- outputs
def shot_caption(d):
    """One line from a shots.tsv / _vision.jsonl record: scale; setting; N people;
    action; light. The same string whichever pass renders it."""
    people = str(d.get("people") or "").strip()
    bits = [d.get("scale", ""), d.get("setting", ""), f"{people} people" if people and people != "0" else "",
            d.get("action", ""), d.get("light", "")]
    return "; ".join(b for b in bits if b)


def attach_sidecars(out_dir, rows):
    """Fill `caption` (from shots.tsv, on each shot's first row) and `text`
    (from text.tsv, per frame) into rows that lack them, so sync.md carries the
    vision captions and the OCR whichever pass writes it last. The sound-tags
    pass used to rewrite sync.md from INDEX alone and every caption vanished
    (Alien, Obsession, Zodiac: zero `[sees]` lines after a full-pipeline run)."""
    if not out_dir:
        return rows
    sp = os.path.join(out_dir, "shots.tsv")
    if os.path.exists(sp) and not any(r.get("caption") for r in rows):
        caps = {}
        lines = open(sp, encoding="utf-8").read().splitlines()
        if lines:
            hdr = lines[0].split("\t")
            for l in lines[1:]:
                d = dict(zip(hdr, l.split("\t")))
                if d.get("shot"):
                    caps[d["shot"]] = shot_caption(d)
        seen = set()
        for r in rows:
            s = r.get("shot")
            if s and s in caps and s not in seen:
                seen.add(s)
                r["caption"] = caps[s]
    tp = os.path.join(out_dir, "text.tsv")
    if os.path.exists(tp) and not any(r.get("text") for r in rows):
        txt = {}
        lines = open(tp, encoding="utf-8").read().splitlines()
        if lines:
            hdr = lines[0].split("\t")
            for l in lines[1:]:
                d = dict(zip(hdr, l.split("\t")))
                if d.get("text"):
                    txt[d.get("frame", "")] = d["text"]
        for r in rows:
            if r.get("frame") in txt:
                r["text"] = txt[r["frame"]]
    return rows


def write_sync_md(path, title, rows):
    """Render the human-readable sync index. `rows` is an iterable of dicts with
    frame/time/dialogue/motion/loudness/trend/tempo_bpm/brightness/sound, all
    strings (time is the full hms). Shared with redialogue.py so a re-dialogued
    film and a fresh run produce byte-identical sync.md. Captions and OCR text
    missing from the rows are read from the package's sidecars (attach_sidecars)."""
    # state keys: printed only when they change from the previous row. A full
    # film's sync.md was ~93k tokens of mostly repeated audio tags; a reader
    # wants the cut, and the cut is what changes.
    # bpm, trend and tone flip on most rows (Obsession: 767/699/216 of 1236)
    # and are continuous measures, not grammar; they stay in INDEX only.
    # `hear` (CLAP ambience) is shown when it holds for two rows running.
    # `text` (on-screen OCR) is a state key too: a title card held over ten
    # frames prints once.
    STATE = (("motion", "cam"), ("loudness", "loud"), ("sound", "sound"), ("impact", "impact"),
             ("field", "field"), ("voice", "voice"), ("pan", "pan"), ("ambience", "hear"),
             ("shot", "shot"), ("text", "text"))
    esc = lambda x: str(x).replace("|", "\\|")
    rows = attach_sidecars(os.path.dirname(os.path.abspath(path)), list(rows))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Sync index — {title}\n\n")
        f.write("Each row is one kept frame: image ↔ timestamp ↔ the line live then ↔ what "
                "changed. Read top-to-bottom as a cut. The last column lists only what differs "
                "from the previous row (`key=-` means it went blank; the first row lists "
                "everything); `cut` marks a hard cut before this frame; `[sees] …` is a vision-"
                "model caption on a shot's first frame; `held Ns` means this frame stands for N "
                "seconds before the next one. Keys: cam (camera move), loud, sound (event@placement), "
                "impact (LFE), field (front/surround), voice (where speech sits), pan, hear (ambience "
                "tags, shown when they hold for two rows), shot, text (on-screen text read by OCR, "
                "shown when it changes). bpm, trend and tone are in INDEX.tsv "
                "only; the full cue track is dialogue.srt / dialogue.tsv.\n\n")
        f.write("| # | frame | time | dialogue | changes |\n|--|--|--|--|--|\n")
        prev = {}
        for i, r in enumerate(rows, 1):
            bits = []
            if r.get("cut"):
                bits.append("cut")
            if r.get("caption"):
                bits.append("[sees] " + esc(r["caption"]))
            try:
                if float(r.get("held_s") or 0) >= 5:
                    bits.append(f"held {float(r['held_s']):.0f}s")
            except ValueError:
                pass
            for key, name in STATE:
                v = r.get(key) or ""
                if v == "—": v = ""
                if key == "ambience" and v and i < len(rows) and (rows[i].get(key) or "") != v:
                    v = prev.get(key, "")          # a one-row blip: not a change
                if i == 1 or v != prev.get(key, ""):
                    if v:
                        bits.append(f"{name}={esc(v)}")
                    elif i > 1:
                        bits.append(f"{name}=-")
                prev[key] = v
            f.write(f"| {i} | {r.get('frame','')} | {(r.get('time') or '')[:8]} | "
                    f"{esc(r.get('dialogue') or '')} | {', '.join(bits)} |\n")
    return path


SHEET_COLS, SHEET_ROWS, TILE_W = 6, 5, 320


def make_sheets(out_dir, rows=None):
    """Contact sheets: frames tiled SHEET_COLS x SHEET_ROWS (30 a sheet) at
    TILE_W px wide with the frame number and timestamp burned into each tile,
    so a reader can look at a whole stretch of the film in one image. Writes
    sheets/sheet_NNN.jpg and sheets.tsv (sheet · first_frame · last_frame ·
    first_time · last_time). rows: INDEX rows (dicts with frame/time); read
    from INDEX.tsv when not given. Returns the number of sheets."""
    from PIL import Image, ImageDraw
    if rows is None:
        lines = open(os.path.join(out_dir, "INDEX.tsv"), encoding="utf-8").read().splitlines()
        hdr = lines[0].split("\t")
        rows = [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]
    frames_dir = os.path.join(out_dir, "frames")
    sheets_dir = os.path.join(out_dir, "sheets")
    os.makedirs(sheets_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(sheets_dir, "sheet_*.jpg")):
        try: os.remove(stale)
        except OSError: pass
    per = SHEET_COLS * SHEET_ROWS
    index = []
    for k in range(0, len(rows), per):
        chunk = rows[k:k + per]
        tiles = []
        for r in chunk:
            try:
                with Image.open(os.path.join(frames_dir, r["frame"])) as im:
                    im = im.convert("RGB")
                    h = max(1, round(im.height * TILE_W / im.width))
                    tiles.append(im.resize((TILE_W, h)))
            except (OSError, KeyError):
                tiles.append(Image.new("RGB", (TILE_W, TILE_W * 9 // 16), (40, 40, 40)))
        th = max(t.height for t in tiles) + 16
        sheet = Image.new("RGB", (SHEET_COLS * TILE_W, SHEET_ROWS * th), (16, 16, 16))
        d = ImageDraw.Draw(sheet)
        for j, (r, t) in enumerate(zip(chunk, tiles)):
            x = (j % SHEET_COLS) * TILE_W; y = (j // SHEET_COLS) * th
            sheet.paste(t, (x, y + 16))
            label = f"{r['frame'].replace('frame_', '').replace('.jpg', '')}  {r.get('time', '')[:8]}"
            d.rectangle([x, y, x + TILE_W, y + 16], fill=(16, 16, 16))
            d.text((x + 4, y + 2), label, fill=(230, 230, 230))
        n = k // per + 1
        sheet.save(os.path.join(sheets_dir, f"sheet_{n:03d}.jpg"), quality=80)
        index.append((n, chunk[0]["frame"], chunk[-1]["frame"], chunk[0].get("time", "")[:8],
                      chunk[-1].get("time", "")[:8]))
    with open(os.path.join(out_dir, "sheets.tsv"), "w", encoding="utf-8") as f:
        f.write("sheet\tfirst_frame\tlast_frame\tfirst_time\tlast_time\n")
        for n, a, b, ta, tb in index:
            f.write(f"sheet_{n:03d}.jpg\t{a}\t{b}\t{ta}\t{tb}\n")
    return len(index)


# Bump when a change makes new outputs differ from old ones for the same film,
# so MANIFEST says which library entries predate it. History:
#   1  3-digit frame names (before 2026-09-21)
#   2  5-digit frame names
#   3  push-in/pull-out fixed + sub-pixel motion; HDR tonemapped; sound-event
#      gap merge fixed; VobSub/DVB OCR (2026-09-23)
#   4  scene select on the 640px frame; one decode per film (2026-11-01)
#   5  spatial audio layer: impact/field/voice/pan columns, sound placement,
#      loudness on BS.1770 momentary LUFS; `cut` column (2026-11-01)
#   6  zooms need both axes to diverge consistently (static films no longer
#      read push-in/pull-out); cuts need an absolute jump; events.tsv (2026-11-01)
#   7  dialogue carries every cue (was: the line live at the frame -- a third
#      reached INDEX on a static film); a frame kept at least every --max-gap s
#      inside held shots; held_s column; dialogue.srt + dialogue.tsv; TSV
#      fields never carry tab/newline/ASCII quote; sheets/ (2026-11-01)
#   8  Apple Vision OCR on every extracted frame before dedup: a new word keeps
#      a frame (text typing in over a static shot used to collapse with the
#      blank screen before it); `text` column + text.tsv; cuts scaled by
#      luminance; voice gated on a live cue; MANIFEST origin:/container:;
#      sync.md carries captions/OCR from the sidecars; frames 960 px wide at
#      JPEG q3 (were 640 px, default quality) (2026-11-02)
OUTPUT_FORMAT = 8


def engine_version():
    """git describe of the checkout this file lives in (through the deploy
    symlink), e.g. 98a515e or 98a515e-dirty; 'unknown' outside a checkout."""
    here = os.path.dirname(os.path.realpath(__file__))
    try:
        r = run(["git", "-C", here, "describe", "--always", "--dirty", "--abbrev=10"])
    except OSError:
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def write_outputs(out_dir, beats, cues, sub_note, per_beat, sounds, video, notes=None,
                  motion=None):
    """notes: optional MANIFEST lines -- frames, audio, motion, color, engine.
    Missing keys fall back to a generic description."""
    motion = motion or {}
    notes = notes or {}
    frames_dir = os.path.join(out_dir, "frames")
    idx_path = os.path.join(out_dir, "INDEX.tsv")
    rows = []
    prev_t = None
    film_dur = notes.get("duration") or (beats[-1][1] if beats else 0.0)
    for k, (name, t) in enumerate(beats):
        b = per_beat.get(t, {}); mv = motion.get(t, {})
        nxt = beats[k + 1][1] if k + 1 < len(beats) else max(film_dur, t)
        rows.append({"frame": name, "time": hms(t), "held_s": f"{max(0.0, nxt - t):.1f}",
                     "dialogue": lines_since(prev_t, t, cues),
                     "motion": mv.get("motion", ""),
                     "cuts": str(mv.get("cuts", "") if mv else ""),
                     "cut": mv.get("cut", ""),
                     "loudness": b.get("loud", ""), "trend": b.get("trend", ""),
                     "tempo_bpm": str(b.get("tempo") or ""),
                     "brightness": b.get("bright", ""),
                     "sound": sound_at(t, sounds),
                     "impact": b.get("impact", ""), "field": b.get("field", ""),
                     "voice": b.get("voice", ""), "pan": b.get("pan", "")})
        prev_t = t
    carried, total = cue_coverage(rows, cues)
    if cues:
        write_srt(os.path.join(out_dir, "dialogue.srt"), cues)
        write_dialogue_tsv(os.path.join(out_dir, "dialogue.tsv"), cues, beats)

    # on-screen text (OCR of the kept frames, from the dedup pass): a `text`
    # column, and text.tsv for the frames that carry any
    kept_text = notes.get("text")
    for r in rows:
        txt = tsv_safe((kept_text or {}).get(r["frame"], ""))
        r["text"] = txt if ocr_words(txt) else ""
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write("frame\ttime\tdialogue\tmotion\tcuts_min\tloudness\ttrend\ttempo_bpm\tbrightness\tsound"
                "\timpact\tfield\tvoice\tpan\tcut\theld_s" + ("\ttext" if kept_text is not None else "") + "\n")
        for r in rows:
            # a tab or newline inside a subtitle line would shear the TSV row
            dlg = tsv_safe(r["dialogue"])
            f.write("\t".join([r["frame"], r["time"], dlg, r["motion"], r["cuts"],
                               r["loudness"], r["trend"], r["tempo_bpm"],
                               r["brightness"], r["sound"], r["impact"], r["field"],
                               r["voice"], r["pan"], r["cut"], r["held_s"]]
                              + ([r["text"]] if kept_text is not None else [])) + "\n")
    n_text = 0
    if kept_text is not None:
        tp = os.path.join(out_dir, "text.tsv")
        with open(tp + ".tmp", "w", encoding="utf-8") as f:
            f.write("frame\ttime\ttext\n")
            for r in rows:
                if r["text"]:
                    f.write(f"{r['frame']}\t{r['time']}\t{r['text']}\n"); n_text += 1
        os.replace(tp + ".tmp", tp)

    # human-readable
    write_sync_md(os.path.join(out_dir, "sync.md"), os.path.basename(video), rows)
    n_sheets = make_sheets(out_dir, rows) if notes.get("sheets", True) else 0

    if sounds:
        with open(os.path.join(out_dir, "sounds.tsv"), "w", encoding="utf-8") as f:
            f.write("label\tfirst_time\tcount\tsecs\tpitch\ttexture\tplacement\n")
            from collections import Counter
            agg = {}
            for s in sounds:
                a = agg.setdefault(s["label"], {"t": s["t"], "n": 0, "secs": 0.0, "pitch": s["pitch"],
                                                "texture": s["texture"], "pl": Counter()})
                a["n"] += 1; a["secs"] += s["dur"]
                if s.get("placement"): a["pl"][s["placement"]] += 1
            for lab, a in sorted(agg.items(), key=lambda kv: kv[1]["t"]):
                pl = a["pl"].most_common(1)[0][0] if a["pl"] else ""
                f.write(f"{lab}\t{hms(a['t'])[:8]}\t{a['n']}\t{a['secs']:.1f}\t{a['pitch']}\t"
                        f"{a['texture']}\t{pl}\n")

    if sounds:
        # every event with its own time: sounds.tsv is per label, and a
        # tagger (sound_tags.py) needs the spans themselves
        with open(os.path.join(out_dir, "events.tsv"), "w", encoding="utf-8") as f:
            f.write("t\tdur\tlabel\tpitch\ttexture\tplacement\n")
            for s in sorted(sounds, key=lambda s: s["t"]):
                f.write(f"{s['t']:.2f}\t{s['dur']}\t{s['label']}\t{s['pitch']}\t{s['texture']}\t"
                        f"{s.get('placement', '')}\n")

    man = os.path.join(out_dir, "MANIFEST.txt")
    n_sounds = len(set(s["label"] for s in sounds))
    audio = notes.get("audio") or (
        f"loudness/trend/tempo/brightness per beat; {n_sounds} distinct "
        f"non-speech sound(s) in sounds.tsv" if per_beat else "(none)")
    if notes.get("spatial"):
        audio += f"; {notes['spatial']}"
    mot = notes.get("motion") or (
        "camera-grammar tag per beat (static/pan/tilt/push-in/pull-out/handheld) + cuts_min"
        if motion else "(none)")
    with open(man, "w", encoding="utf-8") as f:
        f.write(f"source: {video}\n")
        # which cut this is: the local file is a slug (originals/alien-1979.mkv),
        # so the NAS path the batch driver pulled (CRV_ORIGIN) and the
        # container's own duration and size go on record. Alien was read as
        # the Director's Cut from notes written against another encode.
        origin = os.environ.get("CRV_ORIGIN", "")
        if origin:
            f.write(f"origin: {origin}\n")
        try:
            f.write(f"container: {hms(film_dur)[:8]} · {os.path.getsize(video)} bytes"
                    f" · {os.path.basename(video)}\n")
        except OSError:
            pass
        f.write(f"frames: {len(beats)} kept ({notes.get('frames', 'deduped, uncapped')})\n")
        f.write(f"frames dir: {frames_dir}\n")
        if notes.get("color"):
            f.write(f"color: {notes['color']}\n")
        f.write(f"subtitles: {sub_note}" + ("; verbatim in dialogue.srt, per-frame in dialogue.tsv" if cues else "") + "\n")
        if cues:
            f.write(f"dialogue coverage: {carried}/{total} cues carried in INDEX.tsv"
                    + ("" if carried == total else " -- INCOMPLETE") + "\n")
        f.write(f"sync index: INDEX.tsv (frame -> time -> dialogue -> motion -> audio) | human: sync.md\n")
        if n_sheets:
            f.write(f"sheets: {n_sheets} contact sheets in sheets/ ({SHEET_COLS}x{SHEET_ROWS} frames each, "
                    f"index sheets.tsv)\n")
        if kept_text is not None:
            f.write(f"ocr: Apple Vision on {notes.get('ocr_n', 0)} extracted frames before dedup "
                    f"(a new word keeps a frame), {n_text} kept frames with text in text.tsv / "
                    f"INDEX `text` ({notes.get('ocr_s', 0.0):.0f}s)"
                    + (f" -- {notes['ocr_failed']} frames UNREAD (OCR timed out)"
                       if notes.get("ocr_failed") else "") + "\n")
        f.write(f"motion: {mot}\n")
        f.write(f"audio layer: {audio}\n")
        f.write(f"engine: {notes.get('engine') or engine_version()} "
                f"(output format {OUTPUT_FORMAT})\n")
        if notes.get("timings"):
            # a callable so the total includes writing the index itself
            t = notes["timings"]
            f.write(f"timings: {t() if callable(t) else t}\n")
    return idx_path, man


def _timed(fn, *args):
    """Run fn in a worker and return (result, seconds) so a background pass
    reports its own cost separately from how long main() waited for it."""
    t0 = time.perf_counter()
    return fn(*args), time.perf_counter() - t0


def timings_line(tim, total):
    """One MANIFEST line: 'timings: extract 310.0s · audio 120.0s (waited 0.0s) · … · total 600.0s'.
    Background passes carry their own duration plus the wait at collect time."""
    parts = []
    for k, v in tim.items():
        if isinstance(v, tuple):
            parts.append(f"{k} {v[0]:.1f}s (waited {v[1]:.1f}s)")
        else:
            parts.append(f"{k} {v:.1f}s")
    parts.append(f"total {total:.1f}s")
    return " · ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="crv + sync index + mechanical audio layer")
    ap.add_argument("source")
    ap.add_argument("-o", "--out", default="crv-out")
    ap.add_argument("--scene", type=float, default=0.30)
    ap.add_argument("--fps-floor", type=float, default=1.0,
                    help="seconds between guaranteed frames (upstream crv semantics) "
                         "— a frame is kept at least this often regardless of --scene")
    ap.add_argument("--dedup-threshold", type=float, default=8.0)
    ap.add_argument("--dedup-window", type=int, default=4)
    ap.add_argument("--max-gap", type=float, default=10.0,
                    help="keep a frame at least this often (seconds) even inside a held shot, "
                         "so time and the per-frame audio state keep ticking (0 = off)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--chunk-mins", type=float, default=0.0,
                    help="transcribe audio in N-minute silence-aligned segments, "
                         "each in its own whisper process, so a decoder loop cannot "
                         "spread past one segment (0 = single pass, repair on demand)")
    ap.add_argument("--vad", action="store_true",
                    help="gate whisper with the installed Silero VAD model. Off by "
                         "default: it halved word recall on a 3h feature (see "
                         "transcribe_whisper); loops are already held by -mc 0 + repair")
    ap.add_argument("--no-transcribe", action="store_true",
                    help="Don't OCR/transcribe dialogue when the source has no text subtitles")
    ap.add_argument("--no-sheets", action="store_true", help="skip the contact sheets")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip Apple Vision OCR of the extracted frames (the dedup keep "
                         "rule for on-screen text, and the INDEX text column)")
    ap.add_argument("--hwdec", action="store_true",
                    help="decode for frame extraction with VideoToolbox (software "
                         "retry on failure); off by default, see extract_frames_ts")
    a = ap.parse_args()
    t_start = time.perf_counter()
    tim = {}          # stage -> seconds, or (own seconds, seconds waited) for background passes
    _last = [t_start]
    def lap(name):
        now = time.perf_counter(); tim[name] = now - _last[0]; _last[0] = now
        return tim[name]

    # check before makedirs: a typo'd path should not leave an empty out dir behind
    if not a.source.startswith(("http://", "https://")) and not os.path.exists(a.source):
        print(f"error: {a.source} not found", file=sys.stderr); sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    frames_dir = os.path.join(a.out, "frames")

    # One decode. The frame extraction's ffmpeg also writes the motion pass's
    # gray stream and the audio layer's PCM (see _extract_cmd); the analyses
    # then run on those files in background threads while dedup and the
    # dialogue chain proceed, and are collected once the beats and cues exist.
    src_has_audio = has_audio(a.source)
    do_audio = (not a.no_audio) and src_has_audio
    do_motion = not a.no_motion
    ainfo = audio_info(a.source) if do_audio else {"channels": 0, "layout": "", "names": None}
    do_spatial = do_audio and ainfo["names"] is not None and ainfo["channels"] >= 2
    side = {}
    if do_motion: side["motion"] = os.path.join(a.out, "_motion.gray")
    if do_audio: side["audio"] = os.path.join(a.out, "_audio16k.s16le")
    if do_spatial: side["spatial"] = os.path.join(a.out, "_spatial8k.s16le")
    failed = {}
    pool = fut_audio = fut_motion = None

    def collect(fut, what):
        # one failed background pass costs us that layer, not the whole run and
        # the frames we already have.
        if fut is None:
            return None
        t0 = time.perf_counter()
        try:
            res, own = fut.result()
        except Exception as e:
            print(f"  warning: {what} analysis failed ({e})", file=sys.stderr, flush=True)
            failed[what] = f"{type(e).__name__}: {e}"[:120]
            tim[what] = (0.0, time.perf_counter() - t0)
            return None
        tim[what] = (own, time.perf_counter() - t0)
        if res is None:
            failed[what] = "decode produced nothing usable"
        return res

    color = video_color(a.source)
    color_note = ""
    if color["hdr"]:
        color_note = (f"HDR ({color['transfer']}/{color['primaries']}) tonemapped to "
                      f"SDR BT.709 (203-nit reference white)")
    if color["dv_profile"] is not None:
        dv = f"Dolby Vision profile {color['dv_profile']}"
        if not color["hdr"] or color["dv_compat"] in (0, "0"):
            # profile 5 has no HDR10/SDR base layer: the pixels are IPT, and
            # nothing here (or in Homebrew ffmpeg) can reshape them
            dv += (" with no compatible base layer — colours will be wrong "
                   "(green/purple cast); prefer an HDR10 or SDR release")
            print(f"  warning: {dv}", file=sys.stderr, flush=True)
        color_note = f"{color_note}; {dv}" if color_note else dv
    if color_note:
        print(f"• {color_note}", flush=True)
    lap("probe")

    def rm_side():
        for f in side.values():
            try: os.remove(f)
            except OSError: pass

    def die(msg):
        print(f"error: {msg}", file=sys.stderr)
        if pool: pool.shutdown(wait=False, cancel_futures=True)
        rm_side()
        sys.exit(1)

    print("• extracting frames + timestamps"
          + (" + motion stream" if do_motion else "") + (" + audio" if do_audio else "")
          + "...", flush=True)
    raw = extract_frames_ts(a.source, frames_dir, a.scene, a.fps_floor,
                            color=color, hwdec=a.hwdec, side=side)
    print(f"  {len(raw)} extracted ({lap('extract'):.1f}s)", flush=True)
    if raw and (do_audio or do_motion):
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=2)
        if do_audio:
            print("• analyzing audio in background (energy/tempo/brightness + distinct "
                  "sounds" + (f" + spatial {ainfo['layout']}" if do_spatial else "")
                  + ", BS.1770 loudness)...", flush=True)
            fut_audio = pool.submit(_timed, _audio_all, a.source, side["audio"],
                                    side.get("spatial"), ainfo["names"])
        if do_motion:
            print("• analyzing motion in background "
                  "(camera grammar: static/pan/push/handheld + cut-rate)...", flush=True)
            fut_motion = pool.submit(_timed, _motion_series, a.source, MOTION_FPS, MOTION_W,
                                     MOTION_H, side["motion"])
    if not raw:
        # no frames means no index; reporting "✓ Done" here made a failed decode
        # look like a successful run to whatever called us.
        die(f"no frames extracted from {a.source} — unreadable or not a video?")
    # The fps floor guarantees a frame every --fps-floor seconds to the end, so
    # a last frame well short of the duration means the decode stopped early
    # (truncated file, I/O error). Shipping that as a finished film would let
    # batch_encode delete the original.
    dur = duration(a.source)
    slack = max(60.0, 2 * a.fps_floor)
    if dur > slack and raw[-1][1] < dur - slack:
        die(f"frames stop at {hms(raw[-1][1])[:8]} of {hms(dur)[:8]} — "
            f"decode ended early (truncated or unreadable source?)")
    # OCR every extracted frame before dedup: the reading is a keep signal
    # (text typing in over a static shot) and the kept frames' text goes to
    # INDEX / text.tsv. ~12 ms a frame on the Neural Engine, 4 workers.
    ocr, kept_text, ocr_failed = None, None, []
    ocr_bin = None if a.no_ocr else _vision_ocr_bin()
    if ocr_bin:
        print("• reading on-screen text (Apple Vision)...", flush=True)
        ocr_failed = []
        ocr = vision_ocr(ocr_bin, [p for p, _ in raw], failed=ocr_failed)
        kept_text = {}
        print(f"  {sum(1 for v in ocr.values() if ocr_words(v))}/{len(raw)} frames carry text "
              f"({lap('ocr'):.1f}s)", flush=True)
    print("• deduplicating...", flush=True)
    beats = dedup(raw, frames_dir, a.dedup_threshold, a.dedup_window, a.max_gap,
                  ocr=ocr, kept_text=kept_text)
    print(f"  {len(beats)} kept ({lap('dedup'):.1f}s)", flush=True)

    print("• reading subtitles...", flush=True)
    cues, sub_note = get_cues(a.source, a.source, a.out)
    print(f"  {len(cues)} cues ({sub_note})", flush=True)
    # Fallback chain for missing dialogue: image-based (PGS) subs are the real
    # theatrical lines — OCR them with Apple Vision (accurate on stylized fonts).
    # Only if that's unavailable/fails do we transcribe the audio with Whisper.
    if not cues and not a.no_transcribe:
        bmp = _pick_bitmap_sub(_sub_streams(a.source))
        if bmp is not None and bmp["codec"] == "hdmv_pgs_subtitle":
            print("• image-based (PGS) subtitles — OCR with Apple Vision...", flush=True)
            cues, sub_note = transcribe_pgs_vision(a.source, a.out, stream=f"0:s:{bmp['idx']}")
            print(f"  {len(cues)} cues ({sub_note})", flush=True)
        if not cues and bmp is not None:
            # VobSub/DVB (pgsrip only reads PGS), or PGS that pgsrip choked on
            print(f"• image-based ({bmp['codec']}) subtitles — render + Apple Vision OCR...",
                  flush=True)
            cues, sub_note = transcribe_bitmap_vision(a.source, a.out, bmp)
            print(f"  {len(cues)} cues ({sub_note})", flush=True)
        if not cues and has_audio(a.source):
            print("• no readable subtitles — transcribing dialogue with Whisper...", flush=True)
            cues, sub_note = transcribe_whisper(a.source, a.out, chunk_mins=a.chunk_mins,
                                                vad="auto" if a.vad else "")
            print(f"  {len(cues)} cues ({sub_note})", flush=True)
    sdh, is_sdh = sdh_cues(cues)
    if is_sdh:
        # the captioner's sound descriptions; sound_tags.py reads this note
        # and lets them lead the ambience column
        sub_note += f" (SDH: {len(sdh)} sound descriptions)"
        print(f"  SDH track: {len(sdh)} sound descriptions in the cues", flush=True)
    lap("dialogue")

    per_beat, sounds = {}, []
    if do_audio:
        print("• scoring beats against the audio...", flush=True)
        per_beat, sounds = _audio_beats(collect(fut_audio, "audio"), beats, cues)
        print(f"  {len(set(s['label'] for s in sounds))} distinct non-speech sound(s) "
              f"({tim['audio'][0]:.1f}s, waited {tim['audio'][1]:.1f}s)", flush=True)
        _last[0] = time.perf_counter()

    motion = {}
    if do_motion:
        print("• scoring beats against the motion...", flush=True)
        motion = _motion_beats(collect(fut_motion, "motion"), beats)
        from collections import Counter
        tags = Counter(v["motion"] for v in motion.values())
        print(f"  {dict(tags)} ({tim['motion'][0]:.1f}s, waited {tim['motion'][1]:.1f}s)", flush=True)
        _last[0] = time.perf_counter()
    if pool: pool.shutdown()
    rm_side()

    if a.no_audio: audio_note = "(skipped — --no-audio)"
    elif not src_has_audio: audio_note = "(skipped — no audio track)"
    elif "audio" in failed: audio_note = f"FAILED — {failed['audio']}"
    else: audio_note = None
    if not do_audio or audio_note:
        spatial_note = ""
    elif not do_spatial:
        spatial_note = (f"no spatial layer ({ainfo['channels']}ch "
                        f"{ainfo['layout'] or 'unknown layout'})")
    elif ainfo["channels"] == 2:
        spatial_note = "spatial: stereo (voice/pan only; no LFE, no rears)"
    else:
        spatial_note = f"spatial: {ainfo['layout']} (impact/field/voice/pan + sound placement)"
    if do_audio and not audio_note:
        spatial_note += "; loudness: BS.1770 momentary LUFS"
    if a.no_motion: motion_note = "(skipped — --no-motion)"
    elif "motion" in failed: motion_note = f"FAILED — {failed['motion']}"
    else: motion_note = None
    notes = {"frames": f"scene > {a.scene:g} + a frame every {a.fps_floor:g}s, deduped "
                       f"(threshold {a.dedup_threshold:g}, window {a.dedup_window}, a frame kept at "
                       f"least every {a.max_gap:g}s), uncapped", "duration": dur,
             "audio": audio_note, "motion": motion_note, "color": color_note,
             "spatial": spatial_note, "sheets": not a.no_sheets,
             "text": kept_text, "ocr_n": len(raw) if ocr is not None else 0,
             "ocr_failed": len(ocr_failed),
             "ocr_s": tim.get("ocr", 0.0)}
    notes["timings"] = lambda: timings_line(tim, time.perf_counter() - t_start)
    idx, man = write_outputs(a.out, beats, cues, sub_note, per_beat, sounds, a.source,
                             notes, motion)
    lap("write")
    if cues:
        cov = next((l for l in open(man, encoding="utf-8") if l.startswith("dialogue coverage:")), "").strip()
        print(f"  {cov}", flush=True)
    print(f"\n✓ Done → {a.out}  [{timings_line(tim, time.perf_counter() - t_start)}]")
    print(f"  {len(beats)} frames · {len(cues)} dialogue cues · "
          f"{len(set(s['label'] for s in sounds))} distinct sounds · "
          f"{'motion tagged' if motion else 'no motion'}")
    print(f"  sync index: {idx}")
    print(f"  manifest:   {man}")


if __name__ == "__main__":
    main()
