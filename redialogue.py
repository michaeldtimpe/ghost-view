#!/usr/bin/env python3
"""Patch dialogue into an already-built crv output folder without re-extracting
frames. Transcribes the audio with Whisper, then joins the timecoded lines onto
the existing frames using the timestamps already recorded in INDEX.tsv, and
regenerates sync.md + the MANIFEST subtitle note.

    redialogue.py <crv_out_dir> <local_media_or_wav_or_srt> [--chunk-mins N] [--lang xx]
"""
import argparse, os, re, sys
import enhance_process as E


def hms_to_sec(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("out_dir", help="existing crv output folder (must contain INDEX.tsv)")
    p.add_argument("media", help="media file, .wav, or .srt/.vtt")
    p.add_argument("--vad", action="store_true",
                   help="gate whisper with the Silero VAD model (off by default: halves word recall)")
    p.add_argument("--chunk-mins", type=float, default=0.0,
                   help="N-minute silence-aligned segments so a decoder loop can't spread "
                        "— use on films that stay bad after automatic repair")
    p.add_argument("--lang", default="en")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir, media = args.out_dir, args.media

    idx = os.path.join(out_dir, "INDEX.tsv")
    if not os.path.exists(idx):
        print(f"✗ {idx} not found"); sys.exit(1)
    if not os.path.exists(media):
        print(f"✗ {media} not found"); sys.exit(1)

    # accept a ready-made subtitle file (e.g. OCR'd PGS) or transcribe from audio
    if media.lower().endswith((".srt", ".vtt")):
        cues = E.parse_srt(media)
        note = f"OCR/subtitle file ({os.path.basename(media)}, {len(cues)} lines)"
    else:
        cues, note = E.transcribe_whisper(media, out_dir, lang=args.lang, chunk_mins=args.chunk_mins,
                                          vad="auto" if args.vad else "")
    if not cues:
        print(f"✗ {out_dir}: {note} — nothing changed"); sys.exit(1)

    lines = open(idx, encoding="utf-8").read().splitlines()
    header = lines[0].split("\t")
    ti, di = header.index("time"), header.index("dialogue")
    rows = [l.split("\t") for l in lines[1:]]

    filled = 0
    prev_t = None
    for r in rows:
        while len(r) < len(header):
            r.append("")
        t = hms_to_sec(r[ti])
        line = E.tsv_safe(E.lines_since(prev_t, t, cues))
        prev_t = t
        r[di] = line
        if line:
            filled += 1
    E.write_srt(os.path.join(out_dir, "dialogue.srt"), cues)
    fi = header.index("frame")
    E.write_dialogue_tsv(os.path.join(out_dir, "dialogue.tsv"), cues,
                         [(r[fi], hms_to_sec(r[ti])) for r in rows])

    tmp = idx + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    os.replace(tmp, idx)

    # regenerate sync.md from the (now dialogue-filled) columns
    keys = ("frame", "time", "dialogue", "motion", "loudness", "trend",
            "tempo_bpm", "brightness", "sound", "impact", "field", "voice", "pan", "cut",
            "shot", "ambience", "held_s")
    ci = {n: (header.index(n) if n in header else -1) for n in keys}
    def g(r, n):
        j = ci[n]; return r[j] if 0 <= j < len(r) else ""
    row_dicts = [{n: g(r, n) for n in keys} for r in rows]

    title = os.path.basename(os.path.normpath(out_dir))
    man = os.path.join(out_dir, "MANIFEST.txt")
    manifest_txt = None
    if os.path.exists(man):
        manifest_txt = open(man, encoding="utf-8").read()
        m = re.search(r"(?m)^source: (.*)$", manifest_txt)
        if m:
            title = os.path.basename(m.group(1).strip())

    E.write_sync_md(os.path.join(out_dir, "sync.md"), title, row_dicts)

    if manifest_txt is not None:
        manifest_txt = re.sub(r"(?m)^subtitles: .*$", lambda _: f"subtitles: {note}",
                               manifest_txt, count=1)
        open(man, "w", encoding="utf-8").write(manifest_txt)

    print(f"✓ {os.path.basename(out_dir)}: {note}; {filled}/{len(rows)} frames now carry dialogue")


if __name__ == "__main__":
    main()
