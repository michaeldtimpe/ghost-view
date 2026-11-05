#!/usr/bin/env python3
"""Join PGS timing (timing.tsv) with Vision OCR text (vision_out.tsv) into an SRT."""
import os, sys
timing_tsv, vision_tsv, out_srt = sys.argv[1], sys.argv[2], sys.argv[3]

timing = {}
for line in open(timing_tsv, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        continue
    idx, start, end = parts
    timing[idx] = (start, end)

text = {}
for line in open(vision_tsv, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    p = parts[0]
    t = parts[1] if len(parts) > 1 else ""
    idx = os.path.splitext(os.path.basename(p))[0]
    text[idx] = t.strip()

rows = []
for idx in sorted(timing):
    t = text.get(idx, "")
    if not t:
        continue
    start, end = timing[idx]
    rows.append((start, end, t))

with open(out_srt, "w", encoding="utf-8") as f:
    for i, (start, end, t) in enumerate(rows, 1):
        f.write(f"{i}\n{start} --> {end}\n{t}\n\n")
print(f"{len(rows)} cues -> {out_srt}")
