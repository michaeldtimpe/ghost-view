#!/usr/bin/env python3
"""Render each PGS subtitle to an upscaled PNG (good for OCR) + a timing table,
reusing the engine's renderer. Usage: pgs_to_pngs.py <.sup> <outdir>

timing.tsv rows are `key<TAB>start<TAB>end` as SRT timestamps. A subtitle with
no end of its own is held until the next one (max 4s), exactly as the engine
does -- writing end=start made a zero-length cue that no frame ever matched."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import enhance_process as E

sup, outdir = sys.argv[1], sys.argv[2]
items = E.pgs_render(sup, outdir)
timing = E.resolve_ends([(k, st, en) for k, _, st, en in items])
with open(os.path.join(outdir, "timing.tsv"), "w", encoding="utf-8") as f:
    for key in sorted(timing):
        st, en = timing[key]
        f.write(f"{key}\t{E.srt_ts(st)}\t{E.srt_ts(en)}\n")
print(f"wrote {len(items)} subtitle images to {outdir}")
