#!/usr/bin/env python3
"""Clear the INDEX `voice` placement on rows where no subtitle cue is live.

Usage: voice_gate.py <out_dir>

A voice placement with no line on screen is music or effects read as speech.
Leaves a package without dialogue.srt as it is. Idempotent.
"""
import argparse, sys
import enhance_process as E
import pkg


def run(out_dir):
    cues = pkg.cues(out_dir)
    if not cues:
        print("no cue track, voice left as is")
        return 0
    header, rows = pkg.read_index(out_dir)
    had = cleared = 0
    for r in rows:
        if not r.get("voice"):
            continue
        had += 1
        if not E.line_at(pkg.sec(r["time"]), cues):
            r["voice"] = ""
            cleared += 1
    pkg.write_index(out_dir, header, rows)
    # a second run clears nothing; keep the first run's count on record
    if cleared or not pkg.manifest_get(out_dir, "voice"):
        pkg.manifest_set(out_dir, "voice",
                         f"gated on the cue track ({cleared} of {had} placements cleared, no cue live)")
    pkg.rebuild_sync(out_dir, rows)
    print(f"voice: {cleared} of {had} placements cleared")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    return run(ap.parse_args(argv).out_dir)


if __name__ == "__main__":
    sys.exit(main())
