#!/usr/bin/env python3
"""Apple Vision OCR on every kept frame of a finished crv package.

Usage: ocr_pass.py <out_dir> [--workers N] [--force]

Reads each INDEX row's frames/<frame>, keeps a reading with at least three
alphanumeric characters, writes text.tsv (frame, time, text; only frames with
text), fills an INDEX `text` column, sets the MANIFEST `ocr:` line and rebuilds
sync.md. Skips a package that already has an `ocr:` line unless --force.
"""
import argparse, os, re, sys, time
import enhance_process as E
import pkg


def keep(txt):
    txt = E.tsv_safe(txt)
    return txt if len(re.sub(r"[^A-Za-z0-9]", "", txt)) >= 3 else ""


def run(out_dir, workers=4, force=False):
    if pkg.manifest_get(out_dir, "ocr") and not force:
        print(f"ocr already done: {pkg.manifest_get(out_dir, 'ocr')} (--force to redo)")
        return 0
    binp = E._vision_ocr_bin()
    if not binp:
        print("Apple Vision OCR helper not found (vision_ocr / vision_ocr.swift)", file=sys.stderr)
        return 1
    header, rows = pkg.read_index(out_dir)
    paths = [os.path.join(pkg.frames_dir(out_dir), r["frame"]) for r in rows]
    t0 = time.time()
    got = E.vision_ocr(binp, paths, workers=workers)
    secs = time.time() - t0
    for r in rows:
        r["text"] = keep(got.get(os.path.splitext(r["frame"])[0], ""))
    tp = os.path.join(out_dir, "text.tsv")
    with open(tp + ".tmp", "w", encoding="utf-8") as f:
        f.write("frame\ttime\ttext\n")
        for r in rows:
            if r["text"]:
                f.write(f"{r['frame']}\t{r['time']}\t{r['text']}\n")
    os.replace(tp + ".tmp", tp)
    pkg.write_index(out_dir, pkg.add_column(header, "text"), rows)
    n_text = sum(1 for r in rows if r["text"])
    pkg.manifest_set(out_dir, "ocr",
                     f"Apple Vision on {len(rows)} frames, {n_text} with text ({secs:.0f}s)",
                     after="vision")
    pkg.rebuild_sync(out_dir, rows)
    print(f"ocr: {len(rows)} frames, {n_text} with text, {secs:.1f}s -> {tp}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return run(a.out_dir, a.workers, a.force)


if __name__ == "__main__":
    sys.exit(main())
