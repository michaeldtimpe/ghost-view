#!/usr/bin/env python3
"""Map an older package's frame numbers onto a regenerated package's.

Usage: cite_map.py <old_dir> <new_dir>

Every old frame is matched to the new frame showing the same image: 24x14
grayscale thumbnails of the picture inside any letterbox bars, z-normalised, scored by Pearson correlation. When both
packages have an INDEX.tsv with times, candidates are limited to new frames
within 20 s (all frames if none is that close). Writes <new_dir>/cite_map.tsv
(old_frame, old_time, new_frame, new_time, score, dt_s) and a MANIFEST
`cite map:` line on the new package.
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import pkg

W, H, WINDOW, CHUNK = 24, 14, 20.0, 500


def picture_box(paths, step=10, floor=4.0):
    """(left, top, right, bottom) of the picture inside letterbox/pillarbox bars,
    from the mean brightness of each row and column over every step-th frame.
    One package kept the bars and the other cropped them (Alien: 640x360 vs
    640x267), and a whole-frame thumbnail then never matches."""
    rows = cols = 0.0
    sample = paths[::step]
    for p in sample:
        with Image.open(p) as im:
            a = np.asarray(im.convert("L"), np.float32)
        rows, cols = rows + a.mean(axis=1), cols + a.mean(axis=0)
    r = np.nonzero(rows / len(sample) > floor)[0]
    c = np.nonzero(cols / len(sample) > floor)[0]
    if not len(r) or not len(c):
        return None
    return int(c[0]), int(r[0]), int(c[-1]) + 1, int(r[-1]) + 1


def thumbs(out_dir):
    paths = sorted(glob.glob(os.path.join(pkg.frames_dir(out_dir), "*.jpg")))
    box = picture_box(paths) if paths else None
    v = np.empty((len(paths), W * H), np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("L")
            v[i] = np.asarray((im.crop(box) if box else im).resize((W, H), Image.LANCZOS),
                              np.float32).ravel()
    v -= v.mean(axis=1, keepdims=True)
    v /= v.std(axis=1, keepdims=True) + 1e-6
    return [os.path.basename(p) for p in paths], v


def times(out_dir):
    """{frame: 'HH:MM:SS.mmm'} from INDEX.tsv, or {} when there is none or no time column."""
    if not os.path.exists(os.path.join(out_dir, "INDEX.tsv")):
        return {}
    header, rows = pkg.read_index(out_dir)
    if "time" not in header:
        return {}
    return {r["frame"]: r["time"] for r in rows if r.get("time")}


def match(ov, nv, new_t, centre):
    """Best new frame and its score for each old frame, in chunks of CHUNK old
    frames. centre: per-old-frame time (NaN = unknown) to window around, or None."""
    best, score = np.empty(len(ov), int), np.empty(len(ov), np.float32)
    for a in range(0, len(ov), CHUNK):
        sim = ov[a:a + CHUNK] @ nv.T / (W * H)
        for k, s in enumerate(sim):
            if centre is not None and not np.isnan(centre[a + k]):
                near = np.abs(new_t - centre[a + k]) <= WINDOW     # NaN compares False
                if near.any():
                    s = np.where(near, s, -np.inf)
            best[a + k] = j = int(np.argmax(s))
            score[a + k] = sim[k, j]
    return best, score


def run(old_dir, new_dir, tag=""):
    """tag: names this map when a package has several older generations --
    cite_map_<tag>.tsv and a `cite map <tag>:` MANIFEST line."""
    old_names, ov = thumbs(old_dir)
    new_names, nv = thumbs(new_dir)
    if not old_names or not new_names:
        print("no frames to match", file=sys.stderr)
        return 1
    ot, nt = times(old_dir), times(new_dir)
    new_t = np.array([pkg.sec(nt[n]) if n in nt else np.nan for n in new_names])
    old_t = np.array([pkg.sec(ot[n]) if n in ot else np.nan for n in old_names])
    timed = bool(ot and nt)
    best, score = match(ov, nv, new_t, old_t if timed else None)
    out = []
    for i, name in enumerate(old_names):
        nn = new_names[best[i]]
        t = old_t[i] if timed else np.nan
        dt = f"{new_t[best[i]] - t:.1f}" if not np.isnan(t) and nn in nt else ""
        out.append((name, ot.get(name, ""), nn, nt.get(nn, ""), float(score[i]), dt))
    tp = os.path.join(new_dir, f"cite_map_{tag}.tsv" if tag else "cite_map.tsv")
    with open(tp + ".tmp", "w", encoding="utf-8") as f:
        f.write("old_frame\told_time\tnew_frame\tnew_time\tscore\tdt_s\n")
        for o, ot_, n, nt_, sc, dt in out:
            f.write(f"{o}\t{ot_}\t{n}\t{nt_}\t{sc:.3f}\t{dt}\n")
    os.replace(tp + ".tmp", tp)
    scores = np.array([r[4] for r in out])
    med, low = float(np.median(scores)), int((scores < 0.8).sum())
    pkg.manifest_set(new_dir, f"cite map {tag}".strip(),
                     f"{os.path.basename(tp)} ({len(out)} frames of {os.path.basename(os.path.normpath(old_dir))}"
                     f" -> this package; median score {med:.3f}; {low} below 0.8 = unmapped, "
                     f"mostly dark frames)")
    print(f"cite map: {len(out)} old frames, median score {med:.3f}, {low} below 0.8"
          f" ({'time-windowed' if timed else 'time-free'}) -> {tp}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--tag", default="", help="suffix for cite_map_<tag>.tsv and the MANIFEST key")
    a = ap.parse_args(argv)
    return run(a.old_dir, a.new_dir, a.tag)


if __name__ == "__main__":
    sys.exit(main())
