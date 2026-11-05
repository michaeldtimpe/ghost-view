#!/usr/bin/env python3
"""Find the hard cuts the engine missed inside long dark shots, from the kept
frames alone (no source needed), and mark them in INDEX's `cut` column.

    dark_cuts.py <out_dir> [--min-shot 60] [--max-lum 64] [--thr auto] [--dry-run]

The engine's cut detector reads the mean |gray diff| of the decoded stream;
between two near-black set-ups that is 3-8 where a lit cut is 20-250, so dark
set-pieces came out as single shots (Alien: the opening tour 423 s, the egg
chamber 212 s, Brett's death 184 s). The engine now scales that diff by
luminance (enhance_process.cut_mask); this pass is the same idea for packages
already on disk. For every pair of consecutive kept frames inside a shot
longer than --min-shot whose frames are darker than --max-lum (mean gray of
255), both frames are contrast-normalised (z-scored), the second is aligned to
the first by phase correlation (a pan is not a cut) and the mean |difference|
is the score. The threshold is calibrated on the film itself: the midpoint
between the score the engine's own cuts reach and the score ordinary in-shot
pairs reach (--thr overrides). A pair over the threshold gets `cut` on its
second frame; `cuts_min` is recounted. Run vision_pass.py afterwards: it
regroups shots from the `cut` column and captions the new shots' first frames.
"""
import argparse, os, sys, time
import numpy as np
import enhance_process as E
import pkg

W, H = 128, 72


def load_gray(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((W, H), Image.BILINEAR), dtype=np.float32)


def znorm(a):
    return (a - a.mean()) / (a.std() + 1e-6)


def aligned_diff(a, b):
    """mean |za - zb| after shifting b onto a by phase correlation (wrap-around;
    a shift large enough for the wrap to matter is a cut anyway)."""
    za, zb = znorm(a), znorm(b)
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    A, B = np.fft.rfft2(za * win), np.fft.rfft2(zb * win)
    R = A * np.conj(B); R /= np.abs(R) + 1e-8
    c = np.fft.irfft2(R, s=(H, W))
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    if iy > H / 2: iy -= H
    if ix > W / 2: ix -= W
    zb = np.roll(zb, (iy, ix), axis=(0, 1))
    return float(np.abs(za - zb).mean())


def shots_of(rows):
    """[(first_index, last_index)] by the `cut` column."""
    out = []; start = 0
    for i, r in enumerate(rows):
        if i and r.get("cut"):
            out.append((start, i - 1)); start = i
    out.append((start, len(rows) - 1))
    return out


def shot_len(rows, a, b):
    t0 = pkg.sec(rows[a]["time"]); t1 = pkg.sec(rows[b]["time"])
    try:
        t1 += float(rows[b].get("held_s") or 0)
    except ValueError:
        pass
    return t1 - t0


def run(out_dir, min_shot=60.0, max_lum=64.0, thr=None, dry_run=False, log=print):
    header, rows = pkg.read_index(out_dir)
    if "cut" not in header:
        log("✗ no `cut` column (pre-format-5 package)"); return 0
    fdir = pkg.frames_dir(out_dir)
    t0 = time.perf_counter()
    cache = {}
    def gray(i):
        if i not in cache:
            cache[i] = load_gray(os.path.join(fdir, rows[i]["frame"]))
        return cache[i]

    # calibration: what score do the engine's own cuts reach, and ordinary
    # consecutive pairs inside shots? Sampled across the film, lit or dark.
    cut_i = [i for i in range(1, len(rows)) if rows[i].get("cut")]
    step = max(1, len(cut_i) // 150)
    s_cut = [aligned_diff(gray(i - 1), gray(i)) for i in cut_i[::step]]
    non_i = [i for i in range(1, len(rows)) if not rows[i].get("cut")]
    step = max(1, len(non_i) // 300)
    s_non = [aligned_diff(gray(i - 1), gray(i)) for i in non_i[::step]]
    cache.clear()
    if thr is None:
        if len(s_cut) < 10:
            log(f"✗ only {len(s_cut)} engine cuts to calibrate on; pass --thr"); return 0
        lo, hi = float(np.percentile(s_non, 90)), float(np.percentile(s_cut, 25))
        thr = 0.5 * (lo + hi) if hi > lo else hi
    log(f"• calibration: in-shot pairs median {np.median(s_non):.2f} (p90 {np.percentile(s_non, 90):.2f}), "
        f"engine cuts median {np.median(s_cut):.2f} (p25 {np.percentile(s_cut, 25):.2f}) -> thr {thr:.2f}")

    added, in_shots, scanned = [], 0, 0
    for a, b in shots_of(rows):
        if b <= a or shot_len(rows, a, b) < min_shot:
            continue
        hit = False
        for i in range(a + 1, b + 1):
            ga, gb = gray(i - 1), gray(i)
            if 0.5 * (ga.mean() + gb.mean()) >= max_lum:
                continue
            scanned += 1
            s = aligned_diff(ga, gb)
            if s >= thr:
                added.append((i, s)); hit = True
            cache.pop(i - 1, None)
        cache.clear()
        in_shots += hit
    secs = time.perf_counter() - t0
    log(f"• {len(added)} cuts added in {in_shots} shots over {min_shot:g} s "
        f"({scanned} dark pairs scored, {secs:.0f}s)")
    if dry_run:
        for i, s in added[:40]:
            log(f"    {rows[i]['time'][:8]}  {rows[i]['frame']}  {s:.2f}")
        return len(added)
    for i, _ in added:
        rows[i]["cut"] = "cut"
    if "cuts_min" in header:
        ts = np.array([pkg.sec(r["time"]) for r in rows])
        ct = np.array([pkg.sec(rows[i]["time"]) for i in range(1, len(rows)) if rows[i].get("cut")])
        for r, t in zip(rows, ts):
            r["cuts_min"] = str(int(((ct >= t - 15) & (ct <= t + 15)).sum()) * 2)
    pkg.write_index(out_dir, header, rows)
    pkg.manifest_set(out_dir, "dark cuts",
                     f"{len(added)} added in {in_shots} shots over {min_shot:g} s (contrast-normalised, "
                     f"motion-aligned diff between kept frames darker than {max_lum:g}/255; thr {thr:.2f}, "
                     f"calibrated on the engine's cuts); run vision_pass.py to regroup shots",
                     after="motion")
    pkg.rebuild_sync(out_dir)
    return len(added)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--min-shot", type=float, default=60.0)
    ap.add_argument("--max-lum", type=float, default=E.CUT_LUM_REF)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run(a.out_dir, a.min_shot, a.max_lum, a.thr, a.dry_run)


if __name__ == "__main__":
    main()
