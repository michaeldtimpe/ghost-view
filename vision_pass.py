#!/usr/bin/env python3
"""Vision pre-pass: let a local VLM look at the frames before Claude does.

Groups a finished folder's frames into shots (a hard cut = a big pixel jump
between consecutive kept frames), captions one frame per shot (plus one every
--every seconds inside a long take) through an OpenAI-style chat endpoint
(oMLX on :8000 by default), and writes:

  shots.tsv     shot · start · end · frames · setting · people · scale · action · light ·
                text (Apple Vision OCR) · text_vlm (the captioner's reading: a guess)
  INDEX.tsv     gains a `shot` column (S0001…), written atomically
  sync.md       regenerated; each shot's first row carries its caption
  MANIFEST.txt  gains a `vision:` line (model, shots, captions, seconds)

Resumable: captions land in _vision.jsonl as they arrive and a rerun skips
the ones it has. The API key comes from $OMLX_API_KEY (never a flag).

usage: vision_pass.py <crv_out_dir> [--model M] [--base URL] [--cut 40]
                      [--every 20] [--limit N] [--dry-run]
"""
from __future__ import annotations
import argparse, base64, json, os, re, sys, time, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import enhance_process as E

PROMPT = ("You are annotating one frame of a feature film for a reader who cannot see it. "
          "Reply with compact JSON only, keys: setting (<=8 words), people (integer), "
          "scale (one of: extreme-wide, wide, medium, close-up, extreme-close-up, insert), "
          "action (<=12 words, what is happening), "
          "light (one of: dark, low-key, neutral, bright, high-key), "
          "text (readable on-screen text, else \"\"). "
          # the dark-shot rule: without it backlit fog and blue light read as
          # water (Alien's planet surface: "underwater", "divers in vintage
          # suits", "a submarine"); with it 5 of 6 such shots on Alien came
          # back as dark foggy interiors (2026-11-02)
          "This is a frame of a live-action film shot on sets and locations. Darkness, fog, haze, "
          "grain or blue light are not water: say underwater only if water is clearly visible. Name "
          "the place as the story shows it (a ship corridor, a planet surface, a kitchen), not as a set.")
KEYS = ("setting", "people", "scale", "action", "light", "text")


def hms_to_sec(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def _sig(path, size=16):
    from PIL import Image
    import numpy as np
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size)), dtype=np.int16)


def _pct(a, b, tol=25):
    import numpy as np
    return 100.0 * float((np.abs(a - b).max(axis=2) > tol).mean())


def group_shots(rows, frames_dir, cut=40.0, every=20.0):
    """Group kept frames into shots. With the engine's `cut` column (format 5)
    a shot starts wherever the motion pass marked a hard cut. Without it
    (older folders) consecutive frames whose pixel jump exceeds `cut`% start
    one -- cruder: in-shot motion also trips it, so long takes over-split.
    Returns [{"id", "start", "end", "rows": [row indices], "caption_rows": [...]}].
    caption_rows: the shot's first frame, then one every `every` seconds."""
    have_cut = rows and "cut" in rows[0] and any(r.get("cut") for r in rows)
    shots, prev = [], None
    for i, r in enumerate(rows):
        t = hms_to_sec(r["time"])
        if have_cut:
            new = prev is None or r.get("cut") == "cut"
            prev = True
        else:
            sig = _sig(os.path.join(frames_dir, r["frame"]))
            new = prev is None or _pct(sig, prev) > cut
            prev = sig
        if new:
            shots.append({"id": f"S{len(shots) + 1:04d}", "start": t, "end": t,
                          "rows": [i], "caption_rows": [i], "_last_cap": t})
        else:
            sh = shots[-1]
            sh["rows"].append(i); sh["end"] = t
            if t - sh["_last_cap"] >= every:
                sh["caption_rows"].append(i); sh["_last_cap"] = t
    for sh in shots:
        del sh["_last_cap"]
    return shots


class Captioner:
    def __init__(self, base, model, key, timeout=600):
        self.base, self.model, self.key, self.timeout = base.rstrip("/"), model, key, timeout

    def _post(self, path, body):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                   headers={"Authorization": f"Bearer {self.key}",
                                            "Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def caption(self, jpg_path):
        """-> (dict with KEYS, seconds). Raises on transport failure; a reply
        that is not JSON is retried once and then returned as {"action": raw}."""
        b64 = base64.b64encode(open(jpg_path, "rb").read()).decode()
        body = {"model": self.model, "temperature": 0, "max_tokens": 400,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": PROMPT}]}]}
        t0 = time.perf_counter()
        for attempt in (1, 2):
            out = self._post("/chat/completions", body)
            msg = out["choices"][0]["message"]
            raw = (msg.get("content") or "").strip()
            d = _parse_json(raw)
            if d is not None:
                return {k: _clean(k, d.get(k, "")) for k in KEYS}, time.perf_counter() - t0
        return {"setting": "", "people": "", "scale": "", "light": "", "text": "",
                "action": re.sub(r"\s+", " ", raw)[:120]}, time.perf_counter() - t0


def _parse_json(raw):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _clean(k, v):
    if k == "people":
        try: return str(int(v))
        except (TypeError, ValueError): return ""
    return E.tsv_safe(v)


def read_index(idx):
    lines = open(idx, encoding="utf-8").read().splitlines()
    header = lines[0].split("\t")
    rows = []
    for l in lines[1:]:
        parts = l.split("\t")
        parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return header, rows


def write_index(idx, header, rows):
    tmp = idx + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r.get(h, "") for h in header) + "\n")
    os.replace(tmp, idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=os.environ.get("CRV_VLM_MODEL", "Qwen3.6-35B-A3B-4bit"))
    ap.add_argument("--base", default=os.environ.get("CRV_VLM_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--cut", type=float, default=40.0, help="%% of pixels changed that starts a new shot")
    ap.add_argument("--every", type=float, default=20.0, help="extra caption every N s inside a shot")
    ap.add_argument("--limit", type=int, default=0, help="caption at most N frames (a probe)")
    ap.add_argument("--dry-run", action="store_true", help="group shots and report, no captions")
    a = ap.parse_args()

    idx = os.path.join(a.out_dir, "INDEX.tsv")
    frames_dir = os.path.join(a.out_dir, "frames")
    if not os.path.exists(idx) or not os.path.isdir(frames_dir):
        print(f"✗ {a.out_dir}: no INDEX.tsv/frames", file=sys.stderr); sys.exit(1)
    key = os.environ.get("OMLX_API_KEY", "")
    if not a.dry_run and not key:
        print("✗ OMLX_API_KEY not set (export it, or source the file named by CRV_ENV_FILE)", file=sys.stderr); sys.exit(1)

    header, rows = read_index(idx)
    t0 = time.perf_counter()
    shots = group_shots(rows, frames_dir, a.cut, a.every)
    n_cap = sum(len(s["caption_rows"]) for s in shots)
    how = "engine cut column" if any(r.get("cut") for r in rows) else "pixel jumps (no cut column)"
    print(f"• {len(rows)} frames -> {len(shots)} shots by {how}, {n_cap} frames to caption "
          f"({time.perf_counter() - t0:.1f}s grouping)", flush=True)
    if a.dry_run:
        lens = sorted(len(s["rows"]) for s in shots)
        print(f"  frames per shot: median {lens[len(lens)//2]}, max {lens[-1]}; "
              f"shots of 1 frame: {sum(1 for n in lens if n == 1)}")
        return

    # resume from the jsonl
    jl = os.path.join(a.out_dir, "_vision.jsonl")
    done = {}
    if os.path.exists(jl):
        for l in open(jl, encoding="utf-8"):
            try:
                d = json.loads(l); done[d["frame"]] = d
            except ValueError:
                pass            # a torn trailing line from a crash
    cap = Captioner(a.base, a.model, key)
    todo = [(s, i) for s in shots for i in s["caption_rows"] if rows[i]["frame"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"• captioning {len(todo)} frames with {a.model} ({len(done)} already done)...", flush=True)
    secs, k = 0.0, 0
    with open(jl, "a", encoding="utf-8") as f:
        for s, i in todo:
            frame = rows[i]["frame"]
            try:
                d, dt = cap.caption(os.path.join(frames_dir, frame))
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                print(f"\n✗ {frame}: {type(e).__name__}: {str(e)[:160]} -- stopping; rerun to resume",
                      file=sys.stderr)
                break
            d.update({"frame": frame, "shot": s["id"], "model": a.model})
            f.write(json.dumps(d, ensure_ascii=False) + "\n"); f.flush()
            done[frame] = d; secs += dt; k += 1
            if k in (1, 5) or k % 50 == 0:
                print(f"  {k}/{len(todo)}  {secs / k:.1f}s/frame  eta {(len(todo) - k) * secs / k / 60:.0f} min",
                      flush=True)
    if not done:
        print("✗ nothing captioned", file=sys.stderr); sys.exit(1)

    # On-screen text: the VLM guesses words ("Lufkin Air School" for a letter
    # that reads "Luther Art School"); Apple Vision reads letters. OCR every
    # captioned frame and let a confident Vision reading replace the field.
    binp = E._vision_ocr_bin()
    if binp:
        t1 = time.perf_counter()
        paths = [os.path.join(frames_dir, fr) for fr in done]
        ocr = E.vision_ocr(binp, paths)
        n_ocr = 0
        for fr, d in done.items():
            txt = E.tsv_safe(ocr.get(os.path.splitext(fr)[0], ""))
            d["text_vlm"] = d.get("text_vlm", d.get("text", ""))
            d["text"] = txt if len(re.sub(r"[^A-Za-z0-9]", "", txt)) >= 3 else ""
            n_ocr += bool(d["text"])
        print(f"• Vision OCR read text on {n_ocr}/{len(done)} captioned frames "
              f"({time.perf_counter() - t1:.0f}s)", flush=True)
    else:
        n_ocr = -1
        for d in done.values():
            d["text_vlm"] = d.get("text_vlm", d.get("text", "")); d["text"] = ""

    # shots.tsv: the shot's first caption; later captions inside a long take
    # are appended to action with their time
    sp = os.path.join(a.out_dir, "shots.tsv")
    with open(sp + ".tmp", "w", encoding="utf-8") as f:
        f.write("shot\tstart\tend\tframes\tsetting\tpeople\tscale\taction\tlight\ttext\ttext_vlm\n")
        for s in shots:
            caps = [done[rows[i]["frame"]] for i in s["caption_rows"] if rows[i]["frame"] in done]
            if not caps:
                continue
            c0 = caps[0]
            action = c0["action"]
            for i, c in zip(s["caption_rows"][1:], caps[1:]):
                if c["action"] and c["action"] != c0["action"]:
                    action += f" | {rows[i]['time'][:8]}: {c['action']}"
            f.write("\t".join([s["id"], E.hms(s["start"])[:8], E.hms(s["end"])[:8], str(len(s["rows"])),
                               c0["setting"], c0["people"], c0["scale"], action, c0["light"],
                               c0.get("text", ""), c0.get("text_vlm", "")]) + "\n")
    os.replace(sp + ".tmp", sp)

    # INDEX: shot column (appended, so columns 0-13 keep their positions)
    if "shot" not in header:
        header.append("shot")
    for s in shots:
        for i in s["rows"]:
            rows[i]["shot"] = s["id"]
    write_index(idx, header, rows)

    # sync.md with the caption on each shot's first row
    first = {rows[s["rows"][0]]["frame"]: s for s in shots}
    for r in rows:
        s = first.get(r["frame"])
        c = done.get(r["frame"]) if s else None
        if c:
            r["caption"] = E.shot_caption(c)
    title = os.path.basename(os.path.normpath(a.out_dir))
    man = os.path.join(a.out_dir, "MANIFEST.txt")
    txt = open(man, encoding="utf-8").read() if os.path.exists(man) else ""
    m = re.search(r"(?m)^source: (.*)$", txt)
    if m:
        title = os.path.basename(m.group(1).strip())
    E.write_sync_md(os.path.join(a.out_dir, "sync.md"), title, rows)

    line = (f"vision: {a.model} via {a.base}; {len(shots)} shots in shots.tsv, "
            f"{len(done)} frames captioned, {secs:.0f}s this run; INDEX `shot` column; "
            + (f"on-screen text by Apple Vision OCR on {n_ocr} frames" if n_ocr >= 0
               else "on-screen text by the VLM (no Vision OCR helper)"))
    if txt:
        txt = re.sub(r"(?m)^vision: .*\n", "", txt)
        if not txt.endswith("\n"):
            txt += "\n"
        open(man, "w", encoding="utf-8").write(txt + line + "\n")
    print(f"\n✓ {title}: {len(shots)} shots, {len(done)} captions ({k} new, {secs:.0f}s)")


if __name__ == "__main__":
    main()
