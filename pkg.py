#!/usr/bin/env python3
"""The package on disk, for every pass that edits a finished crv folder in
place (vision_pass, sound_tags, redialogue, ocr_pass, dark_cuts, sdh_sounds,
voice_gate, cite_map, origin_patch): read/write INDEX.tsv atomically, read a
MANIFEST line or set one, rebuild sync.md from the tables. Nothing here
decodes media."""
import os, re
import enhance_process as E


def read_index(out_dir):
    """-> (header, rows as dicts). Short rows are padded so every key exists."""
    idx = os.path.join(out_dir, "INDEX.tsv")
    lines = open(idx, encoding="utf-8").read().splitlines()
    header = lines[0].split("\t")
    rows = []
    for l in lines[1:]:
        if not l.strip():
            continue
        parts = l.split("\t")
        parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return header, rows


def write_index(out_dir, header, rows):
    """tmp + os.replace: a crash mid-write leaves the old INDEX, not half of one."""
    idx = os.path.join(out_dir, "INDEX.tsv")
    tmp = idx + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(E.tsv_safe(r.get(k, "")) for k in header) + "\n")
    os.replace(tmp, idx)


def add_column(header, name):
    """Append a column name (columns 0-13 keep their positions)."""
    if name not in header:
        header.append(name)
    return header


def manifest_path(out_dir):
    return os.path.join(out_dir, "MANIFEST.txt")


def manifest_text(out_dir):
    p = manifest_path(out_dir)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def manifest_get(out_dir, key):
    m = re.search(rf"(?m)^{re.escape(key)}: (.*)$", manifest_text(out_dir))
    return m.group(1).strip() if m else ""


def manifest_set(out_dir, key, value, after=None):
    """Replace the `key:` line, or add one (after the `after:` line when that
    exists, else before `engine:`, else at the end). Atomic."""
    txt = manifest_text(out_dir)
    line = f"{key}: {value}"
    if re.search(rf"(?m)^{re.escape(key)}: ", txt):
        txt = re.sub(rf"(?m)^{re.escape(key)}: .*$", lambda _: line, txt, count=1)
    else:
        lines = txt.splitlines()
        at = None
        if after:
            at = next((i + 1 for i, l in enumerate(lines) if l.startswith(f"{after}: ")), None)
        if at is None:
            at = next((i for i, l in enumerate(lines) if l.startswith("engine: ")), len(lines))
        lines.insert(at, line)
        txt = "\n".join(lines) + "\n"
    p = manifest_path(out_dir)
    tmp = p + ".tmp"
    open(tmp, "w", encoding="utf-8").write(txt)
    os.replace(tmp, p)


def title(out_dir):
    """What sync.md's heading calls the film: the MANIFEST source's basename,
    else the folder name."""
    src = manifest_get(out_dir, "source")
    return os.path.basename(src) if src else os.path.basename(os.path.normpath(out_dir))


def rebuild_sync(out_dir, rows=None):
    """sync.md from INDEX.tsv (+ shots.tsv captions and text.tsv OCR, which
    write_sync_md reads itself)."""
    if rows is None:
        _, rows = read_index(out_dir)
    return E.write_sync_md(os.path.join(out_dir, "sync.md"), title(out_dir), rows)


def cues(out_dir):
    """The cue track as [(start, end, text)], or [] without one."""
    p = os.path.join(out_dir, "dialogue.srt")
    return E.parse_srt(p) if os.path.exists(p) else []


def sec(hms):
    """'01:02:03.456' -> seconds."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def frames_dir(out_dir):
    return os.path.join(out_dir, "frames")
