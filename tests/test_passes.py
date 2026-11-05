"""ocr_pass, voice_gate, cite_map, origin_patch on a tiny synthetic package."""
import importlib.util, os, shutil
import numpy as np
from PIL import Image
import enhance_process as E
import pkg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ocr_pass, voice_gate, cite_map, origin_patch = (
    _load(n) for n in ("ocr_pass", "voice_gate", "cite_map", "origin_patch"))

SRT = ("1\n00:00:00,000 --> 00:00:05,000\nHello.\n\n"
       "2\n00:00:19,000 --> 00:00:22,000\nStill here.\n")


def _pkg(d, n=5, times=None, index=True, srt=False, seed=0):
    """n frames of distinct random noise; INDEX rows 10 s apart by default."""
    os.makedirs(d / "frames")
    rng = np.random.default_rng(seed)
    names = []
    for i in range(1, n + 1):
        name = f"frame_{i:05d}.jpg"
        img = rng.integers(0, 256, (56, 96, 3), dtype=np.uint8)
        Image.fromarray(img).resize((192, 112), Image.NEAREST).save(d / "frames" / name, quality=95)
        names.append(name)
    times = times or [10.0 * i for i in range(n)]
    if index:
        lines = ["frame\ttime\tdialogue\tvoice"]
        lines += [f"{nm}\t{E.hms(t)}\t\tcentre" for nm, t in zip(names, times)]
        (d / "INDEX.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "MANIFEST.txt").write_text("source: originals/test-film.mkv\nvision: x\nengine: crv test\n",
                                    encoding="utf-8")
    if srt:
        (d / "dialogue.srt").write_text(SRT, encoding="utf-8")
    return names


# ----------------------------------------------------------------- voice_gate
def test_voice_gate_clears_only_cueless_rows_and_is_idempotent(tmp_path):
    _pkg(tmp_path, srt=True)                 # rows at 0,10,20,30,40 s; cues 0-5 and 19-22
    assert voice_gate.run(str(tmp_path)) == 0
    _, rows = pkg.read_index(str(tmp_path))
    assert [r["voice"] for r in rows] == ["centre", "", "centre", "", ""]
    assert pkg.manifest_get(str(tmp_path), "voice") == \
        "gated on the cue track (3 of 5 placements cleared, no cue live)"
    assert (tmp_path / "sync.md").exists()
    before = (tmp_path / "INDEX.tsv").read_text(encoding="utf-8")
    voice_gate.run(str(tmp_path))
    assert (tmp_path / "INDEX.tsv").read_text(encoding="utf-8") == before
    # the first run's count stays on record; a no-op rerun does not overwrite it
    assert "(3 of 5 placements cleared" in pkg.manifest_get(str(tmp_path), "voice")


def test_voice_gate_without_cue_track_changes_nothing(tmp_path, capsys):
    _pkg(tmp_path)
    snap = {p: (tmp_path / p).read_bytes() for p in ("INDEX.tsv", "MANIFEST.txt")}
    assert voice_gate.run(str(tmp_path)) == 0
    assert "no cue track" in capsys.readouterr().out
    assert {p: (tmp_path / p).read_bytes() for p in snap} == snap
    assert not (tmp_path / "sync.md").exists()


# ------------------------------------------------------------------- cite_map
def _old_from_new(new, old, order, index):
    """Old package = some new frames under other numbers (old frame k = new frame order[k])."""
    os.makedirs(old / "frames")
    _, rows = pkg.read_index(str(new))
    lines = ["frame\ttime\tdialogue"]
    for k, src in enumerate(order, 1):
        shutil.copy(new / "frames" / src, old / "frames" / f"frame_{k:04d}.jpg")
        t = next(r["time"] for r in rows if r["frame"] == src)
        lines.append(f"frame_{k:04d}.jpg\t{t}\t")
    if index:
        (old / "INDEX.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (old / "MANIFEST.txt").write_text("source: x\n", encoding="utf-8")


def _cite(new):
    lines = (new / "cite_map.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "old_frame\told_time\tnew_frame\tnew_time\tscore\tdt_s"
    return [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:]]


def test_cite_map_time_free(tmp_path):
    new, old = tmp_path / "new-crv", tmp_path / "old-crv"
    names = _pkg(new, n=6)
    order = [names[4], names[1], names[5], names[0]]
    _old_from_new(new, old, order, index=False)
    assert cite_map.run(str(old), str(new)) == 0
    got = _cite(new)
    assert [g["new_frame"] for g in got] == order
    assert all(g["old_time"] == "" and g["dt_s"] == "" for g in got)
    assert all(float(g["score"]) > 0.99 for g in got)
    assert got[0]["new_time"] == "00:00:40.000"
    assert pkg.manifest_get(str(new), "cite map") == \
        "cite_map.tsv (4 frames of old-crv -> this package; median score 1.000; 0 below 0.8 = unmapped, mostly dark frames)"
    first = (new / "cite_map.tsv").read_bytes()
    cite_map.run(str(old), str(new))
    assert (new / "cite_map.tsv").read_bytes() == first


def test_cite_map_time_windowed(tmp_path):
    new, old = tmp_path / "new-crv", tmp_path / "old-crv"
    names = _pkg(new, n=6)
    order = [names[3], names[0], names[2]]
    _old_from_new(new, old, order, index=True)
    cite_map.run(str(old), str(new))
    got = _cite(new)
    assert [g["new_frame"] for g in got] == order
    assert [g["dt_s"] for g in got] == ["0.0"] * 3
    assert got[0]["old_time"] == "00:00:30.000"


def test_cite_map_window_excludes_far_frames(tmp_path):
    """Two new frames show the same image 50 s apart; the timed match picks the near one."""
    new, old = tmp_path / "new-crv", tmp_path / "old-crv"
    names = _pkg(new, n=6)
    shutil.copy(new / "frames" / names[0], new / "frames" / names[5])    # 0 s and 50 s
    _old_from_new(new, old, [names[5]], index=True)                      # old frame at 50 s
    cite_map.run(str(old), str(new))
    assert _cite(new)[0]["new_frame"] == names[5]
    (old / "INDEX.tsv").unlink()
    cite_map.run(str(old), str(new))
    assert _cite(new)[0]["new_frame"] == names[0]   # time-free: first best by sort order


def test_cite_map_letterbox_cropped_in_one_package(tmp_path):
    """New frames carry black bars, old ones were cropped to the picture."""
    new, old = tmp_path / "new-crv", tmp_path / "old-crv"
    names = _pkg(new, n=4)
    for n in names:                                   # 192x112 picture -> 192x152 with bars
        p = new / "frames" / n
        im = Image.open(p); boxed = Image.new("RGB", (192, 152)); boxed.paste(im, (0, 20))
        boxed.save(p, quality=95)
    os.makedirs(old / "frames")
    for k, src in enumerate([names[2], names[0]], 1):
        Image.open(new / "frames" / src).crop((0, 20, 192, 132)).save(old / "frames" / f"frame_{k}.jpg")
    cite_map.run(str(old), str(new))
    got = _cite(new)
    assert [g["new_frame"] for g in got] == [names[2], names[0]]
    assert all(float(g["score"]) > 0.95 for g in got)


# --------------------------------------------------------------- origin_patch
def test_origin_patch_writes_and_replaces(tmp_path):
    _pkg(tmp_path)
    origin_patch.run(str(tmp_path), "nas:/media/films/test-film.mkv", "01:56:49", 7488792892)
    m = (tmp_path / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    assert m[:3] == ["source: originals/test-film.mkv",
                     "origin: nas:/media/films/test-film.mkv",
                     "container: 01:56:49 · 7488792892 bytes · test-film.mkv"]
    origin_patch.run(str(tmp_path), "nas:/media/other.mkv", "01:00:00", 5, "renamed.mkv")
    txt = (tmp_path / "MANIFEST.txt").read_text(encoding="utf-8")
    assert txt.count("origin: ") == 1 and txt.count("container: ") == 1
    assert "origin: nas:/media/other.mkv\ncontainer: 01:00:00 · 5 bytes · renamed.mkv\n" in txt
    assert not (tmp_path / "sync.md").exists()


def test_origin_patch_origin_only(tmp_path):
    _pkg(tmp_path)
    origin_patch.main([str(tmp_path), "--origin", "backup:/x/y.mkv"])
    assert pkg.manifest_get(str(tmp_path), "origin") == "backup:/x/y.mkv"
    assert pkg.manifest_get(str(tmp_path), "container") == ""


# ------------------------------------------------------------------- ocr_pass
def test_ocr_pass_filters_and_writes(tmp_path, monkeypatch, capsys):
    names = _pkg(tmp_path, n=4)
    seen = {}

    def fake(binp, paths, batch=200, workers=None):
        seen["paths"] = paths
        return {"frame_00001": "SPECIAL ORDER 937", "frame_00002": "a.b",
                "frame_00004": 'CREW\tEXPENDABLE "now"'}
    monkeypatch.setattr(E, "_vision_ocr_bin", lambda: "x")
    monkeypatch.setattr(E, "vision_ocr", fake)
    assert ocr_pass.run(str(tmp_path), workers=2) == 0
    assert seen["paths"] == [os.path.join(str(tmp_path), "frames", n) for n in names]
    assert (tmp_path / "text.tsv").read_text(encoding="utf-8").splitlines() == [
        "frame\ttime\ttext",
        "frame_00001.jpg\t00:00:00.000\tSPECIAL ORDER 937",
        "frame_00004.jpg\t00:00:30.000\tCREW EXPENDABLE ”now”"]
    header, rows = pkg.read_index(str(tmp_path))
    assert "text" in header
    assert [r["text"] for r in rows] == ["SPECIAL ORDER 937", "", "", "CREW EXPENDABLE ”now”"]
    m = (tmp_path / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    assert m[2].startswith("ocr: Apple Vision on 4 frames, 2 with text (")
    assert m[1].startswith("vision: ")
    assert "text=SPECIAL ORDER 937" in (tmp_path / "sync.md").read_text(encoding="utf-8")

    # second run without --force: no work
    monkeypatch.setattr(E, "vision_ocr", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert ocr_pass.main([str(tmp_path)]) == 0
    assert "already" in capsys.readouterr().out
    # --force reruns and stays the same shape
    monkeypatch.setattr(E, "vision_ocr", fake)
    ocr_pass.main([str(tmp_path), "--force"])
    assert pkg.read_index(str(tmp_path))[0].count("text") == 1
    assert (tmp_path / "MANIFEST.txt").read_text(encoding="utf-8").count("ocr: ") == 1


def test_ocr_pass_without_binary_exits_1(tmp_path, monkeypatch):
    _pkg(tmp_path, n=2)
    monkeypatch.setattr(E, "_vision_ocr_bin", lambda: None)
    assert ocr_pass.run(str(tmp_path)) == 1
    assert not (tmp_path / "text.tsv").exists()
