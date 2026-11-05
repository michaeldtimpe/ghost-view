"""Media tests for the 2026-09 review fixes: synthetic clips through real
ffmpeg (motion grammar, HDR tonemapping, truncated sources, VobSub OCR) plus
the Vision helper's reading order and the manual PGS -> SRT scripts.

Slow, like test_e2e.py. Vision-dependent tests skip off macOS / without swiftc.
"""
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import enhance_process as E

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"),
]
needs_vision = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None,
    reason="Apple Vision OCR needs macOS + swiftc")


def ff(*args):
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def run_engine(*args):
    return subprocess.run([sys.executable, str(REPO / "enhance_process.py"), *args],
                          capture_output=True, text=True)


# ---------------------------------------------------------------- motion
@pytest.fixture(scope="module")
def texture(tmp_path_factory):
    """A still, richly textured frame to move a virtual camera over -- unlike
    testsrc2, nothing inside it moves on its own."""
    d = tmp_path_factory.mktemp("motion")
    ff("-f", "lavfi", "-i", "gradients=s=1920x1080:n=5:speed=0,noise=alls=60:allf=u",
       "-frames:v", "1", "-update", "1", str(d / "tex.png"))
    return d


def _camera_clip(texture, name, vf):
    out = texture / f"{name}.mp4"
    if not out.exists():
        ff("-loop", "1", "-framerate", "24", "-i", str(texture / "tex.png"), "-t", "6",
           "-vf", f"{vf},format=yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(out))
    return str(out)


def _zoom(z):
    return (f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s=640x360:fps=24")


@pytest.mark.parametrize("name,vf,want", [
    ("still", "crop=640:360:x=200:y=300", "static"),
    ("pan", "crop=640:360:x='200+t*20':y=300", "pan-R"),
    ("tilt", "crop=640:360:x=200:y='300+t*40'", "tilt-D"),
    # ~0.74%/s: a slow push, well under the old whole-pixel resolution
    ("zoom_in", _zoom("1.3+0.0004*on"), "push-in"),
    ("zoom_out", _zoom("1.5-0.0004*on"), "pull-out"),
    # a figure walking through a locked-off frame: subject motion, not a zoom
    ("walker", "crop=640:360:x=200:y=300,drawbox=x='40+t*80':y=120:w=70:h=140:color=white:t=fill", "static"),
])
def test_camera_grammar(texture, name, vf, want):
    ser = E._motion_series(_camera_clip(texture, name, vf))
    labels = Counter(v["motion"] for v in
                     E._motion_beats(ser, [("f", t) for t in (1.5, 2.5, 3.5, 4.5)]).values())
    assert labels.most_common(1)[0][0] == want, labels
    assert labels[want] >= 3, labels


# ---------------------------------------------------------------- HDR
NITS = [0, 1, 10, 100, 203, 1000]


def _pq_patch_clip(d):
    """Six grey patches at known luminances, PQ-encoded 10-bit BT.2020, tagged.
    Written as a 16-bit PPM so no OpenCV is needed; ffv1 keeps the tags."""
    def pq(n):
        y = (n / 10000.0) ** E._M1
        return int(round(((E._C1 + E._C2 * y) / (1 + E._C3 * y)) ** E._M2 * 65535))
    img = np.zeros((360, 1920, 3), ">u2")
    for i, n in enumerate(NITS):
        img[:, i * 320:(i + 1) * 320] = pq(n)
    ppm = d / "patch.ppm"
    ppm.write_bytes(b"P6 1920 360 65535\n" + img.tobytes())
    out = d / "pq.mkv"
    ff("-loop", "1", "-framerate", "24", "-i", str(ppm), "-t", "2", "-vf",
       "scale=out_color_matrix=bt2020:out_range=tv,format=yuv420p10le,"
       "setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc:range=tv",
       "-c:v", "ffv1", str(out))
    return out


def test_hdr_source_is_detected_and_tonemapped(tmp_path):
    from PIL import Image
    clip = _pq_patch_clip(tmp_path)
    c = E.video_color(str(clip))
    assert c["hdr"] == "pq" and c["width"] == 1920

    r = run_engine(str(clip), "-o", str(tmp_path / "out"), "--no-audio", "--no-motion")
    assert r.returncode == 0, r.stderr
    img = np.asarray(Image.open(tmp_path / "out" / "frames" / "frame_00001.jpg").convert("RGB"))
    w = img.shape[1] // 6
    got = [int(img[img.shape[0] // 2, i * w + w // 2].mean()) for i in range(6)]
    assert got[0] <= 3
    assert abs(got[3] - 186) <= 4 and abs(got[4] - 241) <= 4 and got[5] >= 250, got
    assert got == sorted(got)       # untonemapped, 203-nit white sat near 148
    man = (tmp_path / "out" / "MANIFEST.txt").read_text()
    assert "color: HDR (smpte2084/bt2020) tonemapped" in man


def test_sdr_source_has_no_color_note(tmp_path):
    clip = tmp_path / "sdr.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=3", "-c:v", "libx264",
       "-preset", "ultrafast", str(clip))
    assert E.video_color(str(clip))["hdr"] is None
    r = run_engine(str(clip), "-o", str(tmp_path / "out"), "--no-audio", "--no-motion")
    assert r.returncode == 0, r.stderr
    assert "color:" not in (tmp_path / "out" / "MANIFEST.txt").read_text()


# ---------------------------------------------------------------- truncated source
def test_truncated_source_fails_instead_of_shipping_partial(tmp_path):
    full = tmp_path / "full.mkv"
    ff("-f", "lavfi", "-i", "testsrc2=s=160x90:r=24:d=150", "-c:v", "libx264",
       "-preset", "ultrafast", "-g", "24", str(full))
    cut = tmp_path / "cut.mkv"
    data = full.read_bytes()
    cut.write_bytes(data[: len(data) // 3])      # header still claims 150s
    r = run_engine(str(cut), "-o", str(tmp_path / "out"), "--no-audio", "--no-motion")
    assert r.returncode == 1
    assert "ended early" in r.stderr
    assert not (tmp_path / "out" / "INDEX.tsv").exists()
    # and the intact file still passes the same check
    r = run_engine(str(full), "-o", str(tmp_path / "ok"), "--no-audio", "--no-motion")
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- VobSub via Vision
@needs_vision
def test_vobsub_dialogue_is_ocrd(tmp_path):
    import vobsub_fixture as V
    V.write_vobsub(str(tmp_path / "vs"), [(1.0, 3.0, "Hello from the DVD"),
                                          (4.0, 6.5, "Second subtitle here")])
    clip = tmp_path / "dvd.mkv"
    ff("-f", "lavfi", "-i", "testsrc2=s=720x480:r=24:d=8", "-i", str(tmp_path / "vs.idx"),
       "-map", "0:v", "-map", "1:s", "-c:v", "libx264", "-preset", "ultrafast",
       "-c:s", "copy", "-metadata:s:s:0", "language=eng", str(clip))
    r = run_engine(str(clip), "-o", str(tmp_path / "out"), "--no-motion")
    assert r.returncode == 0, r.stderr
    rows = [l.split("\t") for l in
            (tmp_path / "out" / "INDEX.tsv").read_text().splitlines()[1:]]
    by_sec = {round(_secs(r[1])): r[2] for r in rows}
    assert by_sec[2] == "Hello from the DVD"
    assert by_sec[5] == "Second subtitle here"
    assert by_sec[7] == ""
    man = (tmp_path / "out" / "MANIFEST.txt").read_text()
    assert "Apple Vision OCR of dvd_subtitle subtitles (2 lines)" in man
    assert not list((tmp_path / "out").glob("_bitmap_png"))


def _secs(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


# ---------------------------------------------------------------- Vision helper
@needs_vision
def test_vision_ocr_reads_top_line_first(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    binp = tmp_path / "vision_ocr"
    subprocess.run(["swiftc", "-O", str(REPO / "vision_ocr.swift"), "-o", str(binp)],
                   check=True, capture_output=True)
    im = Image.new("L", (1200, 260), 255)
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=64)
    d.text((80, 30), "First line on top", fill=0, font=f)
    d.text((80, 140), "Second line below", fill=0, font=f)
    im.save(tmp_path / "two.png")
    text = E.vision_ocr(str(binp), [str(tmp_path / "two.png")])
    assert text["two"] == "First line on top Second line below"


def test_vision_ocr_bin_rebuilds_when_source_is_newer(tmp_path, monkeypatch):
    calls = []
    src = tmp_path / "vision_ocr.swift"; src.write_text("// new")
    b = tmp_path / "vision_ocr"; b.write_text("old")
    os.utime(b, (1, 1))                                    # binary older than source
    def fake_run(cmd, **kw):
        calls.append(cmd); Path(cmd[-1]).write_text("rebuilt")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(E.subprocess, "run", fake_run)
    # _vision_ocr_bin imports shutil locally: patch the module it gets
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/swiftc" if n == "swiftc" else None)
    monkeypatch.delenv("CRV_VISION_OCR", raising=False)
    monkeypatch.setattr(E.os.path, "realpath", lambda p: str(tmp_path / "enhance_process.py"))
    assert E._vision_ocr_bin() == str(b)
    assert calls and calls[0][-1] == str(b) and b.read_text() == "rebuilt"


# ---------------------------------------------------------------- manual PGS path
def test_build_srt_joins_timing_and_ocr(tmp_path):
    (tmp_path / "timing.tsv").write_text(
        "00001\t00:00:01,000\t00:00:03,000\n00002\t00:00:04,000\t00:00:05,500\n"
        "00003\t00:00:06,000\t00:00:07,000\n")
    (tmp_path / "ocr.tsv").write_text(
        f"{tmp_path}/00001.png\tHello\n{tmp_path}/00002.png\t\n{tmp_path}/00003.png\tBye\n")
    r = subprocess.run([sys.executable, str(REPO / "build_srt.py"), str(tmp_path / "timing.tsv"),
                        str(tmp_path / "ocr.tsv"), str(tmp_path / "out.srt")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert E.parse_srt(str(tmp_path / "out.srt")) == [(1.0, 3.0, "Hello"), (6.0, 7.0, "Bye")]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_spatial_layer_on_synthetic_51(tmp_path):
    """A 16 s 5.1(side) clip with known placements: a 1 kHz tone in FC at
    1-3 s (speech-band energy with no cue: voice stays blank, since the clip
    has a cue track and the cue is the only gate), a 40 Hz burst in LFE at
    7-8 s (impact), rear noise
    at 9-11 s with quiet fronts (field=enveloping), a 2.5 s 40 Hz rumble at
    3-5.5 s (a Rumble event in sounds.tsv), a 1 kHz tone in FL only at
    12.5-14.5 s under a subtitle cue (voice=off-L through the cue gate: the
    clip is too short for the film-relative energy threshold to mean much).
    Fronts carry a -40 dB floor throughout."""
    floor = "0.01*random(0)"
    srt = tmp_path / "s51.srt"
    srt.write_text("1\n00:00:12,500 --> 00:00:14,500\nOff-screen line.\n", encoding="utf-8")
    chans = [
        f"aevalsrc='{floor}+0.4*sin(1000*2*PI*t)*between(t,12.5,14.5)':s=8000:d=16",   # FL
        f"aevalsrc='{floor}':s=8000:d=16",                                         # FR
        f"aevalsrc='{floor}+0.4*sin(1000*2*PI*t)*between(t,1,3)':s=8000:d=16",     # FC
        f"aevalsrc='0.5*sin(40*2*PI*t)*between(t,3,5.5)+0.8*sin(40*2*PI*t)*between(t,7,8)':s=8000:d=16",  # LFE: rumble 3-5.5, hit 7-8
        f"aevalsrc='{floor}+0.3*random(0)*between(t,9,11)':s=8000:d=16",           # SL
        f"aevalsrc='{floor}+0.3*random(0)*between(t,9,11)':s=8000:d=16",           # SR
    ]
    clip = tmp_path / "s51.mkv"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=16"]
    for c in chans:
        cmd += ["-f", "lavfi", "-i", c]
    cmd += ["-i", str(srt),
            "-filter_complex", "[1:a][2:a][3:a][4:a][5:a][6:a]join=inputs=6:channel_layout=5.1(side):map=0.0-FL\|1.0-FR\|2.0-FC\|3.0-LFE\|4.0-SL\|5.0-SR[a]",
            "-map", "0:v", "-map", "[a]", "-map", "7:s", "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "ac3", "-ar", "48000", "-c:s", "srt", "-metadata:s:s:0", "language=eng", str(clip)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    info = E.audio_info(str(clip))
    assert info["channels"] == 6 and info["layout"] == "5.1(side)" and info["names"], info

    out = tmp_path / "out"
    r = subprocess.run([sys.executable, E.__file__, str(clip),
                        "-o", str(out), "--no-transcribe", "--no-motion"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = (out / "INDEX.tsv").read_text(encoding="utf-8").splitlines()
    hdr = lines[0].split("\t")
    rows = [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]
    def at(t):
        def sec(s):
            h, m, x = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(x)
        return min(rows, key=lambda r: abs(sec(r["time"]) - t))
    # energy alone no longer places voice when a cue track exists (the rule
    # put off-L/off-R on 158 cue-less beats of Alien: strings read as speech)
    assert at(2.0)["voice"] == "", at(2.0)
    assert at(7.5)["impact"] in ("hit", "hard"), at(7.5)
    assert at(10.0)["field"] == "enveloping", at(10.0)
    assert at(13.5)["voice"] == "off-L" and at(13.5)["dialogue"] == "Off-screen line.", at(13.5)
    assert at(15.0)["impact"] == "" and at(15.0)["voice"] == "", at(15.0)
    assert at(4.0)["sound"].startswith("Rumble-1"), at(4.0)
    snd = (out / "sounds.tsv").read_text(encoding="utf-8").splitlines()
    assert snd[0] == "label\tfirst_time\tcount\tsecs\tpitch\ttexture\tplacement", snd[0]
    rumble = next(l.split("\t") for l in snd[1:] if l.startswith("Rumble-1"))
    assert rumble[1] in ("00:00:02", "00:00:03") and 2.0 <= float(rumble[3]) <= 3.2 \
        and rumble[5] == "rumble", rumble
    man = (out / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "spatial: 5.1(side)" in man and "BS.1770" in man, man
    ev = (out / "events.tsv").read_text(encoding="utf-8").splitlines()
    assert ev[0] == "t\tdur\tlabel\tpitch\ttexture\tplacement", ev[0]
    assert any(l.split("\t")[2] == "Rumble-1" for l in ev[1:]), ev
