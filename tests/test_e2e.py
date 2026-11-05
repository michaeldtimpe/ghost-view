"""End-to-end tests: build a tiny synthetic clip with ffmpeg, run the engine
(and redialogue.py) as real subprocesses, and check the on-disk outputs.

Marked slow, but not skipped by default -- only skipped when ffmpeg itself is
unavailable, since these deliberately exercise real ffmpeg/ffprobe behavior.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENHANCE_PY = str(REPO_ROOT / "enhance_process.py")
REDIALOGUE_PY = str(REPO_ROOT / "redialogue.py")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"),
]


def _hms_to_sec(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def run_engine(args, cwd):
    return subprocess.run(
        [sys.executable, ENHANCE_PY, *args], cwd=str(cwd), capture_output=True, text=True
    )


def run_redialogue(args, cwd):
    return subprocess.run(
        [sys.executable, REDIALOGUE_PY, *args], cwd=str(cwd), capture_output=True, text=True
    )


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """~12s clip: 3 hard-cut 4s segments @640x360/24fps, a sine audio track, and
    an embedded English subrip stream with an ASS override and a numeric-only
    cue, per the harness spec."""
    d = tmp_path_factory.mktemp("crv_e2e_src")
    srt = d / "subs.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n\n"
        "2\n00:00:03,100 --> 00:00:05,000\n1984\n\n"
        "3\n00:00:06,000 --> 00:00:08,000\n{\\an8}Top line\n",
        encoding="utf-8",
    )
    out = d / "clip.mkv"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=4",
        "-f", "lavfi", "-i", "smptebars=s=640x360:r=24:d=4",
        "-f", "lavfi", "-i", "mandelbrot=s=640x360:r=24",
        "-f", "lavfi", "-i", "sine=f=440:d=12",
        "-i", str(srt),
        "-filter_complex",
        "[2:v]trim=duration=4,setpts=PTS-STARTPTS[m];"
        "[0:v][1:v][m]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-map", "4:s",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-c:s", "srt",
        "-metadata:s:s:0", "language=eng",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"clip build failed: {r.stderr}"
    assert out.exists()
    return out


def test_engine_basic_run_produces_expected_outputs(tmp_path, clip):
    out = tmp_path / "out"
    r = run_engine([str(clip), "-o", str(out), "--no-transcribe"], tmp_path)
    assert r.returncode == 0, r.stderr

    frames_dir = out / "frames"
    frame_files = sorted(os.listdir(frames_dir))
    assert len(frame_files) >= 3
    for f in frame_files:
        assert re.fullmatch(r"frame_\d{5}\.jpg", f), f
    assert not any(f.startswith(("raw_", "tmp_")) for f in os.listdir(out))
    assert not any(f.startswith(("raw_", "tmp_")) for f in frame_files)

    idx_path = out / "INDEX.tsv"
    lines = idx_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "frame\ttime\tdialogue\tmotion\tcuts_min\tloudness\ttrend\ttempo_bpm\tbrightness\tsound"
        "\timpact\tfield\tvoice\tpan\tcut\theld_s\ttext"
    )
    rows = [ln.split("\t") for ln in lines[1:]]
    assert rows, "no data rows in INDEX.tsv"
    for row in rows:
        assert len(row) == 17, row
    held = [float(row[15]) for row in rows]
    assert all(h >= 0 for h in held) and sum(held) > 0
    # the clip has hard cuts at 4 s and 8 s; the motion pass must mark them
    cut_rows = [row for row in rows if row[14] == "cut"]
    assert cut_rows, "no cut marked"
    assert all(3.3 <= _hms_to_sec(row[1]) <= 9.0 for row in cut_rows), cut_rows

    frame_col = [row[0] for row in rows]
    assert frame_col == frame_files

    secs = [_hms_to_sec(row[1]) for row in rows]
    assert secs == sorted(secs)

    dialogues = [row[2] for row in rows]
    assert any("1984" in d for d in dialogues)
    man0 = (out / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "dialogue coverage: 3/3 cues" in man0, man0
    dsrt = (out / "dialogue.srt").read_text(encoding="utf-8")
    assert "Hello there." in dsrt and "Top line" in dsrt
    dtsv = (out / "dialogue.tsv").read_text(encoding="utf-8").splitlines()
    assert dtsv[0] == "n\tstart\tend\tframe\ttext" and len(dtsv) == 4
    assert all(l.split("\t")[3].startswith("frame_") for l in dtsv[1:]), dtsv

    sheets = sorted(os.listdir(out / "sheets"))
    assert sheets and sheets[0] == "sheet_001.jpg", sheets
    st = (out / "sheets.tsv").read_text(encoding="utf-8").splitlines()
    assert st[0] == "sheet\tfirst_frame\tlast_frame\tfirst_time\tlast_time" and len(st) == len(sheets) + 1
    assert st[1].split("\t")[1] == frame_files[0]

    manifest = (out / "MANIFEST.txt").read_text(encoding="utf-8")
    tl = next((l for l in manifest.splitlines() if l.startswith("timings:")), "")
    assert re.search(r"extract \d+\.\ds", tl), manifest
    assert re.search(r"audio \d+\.\ds \(waited \d+\.\ds\)", tl), tl
    assert re.search(r"total \d+\.\ds$", tl), tl
    assert re.search(r"\d+ extracted \(\d+\.\ds\)", r.stdout), r.stdout

    idx_text = idx_path.read_text(encoding="utf-8")
    sync_text = (out / "sync.md").read_text(encoding="utf-8")
    assert "{\\an8}" not in idx_text
    assert "{\\an8}" not in sync_text

    manifest = (out / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "embedded subrip" in manifest

    loudness_col = [row[5] for row in rows]
    motion_col = [row[3] for row in rows]
    assert any(x.strip() for x in loudness_col), "no loudness values -- audio didn't run"
    assert any(x.strip() for x in motion_col), "no motion values"


def test_engine_rerun_with_stale_junk_yields_same_frame_list(tmp_path, clip):
    out = tmp_path / "out"
    r1 = run_engine([str(clip), "-o", str(out), "--no-transcribe"], tmp_path)
    assert r1.returncode == 0, r1.stderr
    frames_dir = out / "frames"
    first_list = sorted(os.listdir(frames_dir))

    (frames_dir / "raw_00099.jpg").write_bytes(b"junk")
    (frames_dir / "frame_042.jpg").write_bytes(b"junk")

    r2 = run_engine([str(clip), "-o", str(out), "--no-transcribe"], tmp_path)
    assert r2.returncode == 0, r2.stderr
    second_list = sorted(os.listdir(frames_dir))
    assert second_list == first_list


def test_nonexistent_source_exits_1_and_no_out_dir(tmp_path):
    out = tmp_path / "out"
    missing = tmp_path / "does_not_exist.mp4"
    r = run_engine([str(missing), "-o", str(out), "--no-transcribe"], tmp_path)
    assert r.returncode == 1
    assert not out.exists()


def test_non_video_file_exits_1(tmp_path):
    bad = tmp_path / "bad.mkv"
    bad.write_text("this is just text, not a video\n" * 20, encoding="utf-8")
    out = tmp_path / "out"
    r = run_engine([str(bad), "-o", str(out), "--no-transcribe"], tmp_path)
    assert r.returncode == 1


def test_redialogue_replaces_all_dialogue(tmp_path, clip):
    out = tmp_path / "out_orig"
    r = run_engine([str(clip), "-o", str(out), "--no-transcribe"], tmp_path)
    assert r.returncode == 0, r.stderr

    copy_dir = tmp_path / "out_copy"
    shutil.copytree(out, copy_dir)

    new_srt = tmp_path / "new.srt"
    new_srt.write_text(
        "1\n00:00:00,000 --> 00:00:12,000\nReplaced | line\n", encoding="utf-8"
    )

    r2 = run_redialogue([str(copy_dir), str(new_srt)], tmp_path)
    assert r2.returncode == 0, r2.stderr

    idx_lines = (copy_dir / "INDEX.tsv").read_text(encoding="utf-8").splitlines()
    rows = [ln.split("\t") for ln in idx_lines[1:]]
    assert rows
    assert all(row[2] == "Replaced | line" for row in rows)

    sync_text = (copy_dir / "sync.md").read_text(encoding="utf-8")
    assert "Replaced \\| line" in sync_text

    manifest = (copy_dir / "MANIFEST.txt").read_text(encoding="utf-8")
    sub_line = next((l for l in manifest.splitlines() if l.startswith("subtitles:")), "")
    assert "new.srt" in sub_line


def test_redialogue_missing_index_dir_exits_1(tmp_path):
    empty_dir = tmp_path / "no_index_here"
    empty_dir.mkdir()
    new_srt = tmp_path / "new2.srt"
    new_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nX\n", encoding="utf-8")
    r = run_redialogue([str(empty_dir), str(new_srt)], tmp_path)
    assert r.returncode == 1


def test_batch_driver_default_engine_produces_index(tmp_path, clip):
    """batch_encode.sh with CRV unset must run the enhanced engine, not whatever
    `crv` the venv ships. A rebuilt venv once put stock claude-real-video there
    (150-frame cap, no INDEX.tsv) and every film came back ENCODE_FAIL. Fake
    the NAS with a stub ssh that cats a local file."""
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(REPO_ROOT / "batch_encode.sh", work / "batch_encode.sh")
    os.symlink(REPO_ROOT / "enhance_process.py", work / "enhance_process.py")
    os.symlink(REPO_ROOT / ".venv", work / ".venv")
    nas = tmp_path / "nas"
    nas.mkdir()
    shutil.copy(clip, nas / "clip.mkv")
    binp = tmp_path / "bin"
    binp.mkdir()
    ssh = binp / "ssh"
    ssh.write_text(
        "#!/bin/bash\nshift; cmd=\"$*\"\n"
        f"case \"$cmd\" in\n  'stat -c%s '*) stat -f%z \"{nas}/clip.mkv\" ;;\n"
        f"  'cat '*) cat \"{nas}/clip.mkv\" ;;\n  *) exit 1 ;;\nesac\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    (work / "films.tsv").write_text("Clip\t/clip.mkv\tclip\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "CRV"}
    env["PATH"] = f"{binp}:{env.get('PATH', '')}"
    env["NAS"] = "fakenas"
    r = subprocess.run(["bash", "./batch_encode.sh", "films.tsv"], cwd=str(work),
                       capture_output=True, text=True, env=env)
    status = (work / "batch_status.tsv").read_text(encoding="utf-8")
    assert "Clip\tDONE\t" in status, (status, r.stdout, r.stderr,
                                      (work / "crv-clip.run.log").read_text() if (work / "crv-clip.run.log").exists() else "")
    assert (work / "crv-clip" / "INDEX.tsv").exists()
    assert not (work / "originals" / "clip.mkv").exists()


def test_max_gap_keeps_ticking_through_a_held_shot(tmp_path):
    """A 12 s static clip with a cue in the middle: with --max-gap 3 frames are
    kept at least every 3 s inside the held shot, held_s says so, and the cue
    lands on the frame it was spoken over."""
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:05,000 --> 00:00:06,000\nMid line.\n", encoding="utf-8")
    clip = tmp_path / "static.mkv"
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "smptebars=s=320x180:r=24:d=12",
                        "-f", "lavfi", "-i", "sine=f=440:d=12", "-i", str(srt),
                        "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "libx264",
                        "-preset", "ultrafast", "-c:a", "aac", "-c:s", "srt",
                        "-metadata:s:s:0", "language=eng", str(clip)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "o"
    r = run_engine([str(clip), "-o", str(out), "--no-transcribe", "--no-motion", "--no-audio",
                    "--max-gap", "3"], tmp_path)
    assert r.returncode == 0, r.stderr
    lines = (out / "INDEX.tsv").read_text(encoding="utf-8").splitlines()
    rows = [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:]]
    times = [_hms_to_sec(r["time"]) for r in rows]
    assert len(rows) >= 4, times
    assert max(b - a for a, b in zip(times, times[1:])) <= 3.5, times
    assert all(float(r["held_s"]) <= 3.5 for r in rows[:-1]), rows
    assert "Mid line." in " ".join(r["dialogue"] for r in rows)
    dtsv = (out / "dialogue.tsv").read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert 3.0 <= _hms_to_sec(next(r["time"] for r in rows if r["frame"] == dtsv[3])) <= 5.0, dtsv
