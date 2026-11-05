"""The second-review fixes (2026-11-02): captions survive whichever pass writes
sync.md last, OCR text is a sync.md state key, dark footage cuts, SDH cue
descriptions, voice gated on a cue, MANIFEST edits."""
import os
import numpy as np
import pytest
import enhance_process as E
import pkg


def _rows(n=4):
    return [{"frame": f"frame_{i:05d}.jpg", "time": f"00:00:{10 * i:02d}.000", "dialogue": "",
             "motion": "static", "loudness": "", "sound": "", "shot": f"s{(i // 2) + 1:04d}"}
            for i in range(n)]


# ------------------------------------------------- captions from shots.tsv sidecar
def test_sync_md_carries_captions_from_shots_tsv(tmp_path):
    (tmp_path / "shots.tsv").write_text(
        "shot\tstart\tend\tframes\tsetting\tpeople\tscale\taction\tlight\ttext\ttext_vlm\n"
        "s0001\t00:00:00\t00:00:10\t2\tcorridor\t2\tmedium\ttwo crew walk\tdim\t\t\n"
        "s0002\t00:00:20\t00:00:30\t2\tbridge\t0\twide\tconsoles blink\tdark\t\t\n",
        encoding="utf-8")
    out = tmp_path / "sync.md"
    E.write_sync_md(str(out), "T", _rows())        # rows carry no caption key at all
    text = out.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.startswith("| ")][1:]
    assert "[sees] medium; corridor; 2 people; two crew walk; dim" in lines[0]
    assert "[sees]" not in lines[1]                 # only a shot's first row
    assert "[sees] wide; bridge; consoles blink; dark" in lines[2]   # 0 people omitted


def test_sync_md_rows_with_captions_win_over_sidecar(tmp_path):
    (tmp_path / "shots.tsv").write_text(
        "shot\tstart\tend\tframes\tsetting\tpeople\tscale\taction\tlight\ttext\ttext_vlm\n"
        "s0001\t00:00:00\t00:00:10\t2\tcorridor\t2\tmedium\twalk\tdim\t\t\n", encoding="utf-8")
    rows = _rows(2); rows[0]["caption"] = "from the caller"
    E.write_sync_md(str(tmp_path / "sync.md"), "T", rows)
    assert "[sees] from the caller" in (tmp_path / "sync.md").read_text(encoding="utf-8")


def test_sync_md_text_from_text_tsv_is_a_state_key(tmp_path):
    (tmp_path / "text.tsv").write_text(
        "frame\ttime\ttext\nframe_00001.jpg\t00:00:10.000\tSPECIAL ORDER 937\n"
        "frame_00002.jpg\t00:00:20.000\tSPECIAL ORDER 937\n", encoding="utf-8")
    E.write_sync_md(str(tmp_path / "sync.md"), "T", _rows())
    text = (tmp_path / "sync.md").read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.startswith("| ")][1:]
    assert "text=SPECIAL ORDER 937" in lines[1]
    assert "text=" not in lines[2]                  # unchanged: not repeated
    assert "text=-" in lines[3]                     # gone


# ------------------------------------------------------------- dark footage cuts
def test_cut_mask_scales_dark_footage():
    n = 300
    rng = np.random.default_rng(3)
    fd = 0.6 + rng.uniform(0, 0.8, n)
    fd[100] = 60.0                                  # a lit cut
    fd[200] = 6.0                                   # a cut between two near-black set-ups
    lum = np.full(n, 90.0); lum[150:250] = 10.0     # the dark passage
    assert list(np.flatnonzero(E.cut_mask(fd))) == [100]          # the old reading
    assert list(np.flatnonzero(E.cut_mask(fd, lum))) == [100, 200]
    # the scale never removes a lit cut, and never boosts a bright sample
    assert E.lum_scale([200.0])[0] == 1.0
    assert E.lum_scale([2.0])[0] == E.CUT_LUM_REF / E.CUT_LUM_FLOOR


def test_cut_mask_dark_noise_is_not_a_cut():
    n = 300
    rng = np.random.default_rng(4)
    fd = 0.6 + rng.uniform(0, 0.8, n)
    lum = np.full(n, 9.0)                           # a whole dark film, no cut
    assert not E.cut_mask(fd, lum).any()


# ----------------------------------------------------------------- SDH cue track
@pytest.mark.parametrize("text,want", [
    ("[RHYTHMIC DRONING]", ["rhythmic droning"]),
    ("(chains rattling) Come on.", ["chains rattling"]),
    ("[PARKER] What you say?", []),
    ("(Ripley) Dallas?", []),
    ("[COMPUTER VOICE] Attention.", []),
    ("[NOISES STOP]", ["noises stop"]),
    ("(sighs) [over radio] Copy.", ["sighs", "over radio"]),
    ("♪ ♪", []),
    ("plain line", []),
])
def test_sdh_groups(text, want):
    assert E.sdh_groups(text) == want


def test_sdh_cues_flags_a_track():
    plain = [(i, i + 1, f"line {i}") for i in range(100)]
    sdh = plain + [(200 + i, 201 + i, "[beeping]") for i in range(4)]   # 4% of cues
    assert E.sdh_cues(plain) == ([], False)
    descs, is_sdh = E.sdh_cues(sdh)
    assert is_sdh and descs[0] == (200, 201, "beeping")
    assert E.sdh_cues([])[1] is False


# ---------------------------------------------------------------------- MANIFEST
def test_manifest_set_replaces_or_inserts(tmp_path):
    (tmp_path / "MANIFEST.txt").write_text("source: x.mkv\nframes: 3 kept\nengine: e (output format 7)\n",
                                           encoding="utf-8")
    d = str(tmp_path)
    pkg.manifest_set(d, "origin", "/media/films/X/x.mkv", after="source")
    pkg.manifest_set(d, "frames", "4 kept")
    pkg.manifest_set(d, "ocr", "done")
    txt = (tmp_path / "MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    assert txt == ["source: x.mkv", "origin: /media/films/X/x.mkv", "frames: 4 kept",
                   "ocr: done", "engine: e (output format 7)"]
    assert pkg.manifest_get(d, "origin") == "/media/films/X/x.mkv"
    assert pkg.manifest_get(d, "nope") == ""


def test_index_roundtrip(tmp_path):
    d = str(tmp_path)
    (tmp_path / "INDEX.tsv").write_text("frame\ttime\tdialogue\nf1\t00:00:01.000\tHi\nf2\t00:00:02.000\n",
                                        encoding="utf-8")
    h, rows = pkg.read_index(d)
    assert rows[1]["dialogue"] == ""                # short row padded
    pkg.add_column(h, "text"); rows[0]["text"] = "A\tB"
    pkg.write_index(d, h, rows)
    h2, rows2 = pkg.read_index(d)
    assert h2 == ["frame", "time", "dialogue", "text"] and rows2[0]["text"] == "A B"


# ------------------------------------------------------------ OCR-aware dedup
def _frames(tmp_path, n, seconds=1.0):
    """n identical black frames a second apart (pure dedup would keep one)."""
    from PIL import Image
    d = tmp_path / "frames"; d.mkdir()
    out = []
    for i in range(n):
        p = d / f"raw_{i:05d}.jpg"
        Image.new("RGB", (64, 36), (0, 0, 0)).save(p)
        out.append((str(p), i * seconds))
    return str(d), out


def test_dedup_keeps_a_frame_when_new_text_appears(tmp_path):
    fdir, raw = _frames(tmp_path, 8)
    ocr = {"raw_00000": "", "raw_00001": "", "raw_00002": "SPECIAL ORDER 937",
           "raw_00003": "SPECIAL ORDER 937", "raw_00004": "SPECIAL ORDER 937 CREW EXPENDABLE",
           "raw_00005": "SPECIAL ORDER 937 CREW EXPENDABLE", "raw_00006": "ab", "raw_00007": ""}
    kept_text = {}
    beats = E.dedup(raw, fdir, max_gap=0, ocr=ocr, kept_text=kept_text)
    assert [round(t) for _, t in beats] == [0, 2, 4]          # first, text, more text
    assert kept_text == {"frame_00001.jpg": "", "frame_00002.jpg": "SPECIAL ORDER 937",
                         "frame_00003.jpg": "SPECIAL ORDER 937 CREW EXPENDABLE"}
    assert E.ocr_words("ab, x1 -- Sp3cial!") == {"SP3CIAL"}


def test_dedup_text_keep_is_rate_limited(tmp_path):
    fdir, raw = _frames(tmp_path, 6, seconds=0.5)
    ocr = {f"raw_{i:05d}": f"WORD{i}" for i in range(6)}      # a new word every half second
    beats = E.dedup(raw, fdir, max_gap=0, ocr=ocr, text_gap=2.0)
    assert [t for _, t in beats] == [0.0, 2.0]


def test_dedup_without_ocr_is_unchanged(tmp_path):
    fdir, raw = _frames(tmp_path, 5)
    assert len(E.dedup(raw, fdir, max_gap=0)) == 1


# ------------------------------------------------------------ OCR watchdog
def _fake_ocr(tmp_path, hang_on, hang_times):
    """A stand-in for the Vision helper: prints '<path>\\tTEXT' per argument,
    but sleeps forever when an argument contains `hang_on`, for the first
    `hang_times` calls that include it (a counter file keeps the count)."""
    cnt = tmp_path / "count"; cnt.write_text("0")
    b = tmp_path / "fake_ocr"
    b.write_text(f"""#!/bin/bash
for a in "$@"; do
  case "$a" in *{hang_on}*)
    n=$(cat {cnt}); echo $((n+1)) > {cnt}
    [ "$n" -lt {hang_times} ] && exec sleep 60;;
  esac
done
for a in "$@"; do printf '%s\\tTEXT %s\\n' "$a" "$(basename "$a" .jpg)"; done
""")
    b.chmod(0o755)
    return str(b)


def test_vision_ocr_retries_a_transient_hang(tmp_path):
    paths = [str(tmp_path / f"f{i}.jpg") for i in range(6)]
    binp = _fake_ocr(tmp_path, "f3", hang_times=1)
    failed = []
    got = E.vision_ocr(binp, paths, batch=3, workers=2, failed=failed,
                       timeout_base=1.0, timeout_per=0.0)
    assert got == {f"f{i}": f"TEXT f{i}" for i in range(6)}
    assert failed == []


def test_vision_ocr_isolates_a_frame_that_always_hangs(tmp_path):
    paths = [str(tmp_path / f"f{i}.jpg") for i in range(4)]
    binp = _fake_ocr(tmp_path, "f2", hang_times=99)
    failed = []
    got = E.vision_ocr(binp, paths, batch=4, workers=1, failed=failed,
                       timeout_base=0.5, timeout_per=0.0, tries=2)
    assert got == {"f0": "TEXT f0", "f1": "TEXT f1", "f3": "TEXT f3"}
    assert [os.path.basename(p) for p in failed] == ["f2.jpg"]
