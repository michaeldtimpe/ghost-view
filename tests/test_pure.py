"""Pure-function tests for enhance_process.py — no ffmpeg, no subprocess."""
import enhance_process as E


# ---------------------------------------------------------------- hms
def test_hms_zero():
    assert E.hms(0) == "00:00:00.000"


def test_hms_hours_minutes_seconds():
    assert E.hms(3661.5) == "01:01:01.500"


def test_hms_negative_clamps_to_zero():
    assert E.hms(-5) == "00:00:00.000"


# ---------------------------------------------------------------- parse_srt
def test_parse_srt_basic_two_cue(tmp_path):
    p = tmp_path / "basic.srt"
    p.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n\n"
        "2\n00:00:03,100 --> 00:00:05,000\nBye.\n",
        encoding="utf-8",
    )
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 3.0, "Hello there."), (3.1, 5.0, "Bye.")]


def test_parse_srt_crlf(tmp_path):
    p = tmp_path / "crlf.srt"
    content = "1\r\n00:00:01,000 --> 00:00:03,000\r\nHello there.\r\n\r\n"
    p.write_bytes(content.encode("utf-8"))
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 3.0, "Hello there.")]


def test_parse_srt_utf8_bom(tmp_path):
    p = tmp_path / "bom.srt"
    content = "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n"
    p.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 3.0, "Hello there.")]


def test_parse_srt_numeric_text_survives(tmp_path):
    p = tmp_path / "numeric.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:03,000\n1984\n", encoding="utf-8")
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 3.0, "1984")]


def test_parse_srt_strips_html_tags(tmp_path):
    p = tmp_path / "tags.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:02,000\n<i>Hello</i> there.\n", encoding="utf-8")
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 2.0, "Hello there.")]


def test_parse_srt_strips_ass_override(tmp_path):
    p = tmp_path / "ass.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:02,000\n{\\an8}Top line\n", encoding="utf-8")
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 2.0, "Top line")]


def test_parse_srt_multiline_joined_with_space(tmp_path):
    p = tmp_path / "multiline.srt"
    p.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nLine one\nLine two\n", encoding="utf-8"
    )
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 4.0, "Line one Line two")]


def test_parse_srt_webvtt_mmss_no_hours(tmp_path):
    p = tmp_path / "clip.vtt"
    p.write_text(
        "WEBVTT\n\n01:30.500 --> 01:33.000\nHello there.\n", encoding="utf-8"
    )
    cues = E.parse_srt(str(p))
    assert cues == [(90.5, 93.0, "Hello there.")]


def test_parse_srt_ignores_blocks_without_timecodes(tmp_path):
    p = tmp_path / "note.srt"
    p.write_text(
        "NOTE this block has no timecode\nnothing here\n\n"
        "1\n00:00:01,000 --> 00:00:02,000\nReal cue.\n",
        encoding="utf-8",
    )
    cues = E.parse_srt(str(p))
    assert cues == [(1.0, 2.0, "Real cue.")]


# ---------------------------------------------------------------- _is_marker
def test_is_marker_true_cases():
    for t in ("", "♪", "♪ ♪", "[MUSIC]", "(TRAIN WHISTLE BLOWING)"):
        assert E._is_marker(t) is True, t


def test_is_marker_false_cases():
    for t in ("Hello", "(SIGHS) I know. (LAUGHS)"):
        assert E._is_marker(t) is False, t


# ---------------------------------------------------------------- loop_damage / _loop_spans
def test_loop_damage_no_cues():
    assert E.loop_damage([]) == (0.0, 0.0, "", 0)
    assert E._loop_spans([]) == []


def test_loop_damage_single_run_of_five_in_ten():
    cues = [
        (0, 1, "A"),
        (1, 2, "B"),
        (2, 3, "C"),
        (3, 4, "Same"),
        (4, 5, "Same"),
        (5, 6, "Same"),
        (6, 7, "Same"),
        (7, 8, "Same"),
        (8, 9, "D"),
        (9, 10, "E"),
    ]
    frac, lost, worst, runs = E.loop_damage(cues)
    assert frac == 0.5
    assert runs == 1
    assert worst == "Same"
    assert lost == cues[7][1] - cues[3][0]
    assert E._loop_spans(cues) == [(cues[3][0], cues[7][1])]


def test_loop_damage_alternating_pair_is_a_loop():
    a, b = "I can't help it.", "Just leave before I get used to it."
    cues = [(float(k), k + 1.0, "line %d" % k) for k in range(3)]
    cues += [(10.0 + k, 11.0 + k, (a, b)[k % 2]) for k in range(8)]
    cues += [(30.0, 31.0, "after")]
    frac, lost, worst, runs = E.loop_damage(cues)
    assert runs == 1 and lost == 8.0
    assert E._loop_spans(cues) == [(10.0, 18.0)]


def test_loop_damage_alternating_pair_twice_is_dialogue():
    cues = [(float(k), k + 1.0, ("Yeah.", "No.")[k % 2]) for k in range(4)]
    assert E.loop_damage(cues)[3] == 0


def test_loop_damage_run_of_two_is_not_a_loop():
    cues = [(0, 1, "X"), (1, 2, "X"), (2, 3, "Y")]
    assert E.loop_damage(cues) == (0.0, 0.0, "", 0)
    assert E._loop_spans(cues) == []


def test_loop_damage_ignores_marker_runs():
    cues = [(i, i + 1, "♪") for i in range(6)]  # 6 identical "♪" cues
    assert E.loop_damage(cues) == (0.0, 0.0, "", 0)
    assert E._loop_spans(cues) == []


def test_loop_damage_two_separate_runs():
    cues = [
        (0, 1, "A"), (1, 2, "A"), (2, 3, "A"),
        (3, 4, "B"),
        (4, 5, "C"), (5, 6, "C"), (6, 7, "C"),
    ]
    frac, lost, worst, runs = E.loop_damage(cues)
    assert runs == 2


# ---------------------------------------------------------------- _snap
def test_snap_returns_nearest_seam_within_window():
    seams = [10.0, 50.0, 200.0]
    assert E._snap(15, seams) == 10.0


def test_snap_returns_t_unchanged_when_no_seam_in_window():
    seams = [10.0, 50.0, 200.0]
    assert E._snap(1000, seams) == 1000


def test_snap_after_ignores_seams_at_or_before_cutoff():
    # Without the `after` filter the nearest seam (100) would win outright;
    # with after=90 (cutoff 150) it must be skipped in favor of the farther,
    # qualifying seam at 400.
    seams = [100.0, 400.0]
    assert E._snap(100, seams, window=350.0) == 100.0
    assert E._snap(100, seams, window=350.0, after=90.0) == 400.0


# ---------------------------------------------------------------- subtitle stream picking
def _stream(idx, codec="subrip", lang="", forced=False, title=""):
    return {"idx": idx, "codec": codec, "lang": lang, "forced": forced, "title": title}


def test_pick_text_sub_english_beats_arabic():
    streams = [_stream(0, lang="ara"), _stream(1, lang="eng")]
    assert E._pick_text_sub(streams)["idx"] == 1


def test_pick_text_sub_nonforced_beats_forced_english():
    streams = [_stream(0, lang="eng", forced=True), _stream(1, lang="eng", forced=False)]
    assert E._pick_text_sub(streams)["idx"] == 1


def test_pick_text_sub_nonsdh_beats_sdh_english():
    streams = [_stream(0, lang="eng", title="english sdh"), _stream(1, lang="eng", title="")]
    assert E._pick_text_sub(streams)["idx"] == 1


def test_pick_text_sub_untagged_beats_foreign():
    streams = [_stream(0, lang="fre"), _stream(1, lang="")]
    assert E._pick_text_sub(streams)["idx"] == 1


def test_pick_text_sub_forced_detected_via_title():
    streams = [_stream(0, lang="eng", forced=False, title="forced"),
               _stream(1, lang="eng", forced=False, title="")]
    assert E._pick_text_sub(streams)["idx"] == 1


def test_pick_text_sub_ignores_pgs_and_none_when_only_pgs():
    streams = [_stream(0, codec="hdmv_pgs_subtitle", lang="eng")]
    assert E._pick_text_sub(streams) is None


def test_pick_pgs_sub_mirror():
    # Only text streams present -> no PGS candidate.
    text_only = [_stream(0, codec="subrip", lang="eng")]
    assert E._pick_pgs_sub(text_only) is None
    # PGS present, with the same ranking rules applied among PGS streams.
    streams = [
        _stream(0, codec="hdmv_pgs_subtitle", lang="ara"),
        _stream(1, codec="hdmv_pgs_subtitle", lang="eng"),
    ]
    assert E._pick_pgs_sub(streams)["idx"] == 1


# ---------------------------------------------------------------- line_at
def test_line_at_basic_and_boundaries():
    cues = [(1, 3, "A"), (3.1, 5, "B")]
    assert E.line_at(2, cues) == "A"
    assert E.line_at(3.15, cues) == "B"
    assert E.line_at(3.05, cues) == "B"  # tie in the gap goes to the incoming line
    assert E.line_at(5.2, cues) == "B"   # within 0.3 pad
    assert E.line_at(5.5, cues) == ""
    assert E.line_at(0.8, cues) == "A"
    assert E.line_at(5.4, []) == ""


def test_line_at_unsorted_input_same_answers():
    cues = [(3.1, 5, "B"), (1, 3, "A")]  # reversed order
    assert E.line_at(2, cues) == "A"
    assert E.line_at(3.15, cues) == "B"
    assert E.line_at(5.2, cues) == "B"
    assert E.line_at(5.5, cues) == ""


def test_line_at_overlapping_cues_latest_starting_wins():
    cues = [(0, 10, "long"), (4, 6, "short")]
    assert E.line_at(5, cues) == "short"


def test_line_at_cache_safety_across_different_lists():
    cues_a = [(0, 2, "X")]
    cues_b = [(0, 2, "Y")]
    assert E.line_at(1, cues_a) == "X"
    assert E.line_at(1, cues_b) == "Y"
    assert E.line_at(1, cues_a) == "X"


# ---------------------------------------------------------------- sound_at
def test_sound_at_hit_and_miss():
    sounds = [{"t": 5.0, "dur": 2.0, "label": "scream"}]
    assert E.sound_at(6.0, sounds) == "scream"
    assert E.sound_at(20.0, sounds) == ""


def test_tsv_safe_strips_tabs_and_ascii_quotes():
    assert E.tsv_safe('He said "no"\tnow\n') == "He said \u201dno\u201d now"
    assert E.tsv_safe(None) == ""


def test_sync_md_prints_only_changes(tmp_path):
    rows = [{"frame": "f1", "time": "00:00:01.000", "dialogue": "", "motion": "static",
             "loudness": "quiet", "field": "frontal"},
            {"frame": "f2", "time": "00:00:02.000", "dialogue": "", "motion": "static",
             "loudness": "loud", "field": "frontal", "cut": "cut"},
            {"frame": "f3", "time": "00:00:03.000", "dialogue": "", "motion": "static",
             "loudness": "loud", "field": ""}]
    out = tmp_path / "s.md"
    E.write_sync_md(str(out), "T", rows)
    l = [x for x in out.read_text(encoding="utf-8").splitlines() if x.startswith("| ")][1:]
    assert "cam=static" in l[0] and "loud=quiet" in l[0] and "field=frontal" in l[0]
    assert "cam=" not in l[1] and "loud=loud" in l[1] and l[1].count("cut") == 1
    assert "field=-" in l[2] and "loud=" not in l[2]


# ---------------------------------------------------------------- write_sync_md
def test_write_sync_md(tmp_path):
    rows = [
        {"frame": "frame_00001.jpg", "time": "00:00:01.500", "dialogue": "Hello",
         "motion": "static", "loudness": "quiet", "trend": "—", "tempo_bpm": "",
         "brightness": "dim", "sound": "—"},
        {"frame": "frame_00002.jpg", "time": "01:02:03.456789",
         "dialogue": "Replaced | line", "motion": "—", "loudness": "",
         "trend": "building", "tempo_bpm": "120", "brightness": "—",
         "sound": "scream"},
    ]
    out = tmp_path / "sync.md"
    E.write_sync_md(str(out), "Some Title", rows)
    text = out.read_text(encoding="utf-8")

    assert "| # | frame | time | dialogue | changes |" in text
    lines = text.splitlines()
    row1 = next(l for l in lines if "frame_00001.jpg" in l)
    row2 = next(l for l in lines if "frame_00002.jpg" in l)

    # time truncated to 8 chars
    assert "| 00:00:01 |" in row1
    assert "| 01:02:03 |" in row2

    # pipe in dialogue escaped
    assert "Replaced \\| line" in row2

    # dropped "—" / empty bits, kept the rest, tempo formatted as "120bpm"
    assert "cam=static" in row1 and "loud=quiet" in row1 and "tone=" not in row1
    assert "sound=scream" in row2 and "bpm=" not in row2


# ---------------------------------------------------------------- line_at / sound_at vs brute force
def _line_at_reference(t, cues, pad=0.3):
    """The pre-optimisation algorithm, verbatim in spirit: latest-starting
    spanning cue, else nearest within pad (ties to the incoming line)."""
    ordered = sorted(cues, key=lambda c: (c[0], c[1]))
    span = [c for c in ordered if c[0] <= t <= c[1]]
    if span:
        return span[-1][2]
    import bisect
    i = bisect.bisect_right([c[0] for c in ordered], t)
    best, bd = "", pad + 1e-9
    for k in (i - 1, i):
        if 0 <= k < len(ordered):
            s, e, txt = ordered[k]
            d = (s - t) if t < s else (t - e if t > e else 0.0)
            if d <= bd + 1e-6:
                best, bd = txt, d
    return best


def test_line_at_matches_reference_on_random_cues():
    import random
    rng = random.Random(7)
    cues = []
    for k in range(400):
        s = rng.uniform(0, 3600)
        cues.append((s, s + rng.choice([0.5, 2, 4, 30, 900]), f"L{k}"))
    for _ in range(3000):
        t = rng.uniform(-5, 4600)
        assert E.line_at(t, cues) == _line_at_reference(t, cues), t


def test_sound_at_matches_brute_force_and_handles_unsorted():
    import random
    rng = random.Random(3)
    sounds = [{"t": rng.uniform(0, 600), "dur": rng.uniform(0.3, 8), "label": f"S{k}"}
              for k in range(300)]
    rng.shuffle(sounds)
    ordered = sorted(sounds, key=lambda s: s["t"])
    for _ in range(2000):
        t = rng.uniform(-2, 610)
        # the shortest live sound wins (a long rumble must not mask the
        # events inside it); ties go to the earliest start, as sorted
        live = [s for s in ordered if s["t"] - 0.4 <= t <= s["t"] + s["dur"] + 0.4]
        want = min(live, key=lambda s: s["dur"])["label"] if live else ""
        assert E.sound_at(t, sounds) == want


# ---------------------------------------------------------------- motion classification
def _series(n=30, dx=0.0, dy=0.0, divx=0.0, divy=None, fps_s=3):
    import numpy as np
    z = lambda v: np.r_[0.0, np.full(n - 1, v)]
    if divy is None:
        divy = divx * E.H_OVER_W                 # an isotropic zoom
    return {"times": np.arange(n) / fps_s, "dxs": z(dx), "dys": z(dy),
            "divx": z(divx), "divy": z(divy), "cut_t": np.array([])}


def _labels(ser):
    return {v["motion"] for v in E._motion_beats(ser, [("f", t) for t in (2, 4, 6)]).values()}


def test_motion_pure_zoom_is_not_static():
    # a centred zoom has no net translation -- used to be labelled "static"
    assert _labels(_series(divx=+0.3)) == {"push-in"}
    assert _labels(_series(divx=-0.3)) == {"pull-out"}


def test_motion_subject_motion_in_a_static_frame_is_not_a_zoom():
    import numpy as np
    # horizontal divergence only (someone walks across a locked-off shot)
    assert _labels(_series(divx=+0.3, divy=0.0)) == {"static"}
    # axes disagree
    assert _labels(_series(divx=+0.3, divy=-0.2)) == {"static"}
    # sporadic: the mean says zoom but only a third of the samples agree
    ser = _series(divx=0.0)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(1, 30), 10, replace=False)
    ser["divx"][idx] = 0.9; ser["divy"][idx] = 0.9 * E.H_OVER_W
    assert _labels(ser) == {"static"}


def test_motion_subject_crossing_a_static_frame_is_not_a_pan():
    import numpy as np
    ser = _series()                                  # all still
    ser["dxs"][10:12] = 6.0                          # 2 of the ~5 samples in a window
    assert _labels(ser) == {"static"}


def test_motion_below_zoom_threshold_and_pans():
    assert _labels(_series(divx=0.01)) == {"static"}
    assert _labels(_series(dx=3.0)) == {"pan-R"}
    assert _labels(_series(dx=-3.0)) == {"pan-L"}
    assert _labels(_series(dy=3.0)) == {"tilt-D"}
    assert _labels(_series(dx=9.0)) == {"whip-pan-R"}


# ---------------------------------------------------------------- sound events
def _feat(db):
    import numpy as np
    n = len(db); sr, hop = 16000, 512
    return {"sr": sr, "hop": hop, "n": n, "ftimes": (np.arange(n) * hop + 512) / sr,
            "db": np.asarray(db, float), "centroid": np.full(n, 2000.0),
            "flatness": np.full(n, 0.2), "onset": np.zeros(n)}


def test_sound_events_merge_across_short_gap():
    import numpy as np
    db = np.full(2000, -60.0)
    db[1000:1006] = -10; db[1009:1015] = -10     # 0.19s + 0.1s gap + 0.19s
    _, sounds = E._audio_beats(_feat(db), [], [])
    assert len(sounds) == 1 and sounds[0]["dur"] >= 0.4


def test_sound_events_long_gap_stays_split_and_short_burst_dropped():
    import numpy as np
    db = np.full(3000, -60.0)
    db[1000:1020] = -10                          # 0.64s event
    db[1100:1120] = -10                          # 2.5s later: separate event
    db[2000:2003] = -10                          # 0.1s blip: too short
    _, sounds = E._audio_beats(_feat(db), [], [])
    assert len(sounds) == 2


def test_sound_events_inside_speech_are_ignored():
    import numpy as np
    db = np.full(2000, -60.0); db[1000:1020] = -10
    t0 = _feat(db)["ftimes"][1000]
    _, sounds = E._audio_beats(_feat(db), [], [(t0 - 1, t0 + 2, "talk")])
    assert sounds == []


# ---------------------------------------------------------------- silence seams
def test_seams_pair_in_order_despite_negative_and_trailing_start():
    log = ("[silencedetect] silence_start: -0.0213\n"
           "[silencedetect] silence_end: 2.0 | silence_duration: 2.02\n"
           "[silencedetect] silence_start: 10\n"
           "[silencedetect] silence_end: 12 | silence_duration: 2\n"
           "[silencedetect] silence_start: 50\n")       # runs to EOF, no end
    assert E._seams_from_log(log) == [1.0, 11.0]


# ---------------------------------------------------------------- sidecars
def test_sidecars_exact_then_ranked_tags(tmp_path):
    base = tmp_path / "Movie (2016)"
    for name in ["Movie (2016).fr.srt", "Movie (2016).en.forced.srt",
                 "Movie (2016).en.sdh.srt", "Movie (2016).en.srt", "Movie (2016).nfo",
                 "Movie (2016).vtt", "Other.en.srt"]:
        (tmp_path / name).write_text("x")
    got = [p.split("/")[-1] for p in E._sidecars(str(base) + ".mkv")]
    assert got == ["Movie (2016).vtt", "Movie (2016).en.srt", "Movie (2016).en.sdh.srt",
                   "Movie (2016).fr.srt", "Movie (2016).en.forced.srt"]


def test_get_cues_uses_tagged_sidecar(tmp_path):
    (tmp_path / "film.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    cues, note = E.get_cues(str(tmp_path / "film.mkv"), str(tmp_path / "film.mkv"),
                            str(tmp_path))
    assert cues == [(1.0, 2.0, "Hi")] and "film.en.srt" in note


# ---------------------------------------------------------------- PGS timing / srt
def test_resolve_ends_holds_until_next_capped():
    got = E.resolve_ends([("00002", 5.0, None), ("00001", 1.0, None),
                          ("00003", 20.0, 22.0), ("00004", 30.0, None),
                          ("00005", 30.0, 29.0)])
    assert got["00001"] == (1.0, 5.0)        # held to next start (4s cap not hit)
    assert got["00002"] == (5.0, 9.0)        # capped at 4s before the 20s cue
    assert got["00003"] == (20.0, 22.0)      # has its own end
    assert got["00004"][1] > 30.0            # same-instant successor: still > 0 long
    assert got["00005"][1] > 30.0            # end before start treated as missing


def test_srt_ts():
    assert E.srt_ts(0) == "00:00:00,000"
    assert E.srt_ts(3723.4567) == "01:02:03,457"


# ---------------------------------------------------------------- bitmap change cues
def test_change_cues_blank_clears_and_duplicates_fold():
    changes = [(0.0, ""), (1.05, "Hello"), (3.0, ""), (4.0, "Two"), (5.0, "Two"),
               (6.5, ""), (8.0, "Last")]
    assert E._change_cues(changes, end=9.5) == [
        (1.05, 3.0, "Hello"), (4.0, 6.5, "Two"), (8.0, 9.5, "Last")]


def test_change_cues_hold_cap():
    assert E._change_cues([(1.0, "Stuck")], end=100.0, hold=6.0) == [(1.0, 7.0, "Stuck")]


# ---------------------------------------------------------------- HDR tonemap
def _pq_code(nits):
    y = (nits / 10000.0) ** E._M1
    return int(round(((E._C1 + E._C2 * y) / (1 + E._C3 * y)) ** E._M2 * 65535))


def _tm(code, transfer="pq"):
    import numpy as np
    return E.tonemap_hdr(np.full((1, 1, 3), code, np.uint16), transfer)[0, 0]


def test_tonemap_pq_reference_points():
    assert list(_tm(0)) == [0, 0, 0]
    assert 180 <= _tm(_pq_code(100))[0] <= 192
    assert 236 <= _tm(_pq_code(203))[0] <= 250     # BT.2408 reference white ~ SDR white
    assert 248 <= _tm(_pq_code(400))[0] < 255      # highlights keep gradation...
    assert _tm(_pq_code(1000))[0] == 255           # ...up to the mastering peak
    assert _tm(_pq_code(10000))[0] == 255


def test_tonemap_monotonic_and_neutral():
    vals = [_tm(_pq_code(n)) for n in (0.1, 1, 5, 20, 80, 150, 300, 500)]
    ys = [int(v[0]) for v in vals]
    assert ys == sorted(ys) and len(set(ys)) == len(ys)
    assert all(v[0] == v[1] == v[2] for v in vals)      # greys stay grey


def test_tonemap_hlg_and_colour():
    import numpy as np
    assert list(_tm(0, "hlg")) == [0, 0, 0]
    assert _tm(int(0.75 * 65535), "hlg")[0] > 200         # HLG reference white is bright
    red = np.zeros((1, 1, 3), np.uint16); red[..., 0] = _pq_code(100)
    r, g, b = E.tonemap_hdr(red, "pq")[0, 0]
    assert r > 150 and g < 40 and b < 40


def test_tonemap_rejects_sdr():
    import numpy as np, pytest
    with pytest.raises(ValueError):
        E.tonemap_hdr(np.zeros((1, 1, 3), np.uint16), "bt709")


# ---------------------------------------------------------------- MANIFEST / sounds.tsv
def test_write_outputs_manifest_notes_and_sound_order(tmp_path):
    sounds = [{"label": f"Sound-{lab}", "t": t, "dur": 1.0, "pitch": "mid", "texture": "rough"}
              for lab, t in (("A", 1.0), ("B", 2.0), ("AA", 30.0), ("C", 3.0))]
    beats = [("frame_00001.jpg", 1.0)]
    E.write_outputs(str(tmp_path), beats, [], "no subtitle track", {1.0: {"loud": "mid"}},
                    sounds, "film.mkv",
                    {"frames": "scene > 0.3 + a frame every 2s", "audio": None,
                     "motion": "(skipped — --no-motion)", "color": "HDR (smpte2084/bt2020) x",
                     "engine": "abc123"})
    labels = [l.split("\t")[0] for l in (tmp_path / "sounds.tsv").read_text().splitlines()[1:]]
    assert labels == ["Sound-A", "Sound-B", "Sound-C", "Sound-AA"]
    man = (tmp_path / "MANIFEST.txt").read_text()
    assert "a frame every 2s" in man and "1fps" not in man
    assert "motion: (skipped — --no-motion)" in man
    assert "color: HDR (smpte2084/bt2020) x" in man
    assert "4 distinct non-speech sound(s)" in man
    assert f"engine: abc123 (output format {E.OUTPUT_FORMAT})" in man


def test_engine_version_is_a_string():
    v = E.engine_version()
    assert isinstance(v, str) and v


def test_whisper_cmd_vad_flags(tmp_path, monkeypatch):
    cmd = E.whisper_cmd("whisper-cli", "m.bin", "a.wav", "pref", "en", 4)
    assert "-mc" in cmd and cmd[cmd.index("-mc") + 1] == "0"
    assert "--vad" not in cmd
    cmd = E.whisper_cmd("whisper-cli", "m.bin", "a.wav", "pref", "en", 4, vad="/x/silero.bin")
    assert cmd[cmd.index("--vad") + 1:cmd.index("--vad") + 3] == ["-vm", "/x/silero.bin"]


def test_vad_model_env_then_roots(tmp_path, monkeypatch):
    monkeypatch.delenv("CRV_VAD_MODEL", raising=False)
    monkeypatch.setattr(E, "_model_roots", lambda: (str(tmp_path),))
    assert E.vad_model() == ""
    f = tmp_path / "ggml-silero-v5.1.2.bin"
    f.write_bytes(b"x")
    assert E.vad_model() == str(f)
    g = tmp_path / "other.bin"
    g.write_bytes(b"x")
    monkeypatch.setenv("CRV_VAD_MODEL", str(g))
    assert E.vad_model() == str(g)


def test_cuts_need_an_absolute_jump_and_a_spike(monkeypatch, tmp_path):
    """A static film: tiny MAD must not turn in-shot motion into cuts, and a
    fast pan (several high samples in a row) is not a cut either."""
    import numpy as np
    n = 300
    fd = np.full(n, 0.8); rng = np.random.default_rng(1)
    fd += rng.uniform(0, 1.2, n)                   # in-shot motion, MAD ~0.3
    fd[100] = 60.0                                 # a hard cut
    fd[200:206] = 15.0                             # a fast pan, sustained
    fd[250] = 8.0                                  # a jump under the floor
    med = np.median(fd); mad = np.median(np.abs(fd - med)) + 1e-6
    assert med + 3 * mad < E.CUT_FLOOR             # the case that broke
    assert list(np.flatnonzero(E.cut_mask(fd))) == [100]
    # two cuts close together both count; a cut right at the end counts
    fd[101] = 0.9; fd[103] = 70.0; fd[-1] = 90.0
    assert list(np.flatnonzero(E.cut_mask(fd))) == [100, 103, n - 1]


def test_lines_since_keeps_cues_between_frames():
    cues = [(1.0, 1.4, "Quick line."), (2.0, 4.0, "Long line."), (5.5, 5.8, "Blip.")]
    # frames at 0.5, 3.0, 6.0: the quick line fell between frames and used to vanish
    assert E.lines_since(None, 0.5, cues) == ""
    assert E.lines_since(0.5, 3.0, cues) == "Quick line. / Long line."
    assert E.lines_since(3.0, 6.0, cues) == "Blip."
    # a line live at the frame but started earlier is still shown once
    assert E.lines_since(2.5, 3.5, cues) == "Long line."


def test_write_srt_roundtrip(tmp_path):
    cues = [(1.0, 2.0, "A"), (3.5, 4.25, "B \"quoted\"")]
    p = tmp_path / "d.srt"
    E.write_srt(str(p), cues)
    assert E.parse_srt(str(p)) == [(1.0, 2.0, "A"), (3.5, 4.25, "B \"quoted\"")]


def test_cue_coverage_counts_carried_cues():
    cues = [(1, 2, "A line"), (3, 4, 'Say "hi"'), (5, 6, ""), (7, 8, "Two\nlines")]
    # rows as write_outputs holds them: raw joined text, cleaned only on write
    rows = [{"dialogue": E.lines_since(None, 9, cues)}]
    assert E.cue_coverage(rows, cues) == (3, 3)
    assert E.cue_coverage([{"dialogue": "A line"}], cues) == (1, 3)


def test_write_dialogue_tsv_maps_cues_to_frames(tmp_path):
    beats = [("frame_00001.jpg", 0.0), ("frame_00002.jpg", 4.0), ("frame_00003.jpg", 9.0)]
    p = tmp_path / "d.tsv"
    E.write_dialogue_tsv(str(p), [(5.0, 6.0, "late"), (0.5, 1.0, "early"), (12.0, 13.0, "end")], beats)
    rows = [l.split("\t") for l in p.read_text(encoding="utf-8").splitlines()[1:]]
    assert [(r[0], r[3], r[4]) for r in rows] == [("1", "frame_00001.jpg", "early"),
                                                  ("2", "frame_00002.jpg", "late"),
                                                  ("3", "frame_00003.jpg", "end")]


def test_cuts_on_handheld_footage():
    """Every sample large (handheld): a cut is a jump well above its
    neighbours even though it is not twice them; a fast pan is not."""
    import numpy as np
    rng = np.random.default_rng(2)
    fd = 12.0 + rng.uniform(-3, 3, 300)             # handheld baseline ~12
    fd[100] = 45.0                                   # a cut: +30 over neighbours
    fd[200:206] = 22.0                               # a fast pan: sustained
    cuts = list(np.flatnonzero(E.cut_mask(fd)))
    assert cuts == [100], cuts
