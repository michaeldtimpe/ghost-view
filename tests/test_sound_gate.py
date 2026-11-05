"""The AudioSet gate in sound_tags.py, on hand-made probability vectors (no model load).
sound_tags is a uv script (torch); importing it needs only numpy-free module code."""
import numpy as np
import sound_tags as S


def _ast():
    a = S.Ast.__new__(S.Ast)
    a.labels = list(dict.fromkeys(["Silence", "Music", "Speech", "Whale vocalization", *S.AST_EVENTS, *S.AST_MOOD]))
    a.idx = {l: i for i, l in enumerate(a.labels)}
    return a


def _p(**kw):
    a = _ast(); p = np.zeros(len(a.labels), np.float32)
    for k, v in kw.items(): p[a.idx[k.replace("_", " ")]] = v
    return a, p


def test_quiet_window_gets_no_tag():
    a, p = _p(Silence=0.7)
    assert a.gate(p)[:3] == ("quiet", "", "")


def test_drone_is_music_not_helicopter():
    a, p = _p(Music=0.6, Scary_music=0.3)
    mode, event, mood, _ = a.gate(p)
    assert (mode, event, mood) == ("music", "", "scary music")


def test_confident_event_wins_over_music():
    a, p = _p(Music=0.6, Knock=0.5)
    assert a.gate(p)[:2] == ("effects", "knocking")


def test_unmapped_audioset_label_is_ignored():
    a, p = _p(Whale_vocalization=0.9, Silence=0.6)
    assert a.gate(p)[0] == "quiet"


def test_mechanical_pick_needs_weak_ast_vote():
    a, p = _p(Helicopter=0.02)
    assert not a.confirms(p, "helicopter")
    assert a.confirms(p, "drone score")          # AST has no word for it: passes
    assert a.confirms(p, "crying")               # not a ghost label: passes without a vote
    a, p = _p(Helicopter=0.2)
    assert a.confirms(p, "helicopter")


def test_quiet_gate_needs_strong_silence():
    a, p = _p(Silence=0.4)
    assert a.gate(p)[0] == "effects"
    assert set(S.CONFIRM) <= set(S.VOCAB)


def test_label_map_names_are_vocabulary_names():
    assert set(S.AST_EVENTS.values()) <= set(S.VOCAB)
    assert set(S.AST_MOOD.values()) - {"scary music", "sad music", "tender music", "exciting music",
                                       "happy music", "angry music", "opera", "chant"} <= set(S.VOCAB)
