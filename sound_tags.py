#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch", "transformers", "numpy"]
# ///
"""Name the sounds: zero-shot CLAP tags for a finished folder.

The engine's sounds.tsv clusters non-speech events by timbre (Sound-A/B/C)
and the LFE track gives rumbles; neither says *what* a sound is, and the
salience detector misses the faint off-screen soundtrack under dialogue
(gunshots, shouting, furnaces in The Zone of Interest). This pass embeds
audio with a CLAP model and scores it against a film-sound vocabulary, gated
by a supervised AudioSet classifier (AST) that knows what CLAP cannot say:
that a window is quiet, or that it is music. A quiet window gets no tag; a
music window gets only music labels (so a horror drone is "drone score", not
"helicopter"); an effects window keeps the film vocabulary, with AST's own
event in front when it is sure (knock, gunshot, crickets).

  INDEX.tsv     gains `ambience`: the top non-speech label(s) in the ±1 s
                window around each kept frame, when confident
  events.tsv    gains `tag`: the label of each sound event / rumble
  sounds.tsv    gains `tag`: the commonest tag per Sound-X label
  sync.md       regenerated (hear=… on each row)
  MANIFEST.txt  gains a `sound tags:` line

Runs as a uv script (torch + transformers, MPS on Apple Silicon); the model
(default laion/larger_clap_general, ~2 GB) lands in the Hugging Face cache.

usage: uv run --script sound_tags.py <crv_out_dir> <media> [--model M]
       [--window 5.0] [--min-prob 0.15] [--limit N] [--no-ast]
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

SR = 48000

# label -> prompt. Film-mix vocabulary; `speech` and `silence` absorb what
# they absorb and are never reported as ambience.
VOCAB = {
    "speech": "people talking", "silence": "silence, no sound",
    "orchestral score": "orchestral film music", "piano": "piano music",
    "strings": "string instruments playing", "electronic score": "electronic synthesizer music",
    "drums": "drums and percussion", "singing": "a person singing", "choir": "a choir singing",
    "rock music": "rock music with guitars", "church organ": "a church organ",
    "gunshot": "a gunshot", "gunfire": "machine gun fire", "explosion": "an explosion",
    "scream": "a person screaming", "shouting": "people shouting", "crying": "a person crying",
    "baby crying": "a baby crying", "laughter": "people laughing", "cheering": "a crowd cheering",
    "applause": "applause", "crowd": "a murmuring crowd", "footsteps": "footsteps",
    "door": "a door closing", "knocking": "knocking on a door", "glass breaking": "glass shattering",
    "dishes": "dishes and cutlery clinking", "typing": "typing on a typewriter",
    "telephone": "a telephone ringing", "alarm": "an alarm or siren", "siren": "a police siren",
    "bell": "a bell ringing", "clock": "a clock ticking", "whistle": "a whistle",
    "radio": "voices over a radio", "static": "radio static",
    "car engine": "a car engine", "car horn": "a car horn", "motorcycle": "a motorcycle",
    "train": "a train passing", "airplane": "an airplane overhead", "helicopter": "a helicopter",
    "boat": "a boat engine on water", "horse": "horses galloping",
    "dog": "a dog barking", "birds": "birds singing", "insects": "crickets and insects at night",
    "rain": "rain falling", "thunder": "thunder", "wind": "wind blowing",
    "fire": "a fire crackling", "furnace": "an industrial furnace roaring",
    "machinery": "industrial machinery humming", "river": "a flowing river",
    "waves": "ocean waves", "water splashing": "water splashing",
    "breathing": "heavy breathing", "heartbeat": "a heartbeat", "impact": "a heavy impact or thud",
    "punch": "a punch landing", "rustling": "rustling fabric or leaves", "chains": "metal chains",
    # homes for a film's constant bed, so it does not land on drums/cattle
    "industrial drone": "a low industrial drone", "rumble": "a distant low rumble",
    "hum": "an electrical hum", "room tone": "quiet room tone",
    # homes for score texture, so a horror drone need not become a helicopter
    "drone score": "a low ominous synthesizer drone, horror film music",
    "ambient score": "dark ambient film score, sustained pad",
    "string pad": "sustained string tremolo, suspense music",
    "bass pulse": "a slow sub-bass pulse in a film score",
    "music": "background music",
}
NOT_AMBIENCE = {"speech", "silence"}
MUSIC = {"orchestral score", "piano", "strings", "electronic score", "drums", "singing", "choir",
         "rock music", "church organ", "drone score", "ambient score", "string pad", "bass pulse",
         "music"}

# AST (AudioSet, 527 sigmoid labels) is the gate. Its event labels that are
# worth saying in a film index, mapped onto our names; anything else it
# believes (whale vocalization on a synth drone) stays out of the column.
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000
AST_EVENTS = {
    "Gunshot, gunfire": "gunshot", "Machine gun": "gunfire", "Explosion": "explosion",
    "Screaming": "scream", "Shout": "shouting", "Crying, sobbing": "crying", "Baby cry, infant cry": "baby crying",
    "Laughter": "laughter", "Cheering": "cheering", "Applause": "applause", "Crowd": "crowd",
    "Walk, footsteps": "footsteps", "Door": "door", "Knock": "knocking", "Shatter": "glass breaking",
    "Dishes, pots, and pans": "dishes", "Typewriter": "typing", "Computer keyboard": "typing",
    "Telephone bell ringing": "telephone", "Alarm": "alarm", "Siren": "siren", "Church bell": "bell",
    "Bell": "bell", "Tick-tock": "clock", "Whistle": "whistle", "Radio": "radio", "Static": "static",
    "Car": "car engine", "Vehicle horn, car horn, honking": "car horn", "Motorcycle": "motorcycle",
    "Train": "train", "Aircraft": "airplane", "Helicopter": "helicopter", "Boat, Water vehicle": "boat",
    "Horse": "horse", "Clip-clop": "horse", "Dog": "dog", "Bird": "birds",
    "Owl": "birds", "Insect": "insects", "Cricket": "insects", "Bee, wasp, etc.": "insects", "Rain": "rain", "Thunder": "thunder", "Wind": "wind", "Fire": "fire",
    "Stream": "river", "Ocean": "waves", "Splash, splatter": "water splashing", "Breathing": "breathing",
    "Heart sounds, heartbeat": "heartbeat", "Thump, thud": "impact", "Chink, clink": "chains",
    "Engine": "machinery", "Mechanisms": "machinery", "Hum": "hum", "Mains hum": "hum",
}
# a window AST calls near-silent can still carry a low bed (The Zone of
# Interest's camp drone sits at -52 dBFS and AST says "Silence 0.7"); CLAP
# spreads it over these, and the quiet verdict names it by its best
# non-music member instead of blanking it or calling it score.
LOW_BED = {"rumble", "industrial drone", "hum", "bass pulse", "drone score", "room tone"}
LOW_BED_NAMES = ["rumble", "industrial drone", "hum"]
# the labels CLAP hangs on a drone or a quiet room when nothing fits: these
# need at least a weak AudioSet vote; voices, animals and weather pass through
CONFIRM = {"helicopter", "airplane", "explosion", "gunshot", "gunfire", "alarm", "siren", "train",
           "boat", "motorcycle", "car engine", "machinery", "heartbeat", "thunder", "telephone"}
AST_SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking",
              "Child speech, kid speaking", "Conversation", "Narration, monologue"}
AST_MOOD = {"Scary music": "scary music", "Sad music": "sad music", "Tender music": "tender music",
            "Exciting music": "exciting music", "Happy music": "happy music", "Angry music": "angry music",
            "Opera": "opera", "Choir": "choir", "Chant": "chant", "Orchestra": "orchestral score",
            "Piano": "piano", "Organ": "church organ", "Drum": "drums", "Synthesizer": "electronic score",
            "Singing": "singing", "Techno": "electronic score", "Rock music": "rock music"}


def hms_to_sec(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def read_tsv(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    header = lines[0].split("\t")
    rows = []
    for l in lines[1:]:
        parts = l.split("\t"); parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return header, rows


def write_tsv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r.get(h, "") for h in header) + "\n")
    os.replace(tmp, path)


def decode(media, raw_path, mix="full"):
    """mono s16le at SR. mix="ambience" drops the dialogue channel: on 5.1/7.1
    the off-screen soundtrack lives in L/R and the surrounds while dialogue
    sits in C, so CLAP sees the ambience instead of the words; stereo uses
    the side signal (L-R), which cancels centre-panned dialogue; mono is the
    full mix either way. Returns the mix actually used, or None."""
    import json
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels,channel_layout", "-of", "json", media],
        capture_output=True, text=True).stdout or "{}").get("streams", [{}])[0]
    nch, layout = int(info.get("channels") or 0), info.get("channel_layout", "")
    af, used = ["-ac", "1"], "full"
    if mix == "ambience":
        if nch >= 6 and layout.startswith(("5.1", "7.1")):
            rear = "BL|BR" if layout in ("5.1", "7.1") else "SL|SR"
            a, b = rear.split("|")
            af, used = ["-af", f"pan=mono|c0=0.25*FL+0.25*FR+0.25*{a}+0.25*{b}"], f"no-centre ({layout})"
        elif nch == 2:
            af, used = ["-af", "pan=mono|c0=0.5*FL-0.5*FR"], "side (L-R)"
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", media, "-vn",
                        *af, "-ar", str(SR), "-f", "s16le", raw_path])
    return used if r.returncode == 0 and os.path.getsize(raw_path) > SR * 2 else None


def _vec(x):
    """transformers 4 returns a tensor from get_*_features, 5 an output object."""
    if hasattr(x, "norm"):
        return x
    for k in ("text_embeds", "audio_embeds", "pooler_output"):
        v = getattr(x, k, None)
        if v is not None:
            return v
    return x[0]


class Clap:
    def __init__(self, model):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = ClapModel.from_pretrained(model).to(self.dev).eval()
        self.proc = ClapProcessor.from_pretrained(model)
        with torch.no_grad():
            t = self.proc(text=list(VOCAB.values()), return_tensors="pt", padding=True)
            te = _vec(self.model.get_text_features(**{k: v.to(self.dev) for k, v in t.items()}))
            self.text = te / te.norm(dim=-1, keepdim=True)
        self.scale = (float(self.model.logit_scale_a.detach().exp()) if hasattr(self.model, "logit_scale_a")
                      else 100.0)

    def probs(self, clips):
        """clips: list of float32 arrays at 48 kHz -> (n, vocab) softmax probs."""
        import numpy as np
        out = []
        for a in range(0, len(clips), 16):
            batch = clips[a:a + 16]
            inp = self.proc(audio=batch, sampling_rate=SR, return_tensors="pt", padding=True)
            with self.torch.no_grad():
                ae = _vec(self.model.get_audio_features(**{k: v.to(self.dev) for k, v in inp.items()}))
                ae = ae / ae.norm(dim=-1, keepdim=True)
                logits = self.scale * ae @ self.text.T
                out.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(VOCAB)), np.float32)


def _to16k(x):
    """48 kHz float32 -> 16 kHz: windowed-sinc lowpass at 7 kHz, take every third sample."""
    import numpy as np
    n = 33; k = np.arange(n) - (n - 1) / 2
    h = np.sinc(2 * 7000 / SR * k) * np.hamming(n); h /= h.sum()
    return np.convolve(x, h, mode="same")[::SR // AST_SR].astype(np.float32)


class Ast:
    """Supervised AudioSet tagger: per-label sigmoid probabilities on 16 kHz clips."""
    def __init__(self, model=AST_MODEL):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.fe = ASTFeatureExtractor.from_pretrained(model)
        self.model = ASTForAudioClassification.from_pretrained(model).to(self.dev).eval()
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        self.idx = {l: i for i, l in enumerate(self.labels)}
        missing = [l for l in (*AST_EVENTS, *AST_MOOD, *AST_SPEECH, "Silence", "Music") if l not in self.idx]
        if missing:
            raise SystemExit(f"AST label map is stale, not in {model}: {missing}")

    def probs(self, clips):
        import numpy as np
        out = []
        for a in range(0, len(clips), 16):
            inp = self.fe([_to16k(c) for c in clips[a:a + 16]], sampling_rate=AST_SR, return_tensors="pt")
            with self.torch.no_grad():
                out.append(self.torch.sigmoid(self.model(**{k: v.to(self.dev) for k, v in inp.items()}).logits)
                           .cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), np.float32)

    def confirms(self, p, name, floor=0.1):
        """A CLAP pick that AudioSet also knows (helicopter, explosion) must get at
        least a weak vote from AST; labels AST has no word for pass through."""
        if name not in CONFIRM:
            return True
        back = [l for l, n in AST_EVENTS.items() if n == name]
        return not back or max(p[self.idx[l]] for l in back) >= floor

    def gate(self, p, ev_min=0.25):
        """-> (mode, event, mood, p): mode in quiet/music/effects; event = our name for a
        confident AudioSet event or ""; mood = a music sub-label or ""."""
        sil, mus = p[self.idx["Silence"]], p[self.idx["Music"]]
        best_ev, best_p = "", 0.0
        for l, name in AST_EVENTS.items():
            if p[self.idx[l]] > best_p:
                best_ev, best_p = name, p[self.idx[l]]
        event = best_ev if best_p >= ev_min else ""
        mood, mp = "", 0.0
        for l, name in AST_MOOD.items():
            if p[self.idx[l]] > mp:
                mood, mp = name, p[self.idx[l]]
        mood = mood if mp >= 0.2 else ""
        if event:
            return "effects", event, mood, p
        if mus >= 0.35:
            return "music", "", mood, p
        if sil >= 0.5:
            return "quiet", "", "", p
        return "effects", "", mood, p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir"); ap.add_argument("media")
    ap.add_argument("--model", default=os.environ.get("CRV_CLAP_MODEL", "laion/larger_clap_general"))
    ap.add_argument("--window", type=float, default=5.0,
                    help="± seconds around each frame (both models are trained on 10 s clips)")
    ap.add_argument("--no-ast", action="store_true", help="skip the AudioSet gate (CLAP only)")
    ap.add_argument("--min-prob", type=float, default=0.15, help="label probability floor")
    ap.add_argument("--min-prom", type=float, default=0.10,
                    help="how far above the film's median for that label")
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (a probe)")
    a = ap.parse_args()
    import numpy as np

    idx = os.path.join(a.out_dir, "INDEX.tsv")
    if not os.path.exists(idx) or not os.path.exists(a.media):
        print("✗ need INDEX.tsv and the media file", file=sys.stderr); sys.exit(1)
    header, rows = read_tsv(idx)
    if a.limit:
        rows = rows[:a.limit]
    ev_path = os.path.join(a.out_dir, "events.tsv")
    ev_header, events = read_tsv(ev_path) if os.path.exists(ev_path) else (None, [])
    # an SDH cue track names the sounds itself; it leads, CLAP is demoted to
    # AST-confirmed events (Alien: 'cattle' x786 for the Nostromo hum while
    # the cues carried 'chains', 'rain', 'fire', 'explosion')
    import enhance_process as E
    srt = os.path.join(a.out_dir, "dialogue.srt")
    sdh, is_sdh = E.sdh_cues(E.parse_srt(srt)) if os.path.exists(srt) else ([], False)
    sdh_starts = [c[0] for c in sdh]
    def sdh_at(t0s, t1s):
        """descriptions live anywhere in [t0s, t1s], in cue order, deduped"""
        import bisect
        out = []
        for s_, e_, g in sdh[max(0, bisect.bisect_left(sdh_starts, t0s - 30)):]:
            if s_ > t1s: break
            if e_ >= t0s and g not in out: out.append(g)
        return out
    if is_sdh:
        print(f"• SDH cue track: {len(sdh)} sound descriptions lead the ambience column", flush=True)

    t0 = time.perf_counter()
    fd, raw_amb = tempfile.mkstemp(suffix=".s16le"); os.close(fd)
    fd, raw_full = tempfile.mkstemp(suffix=".s16le"); os.close(fd)
    bed = []
    try:
        mix = decode(a.media, raw_amb, "ambience")
        if not mix or not decode(a.media, raw_full, "full"):
            print("✗ audio decode failed", file=sys.stderr); sys.exit(1)
        amb = np.memmap(raw_amb, dtype=np.int16, mode="r")
        full = np.memmap(raw_full, dtype=np.int16, mode="r")
        def clip(pcm, t0s, t1s):
            n = pcm.size
            s0 = max(0, int(t0s * SR)); s1 = min(n, int(t1s * SR))
            x = pcm[s0:s1].astype(np.float32) / 32768.0
            if x.size < SR:                      # CLAP wants >= 1 s
                x = np.pad(x, (0, SR - x.size))
            return x
        print(f"• decoded {full.size / SR / 60:.1f} min (ambience mix: {mix}); loading {a.model}...",
              flush=True)
        clap = Clap(a.model)
        ast = None if a.no_ast else Ast()
        labels = list(VOCAB)
        music_j = {j for j, l in enumerate(labels) if l in MUSIC}
        print(f"• {len(rows)} frames + {len(events)} events through CLAP"
              f"{'' if a.no_ast else ' + AST'} on {clap.dev}...", flush=True)

        # frames: ±window on the ambience mix. Scored by prominence over the
        # film's own median per label: a constant bed (camp rumble, score)
        # would otherwise be the top label on most frames and the transient
        # on top of it (a gunshot) would never show. The bed is reported once.
        clips = [clip(amb, hms_to_sec(r["time"]) - a.window, hms_to_sec(r["time"]) + a.window)
                 for r in rows]
        P = clap.probs(clips)
        gates = [ast.gate(p) for p in ast.probs(clips)] if ast else [("effects", "", "", None)] * len(clips)
        med = np.median(P, axis=0) if len(P) else np.zeros(len(labels))
        # the bed: labels confident on a quarter of the film or more. Reported
        # once in MANIFEST and kept out of the per-frame column, which then
        # carries what departs from the bed (The Zone of Interest: the camp
        # drone is the bed; the garden's birds, the piano, the party's
        # applause and the river are the departures).
        freq = (P >= a.min_prob).mean(axis=0) if len(P) else np.zeros(len(labels))
        bed_idx = [j for j in np.argsort(freq)[::-1] if freq[j] >= 0.25 and labels[j] not in NOT_AMBIENCE]
        bed = [labels[j] for j in bed_idx]
        prom = P - med
        n_tagged = 0
        n_mode = {"quiet": 0, "music": 0, "effects": 0}
        for r, p, q, (mode, event, mood, pa) in zip(rows, P, prom, gates):
            n_mode[mode] += 1
            order = np.argsort(q)[::-1]
            picks = [labels[j] for j in order[:6]
                     if labels[j] not in NOT_AMBIENCE and j not in bed_idx
                     and p[j] >= a.min_prob and q[j] >= a.min_prom
                     and (mode != "music" or j in music_j)
                     and (pa is None or ast.confirms(pa, labels[j]))]
            if mode == "quiet":
                low = sum(p[j] for j, l in enumerate(labels) if l in LOW_BED)
                picks = ([max(LOW_BED_NAMES, key=lambda l: p[labels.index(l)])]
                         if low >= 0.3 and not any(l in bed for l in LOW_BED_NAMES) else [])
            if mode == "music" and mood and mood not in bed and mood not in picks:
                picks.insert(0, mood)
            if event and event not in bed:
                picks = [event] + [x for x in picks if x != event]
            if is_sdh:
                t_ = hms_to_sec(r["time"])
                picks = sdh_at(t_ - 0.3, t_ + 0.3) + ([event] if event else [])
            r["ambience"] = ", ".join(picks[:2])
            n_tagged += bool(picks)
        # events: their own span on the full mix (they were detected there)
        if events:
            clips = []
            for e in events:
                t, d = float(e["t"]), float(e["dur"])
                pad = max(0.0, (2 * a.window - d) / 2)
                clips.append(clip(full, t - pad, t + d + pad))
            Pe = clap.probs(clips)
            ge = [ast.gate(p) for p in ast.probs(clips)] if ast else [("effects", "", "", None)] * len(clips)
            for e, p, (mode, event, mood, pa) in zip(events, Pe, ge):
                if is_sdh:
                    t, d = float(e["t"]), float(e["dur"])
                    live = sdh_at(t, t + max(d, 0.5))
                    e["tag"] = live[0] if live else (event or ""); continue
                if event:
                    e["tag"] = event; continue
                order = [j for j in np.argsort(p)[::-1] if (mode != "music" or labels[j] in MUSIC)
                         and (pa is None or ast.confirms(pa, labels[j]))]
                j = order[0] if order else int(np.argmax(p))
                if mode == "music":
                    e["tag"] = mood or labels[j]
                else:
                    e["tag"] = labels[j] if p[j] >= a.min_prob else ""
    finally:
        for f_ in (raw_amb, raw_full):
            try: os.remove(f_)
            except OSError: pass
    secs = time.perf_counter() - t0

    if "ambience" not in header:
        header.append("ambience")
    if a.limit:                                  # a probe: only report
        from collections import Counter
        print("  bed:", bed, " modes:", n_mode)
        print("  ambience:", Counter(r["ambience"] for r in rows).most_common(12))
        for r in rows[:: max(1, len(rows) // 25)]:
            print(f"    {r['time'][:8]}  {r['ambience']:<28} {r['dialogue'][:40]!r}")
        if events:
            print("  events:", Counter(e.get("tag", "") for e in events).most_common(10))
        print(f"  {secs:.0f}s"); return
    write_tsv(idx, header, rows)
    if events:
        if "tag" not in ev_header: ev_header.append("tag")
        write_tsv(ev_path, ev_header, events)
        sp = os.path.join(a.out_dir, "sounds.tsv")
        if os.path.exists(sp):
            from collections import Counter, defaultdict
            per = defaultdict(Counter)
            for e in events:
                if e.get("tag"): per[e["label"]][e["tag"]] += 1
            sh, srows = read_tsv(sp)
            if "tag" not in sh: sh.append("tag")
            for r in srows:
                c = per.get(r["label"])
                r["tag"] = c.most_common(1)[0][0] if c else ""
            write_tsv(sp, sh, srows)

    man = os.path.join(a.out_dir, "MANIFEST.txt")
    txt = open(man, encoding="utf-8").read() if os.path.exists(man) else ""
    title = os.path.basename(os.path.normpath(a.out_dir))
    m = re.search(r"(?m)^source: (.*)$", txt)
    if m: title = os.path.basename(m.group(1).strip())
    E.write_sync_md(os.path.join(a.out_dir, "sync.md"), title, rows)
    gate_note = ("CLAP only" if a.no_ast else
                 f"gated by {AST_MODEL} (quiet {n_mode['quiet']} / music {n_mode['music']} / "
                 f"effects {n_mode['effects']} frames)")
    if is_sdh:
        gate_note += (f"; SDH cues lead the ambience column ({len(sdh)} sound descriptions), "
                      f"CLAP demoted to AST-confirmed events")
    line = (f"sound tags: {a.model} zero-shot over {len(VOCAB)} film-sound labels on the {mix} mix, "
            f"{gate_note}; bed: {', '.join(bed) or 'none'}; ambience on "
            f"{n_tagged}/{len(rows)} frames (±{a.window:g} s, p ≥ {a.min_prob:g}, "
            f"prominence ≥ {a.min_prom:g}), "
            f"{sum(1 for e in events if e.get('tag'))}/{len(events)} events tagged; {secs:.0f}s")
    if txt:
        txt = re.sub(r"(?m)^sound tags: .*\n", "", txt)
        if not txt.endswith("\n"): txt += "\n"
        open(man, "w", encoding="utf-8").write(txt + line + "\n")
    print(f"✓ {title}: {line}")


if __name__ == "__main__":
    main()
