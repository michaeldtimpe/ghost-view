#!/usr/bin/env python3
"""On a package whose cue track is SDH (sound descriptions in brackets), let
those descriptions drive the `ambience` column and the event tags, and drop
the CLAP guesses -- without the audio, so it works on a finished package.

    sdh_sounds.py <out_dir>

sound_tags.py does the same on a fresh run, keeping AST-confirmed events next
to the descriptions; here the audio is gone, and which CLAP labels had an AST
vote was never stored, so on an SDH source the column is the cues' words
only. A package whose cues are not SDH is left untouched. Reversible only by
re-running sound_tags.py with the source.
"""
import argparse, bisect, os
import enhance_process as E
import pkg


def run(out_dir, log=print):
    cues = pkg.cues(out_dir)
    sdh, is_sdh = E.sdh_cues(cues)
    if not is_sdh:
        log(f"• {os.path.basename(out_dir)}: cue track is not SDH ({len(sdh)} descriptions in "
            f"{len(cues)} cues); nothing changed")
        return 0
    header, rows = pkg.read_index(out_dir)
    starts = [c[0] for c in sdh]

    def live(t0, t1):
        out = []
        for s, e, g in sdh[max(0, bisect.bisect_left(starts, t0 - 30)):]:
            if s > t1: break
            if e >= t0 and g not in out: out.append(g)
        return out

    pkg.add_column(header, "ambience")
    n = 0
    for r in rows:
        t = pkg.sec(r["time"])
        r["ambience"] = ", ".join(live(t - 0.3, t + 0.3)[:2])
        n += bool(r["ambience"])
    pkg.write_index(out_dir, header, rows)

    ev_path = os.path.join(out_dir, "events.tsv")
    n_ev = 0
    if os.path.exists(ev_path):
        lines = open(ev_path, encoding="utf-8").read().splitlines()
        eh = lines[0].split("\t")
        if "tag" not in eh: eh.append("tag")
        evs = []
        for l in lines[1:]:
            d = dict(zip(eh, l.split("\t") + [""] * len(eh)))
            t, dur = float(d["t"]), float(d["dur"])
            got = live(t, t + max(dur, 0.5))
            d["tag"] = got[0] if got else ""
            n_ev += bool(d["tag"]); evs.append(d)
        tmp = ev_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\t".join(eh) + "\n")
            for d in evs:
                f.write("\t".join(d.get(k, "") for k in eh) + "\n")
        os.replace(tmp, ev_path)
        # sounds.tsv's per-label tag: the commonest tag of that label's events
        sp = os.path.join(out_dir, "sounds.tsv")
        if os.path.exists(sp):
            from collections import Counter, defaultdict
            per = defaultdict(Counter)
            for d in evs:
                if d["tag"]: per[d["label"]][d["tag"]] += 1
            sl = open(sp, encoding="utf-8").read().splitlines()
            sh = sl[0].split("\t")
            if "tag" not in sh: sh.append("tag")
            tmp = sp + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\t".join(sh) + "\n")
                for l in sl[1:]:
                    d = dict(zip(sh, l.split("\t") + [""] * len(sh)))
                    c = per.get(d.get("label"))
                    d["tag"] = c.most_common(1)[0][0] if c else ""
                    f.write("\t".join(d.get(k, "") for k in sh) + "\n")
            os.replace(tmp, sp)

    old = pkg.manifest_get(out_dir, "sound tags")
    note = (f"SDH cue track ({len(sdh)} sound descriptions) drives the ambience column "
            f"({n}/{len(rows)} frames) and the event tags ({n_ev} events); CLAP labels dropped")
    if old and "SDH cue track" not in old:
        note += f" [was: {old}]"
    pkg.manifest_set(out_dir, "sound tags", note)
    pkg.rebuild_sync(out_dir)
    log(f"✓ {os.path.basename(out_dir)}: {note}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    run(a.out_dir)


if __name__ == "__main__":
    main()
