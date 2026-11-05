#!/usr/bin/env python3
"""Does the vision pre-pass change what Claude gets right?
Same film, same ten structural questions, two conditions:
  A  plain INDEX.tsv (the `shot` column stripped) + sounds.tsv + MANIFEST
  B  the same + shots.tsv
Each condition is one `claude -p` call (tools disallowed, data inline).
Answers land in <out>/bench_A.md and bench_B.md for hand grading.
usage: vlm_bench.py <crv_folder> <out_dir> [--model opus]"""
import os, subprocess, sys, time

folder, out = sys.argv[1], sys.argv[2]
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "opus"
os.makedirs(out, exist_ok=True)

QUESTIONS = """\
Answer each question in 1-3 sentences, from the data only. If the data cannot
answer, say "not determinable from the data" rather than guessing. Give
timestamps where you can.

1. How long does the film run before the first visible image, and what is the first visible scene?
2. Where is the family's home situated: what adjoins the garden, and what can be seen beyond it?
3. Some night sequences are shot in an unusual visual style. Describe the style and what happens in them.
4. Which relative comes to visit, and how does the visit end?
5. What recurring off-screen sounds accompany the domestic scenes?
6. What professional news does the father receive around the middle of the film, and how does the family react?
7. Describe the scene at the river: what happens, and why does the family leave the water quickly?
8. What is unusual about the sequence near the very end, after the father walks down a stairwell?
9. What is the dominant camera style of the film?
10. What readable on-screen text appears anywhere in the film?
"""

def read(name):
    p = os.path.join(folder, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

idx_lines = read("INDEX.tsv").splitlines()
hdr = idx_lines[0].split("\t")
def strip_shot(lines):
    if "shot" not in hdr:
        return "\n".join(lines)
    j = hdr.index("shot")
    return "\n".join("\t".join(c for k, c in enumerate(l.split("\t")) if k != j) for l in lines)
index_plain = strip_shot(idx_lines)
index_shot = "\n".join(idx_lines)
manifest = "\n".join(l for l in read("MANIFEST.txt").splitlines() if not l.startswith(("vision:", "source:", "frames dir:")))
sounds = read("sounds.tsv")
shots = read("shots.tsv")

PRE = ("You are reading a machine-made index of a feature film. You cannot see the frames; "
       "you have only the tables below. Columns are documented in the first table's header. "
       "`dialogue` is the subtitle line live at that frame; `sound` names non-speech events; "
       "`impact`/`field`/`voice`/`pan` describe the 5.1 mix; `cut` marks a hard cut.\n\n")

conds = {
    "A": PRE + f"MANIFEST:\n{manifest}\n\nsounds.tsv:\n{sounds}\n\nINDEX.tsv:\n{index_plain}\n\n",
    "B": PRE + ("shots.tsv is a shot list written by a vision model that looked at one frame per shot "
                "(setting, people count, shot scale, action, light, readable text); INDEX.tsv's `shot` "
                "column keys into it.\n\n")
         + f"MANIFEST:\n{manifest}\n\nsounds.tsv:\n{sounds}\n\nshots.tsv:\n{shots}\n\nINDEX.tsv:\n{index_shot}\n\n",
}
for name, data in conds.items():
    prompt = data + QUESTIONS
    print(f"condition {name}: {len(prompt)//1000} KB prompt", flush=True)
    t0 = time.perf_counter()
    # prompt on stdin: --disallowedTools is variadic and would swallow it;
    # no --bare: it skips the Keychain credential lookup ("Not logged in")
    r = subprocess.run(["claude", "-p", "--no-session-persistence", "--model", model,
                        "--output-format", "text",
                        "--disallowedTools", "Bash", "Read", "Edit", "Write", "Glob", "Grep",
                        "WebSearch", "WebFetch", "Agent", "Task"],
                       input=prompt, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    body = r.stdout.strip() or f"(no output; rc={r.returncode})\n{r.stderr[-800:]}"
    open(os.path.join(out, f"bench_{name}.md"), "w", encoding="utf-8").write(
        f"# condition {name} ({model}, {dt:.0f}s, {len(prompt)//1000} KB)\n\n{body}\n")
    print(f"  done in {dt:.0f}s, {len(body)} chars", flush=True)
