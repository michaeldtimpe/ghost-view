#!/usr/bin/env python3
"""Record where a package's source file lives and what it was.

Usage: origin_patch.py <out_dir> --origin <remote path>
                       [--duration HH:MM:SS] [--bytes N] [--name basename]

Sets MANIFEST `origin:` (after `source:`) and, when --duration or --bytes is
given, `container: <HH:MM:SS> · <bytes> bytes · <name>` (after `origin:`), the
engine's own format. --name defaults to the basename of --origin.
"""
import argparse, os, sys
import pkg


def run(out_dir, origin, duration=None, nbytes=None, name=None):
    pkg.manifest_set(out_dir, "origin", origin, after="source")
    if duration or nbytes is not None:
        name = name or os.path.basename(origin)
        parts = [duration or "?", f"{nbytes if nbytes is not None else '?'} bytes", name]
        pkg.manifest_set(out_dir, "container", " · ".join(parts), after="origin")
    print(f"origin: {origin}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--origin", required=True)
    ap.add_argument("--duration")
    ap.add_argument("--bytes", type=int)
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    return run(a.out_dir, a.origin, a.duration, a.bytes, a.name)


if __name__ == "__main__":
    sys.exit(main())
