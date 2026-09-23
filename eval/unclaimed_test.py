"""Does unclaimed speech find the words a transcript is missing?

    python eval/unclaimed_test.py --gaps data/unclaimed/original.jsonl --marks gold.jsonl

The 72 clips aligned on their *original* transcripts are the real corrupted condition: the
words annotators added are words the transcript genuinely lacks, and their human timings say
exactly where each one is. So every unclaimed stretch can be labelled without inventing
anything -- it either overlaps a word somebody added, or it does not.

The floor it has to clear was measured on the same clips aligned on the *corrected*
transcripts, where nothing is missing by construction, and is reported alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gaps", type=Path, required=True, help="Gaps from the ORIGINAL transcript.")
    p.add_argument("--floor", type=Path, help="Gaps from the corrected transcript, for the floor.")
    p.add_argument("--marks", type=Path, required=True)
    p.add_argument("--exclude", action="append", default=["probe"])
    args = p.parse_args()

    marks = [m for m in load_jsonl(args.marks) if m.get("annotator") not in set(args.exclude)]
    added: dict[str, list] = {}
    for m in marks:
        for w in m["words"]:
            if w.get("added"):
                added.setdefault(m["id"], []).append((w["start"], w["end"], w["word"]))

    rows = load_jsonl(args.gaps)
    hit, miss, noise = [], [], []
    for r in rows:
        want = added.get(r["id"].split("#")[0], [])
        claimed = set()
        for g in r["gaps"]:
            over = [i for i, (s, e, _) in enumerate(want)
                    if overlap(g["start"], g["end"], s, e) > 0.5 * (e - s)]
            if over:
                hit.append(g)
                claimed.update(over)
            else:
                noise.append(g)
        miss += [want[i] for i in range(len(want)) if i not in claimed]

    print(f"{len(rows)} clips, {sum(len(v) for v in added.values())} words annotators added")
    print(f"  {len(hit)} unclaimed stretches sit on one of those words")
    print(f"  {len(miss)} added words have no unclaimed stretch at all -- the aligner covered "
          f"them with a neighbour")
    print(f"  {len(noise)} unclaimed stretches sit elsewhere (the false-alarm population)\n")

    def q(vals, f):
        s = sorted(vals)
        return s[min(int(f * len(s)), len(s) - 1)] if s else 0.0

    hs = [g["speech"] for g in hit]
    ns = [g["speech"] for g in noise]
    print(f"speech mass:  on a missing word  median {q(hs, .5) * 1000:>5.0f} ms   "
          f"elsewhere median {q(ns, .5) * 1000:>5.0f} ms")
    if args.floor and args.floor.exists():
        fl = [g["speech"] for r in load_jsonl(args.floor) for g in r["gaps"]]
        print(f"floor (correct transcripts): 95th pct {q(fl, .95) * 1000:.0f} ms, "
              f"max {max(fl) * 1000:.0f} ms")

    print("\nthreshold on speech mass, against the stretches that are not on a missing word:")
    print(f"{'threshold':>10}{'found':>9}{'of added words':>16}{'false alarms':>14}")
    for t in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        found = sum(1 for g in hit if g["speech"] >= t)
        total = len(hit) + len(miss)
        fa = sum(1 for g in noise if g["speech"] >= t)
        print(f"{t * 1000:>8.0f}ms{found:>9}{found / total * 100:>15.1f}%{fa:>14}")


if __name__ == "__main__":
    main()
