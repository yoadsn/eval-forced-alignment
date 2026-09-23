"""End a word where the sound stops, not where CTC stops being sure of the last letter.

    python eval/tail_extend.py --run data/eval_runs/ivrit-ai-corrected \
        --frames data/unclaimed/frames.jsonl

All of MMS's word-end error is in the 24% of words followed by a pause, where it finishes
about 60 ms early. Inside continuous speech it is already near the human floor. The reason
is mechanical: CTC ends a word at the last frame it is confident about the final letter,
while a person ends it where the sound dies away -- the same instant when the next word
starts immediately, and 60 ms apart when nothing follows.

A constant cannot fix that, because how long a word rings on is a property of the word.
So instead: walk forward from MMS's end while the model still says something is being said
(1 - P(blank) above a threshold), stopping at the next word or at a cap. One threshold and
one cap, both fitted on half the clips and measured on the other half.

Needs the per-frame blank probabilities: run eval/aligners/ctc.py with --frames.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_bf():
    spec = importlib.util.spec_from_file_location("bf", ROOT / "eval" / "boundary_fix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extend(row, frames, theta: float, cap: float) -> float:
    """Where the word ends once the trailing sound is included."""
    blank = frames["blank"]
    ratio = frames["ratio"]
    limit = row["end"] + cap
    if row["next_start"] is not None:
        limit = min(limit, row["next_start"])
    i = int(round(row["end"] / ratio))
    while i + 1 < len(blank) and (i + 1) * ratio <= limit and (1.0 - blank[i]) > theta:
        i += 1
    return max(row["end"], min(i * ratio, limit))


def err(rows, frames, theta, cap) -> list[float]:
    return [abs(extend(r, frames[r["clip_key"]], theta, cap) - r["h_end"]) * 1000 for r in rows]


def fit(rows, frames, thetas, caps) -> tuple[float, float]:
    """The threshold and cap with the lowest mean end error on these clips."""
    best = None
    for theta in thetas:
        for cap in caps:
            e = err(rows, frames, theta, cap)
            score = sum(e) / len(e)
            if best is None or score < best[0]:
                best = (score, theta, cap)
    return best[1], best[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    args = p.parse_args()

    from hebrew_training.aligner_eval import pair_words, summarise  # noqa: F401

    bf = load_bf()
    rows = bf.build(args.run, "mms", {"probe"})
    frames = {json.loads(line)["id"]: json.loads(line)
              for line in args.frames.read_text(encoding="utf-8").splitlines() if line.strip()}
    for r in rows:
        r["clip_key"] = f"{r['clip']}#{r['who']}"
    rows = [r for r in rows if r["clip_key"] in frames]

    thetas = [round(0.05 * i, 2) for i in range(1, 19)]
    caps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    clips = sorted({r["clip"] for r in rows})
    half = set(clips[::2])
    held, chosen = [], []
    for train_in_half in (True, False):
        train = [r for r in rows if (r["clip"] in half) == train_in_half]
        test = [r for r in rows if (r["clip"] in half) != train_in_half]
        theta, cap = fit(train, frames, thetas, caps)
        chosen.append((theta, cap))
        held += err(test, frames, theta, cap)

    raw = [abs(r["end"] - r["h_end"]) * 1000 for r in rows]
    print(f"fitted: {chosen}")
    print(f"{'WORD ENDS':<26}{'median':>9}{'p90':>9}{'<=25ms':>9}{'<=50ms':>9}")
    for name, vals in (("mms, raw", raw), ("+ extend to the sound", held)):
        s = summarise(vals)
        print(f"  {name:<24}{s['median_ms']:>9}{s['p90_ms']:>9}{s['within_25ms']:>9}{s['within_50ms']:>9}")

    print(f"\n{'by what follows the word':<28}{'n':>5}{'raw p90':>9}{'fixed p90':>11}"
          f"{'raw <=50':>10}{'fixed <=50':>12}")
    theta, cap = chosen[0]
    for name, sel in (("touching the next word", [r for r in rows if r["gap_after"] < 0.02]),
                      ("20-100 ms of space", [r for r in rows if 0.02 <= r["gap_after"] < 0.1]),
                      ("100-300 ms of space", [r for r in rows if 0.1 <= r["gap_after"] < 0.3]),
                      ("300 ms or more", [r for r in rows if r["gap_after"] >= 0.3])):
        a = summarise([abs(r["end"] - r["h_end"]) * 1000 for r in sel])
        b = summarise(err(sel, frames, theta, cap))
        print(f"  {name:<26}{len(sel):>5}{a['p90_ms']:>9}{b['p90_ms']:>11}"
              f"{a['within_50ms']:>10}{b['within_50ms']:>12}")


if __name__ == "__main__":
    main()
