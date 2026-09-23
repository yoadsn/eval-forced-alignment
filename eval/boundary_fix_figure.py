"""Draw what the boundary correction does, why the gap fraction fails, and what it buys.

    python eval/boundary_fix_figure.py --run data/eval_runs/ivrit-ai-corrected --out fig.png

Three panels, meant to be read left to right:

  1. one real clip, with the audio, where MMS put the words, where the people put them, and
     where the correction puts them
  2. how far the human boundary sits from MMS's, bucketed by how much free space there was
     -- flat, which is why a fraction of that space is the wrong thing to take
  3. the error before and after, as a cumulative curve, so the tail is visible

No Hebrew in the labels: matplotlib reverses it without a bidi shaper, which would be worse
than leaving it out.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INK, SUB, GRID = "#1b1f24", "#6b7280", "#e5e7eb"
MMS, HUM, FIX = "#2563eb", "#15803d", "#b45309"


def load_bf():
    spec = importlib.util.spec_from_file_location("bf", ROOT / "eval" / "boundary_fix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick(rows, bf, a_shift, b_shift):
    """A run of consecutive words that behaves the way the whole set behaves.

    Picking the clip with the biggest pause gave a run where the humans marked *later* than
    MMS at every start -- true of that run, and the opposite of what the correction is for.
    An illustration has to be typical or it teaches the wrong thing, so the run chosen is the
    one whose own bias is closest to the median bias being corrected.
    """
    best = None
    for clip in {r["clip"] for r in rows}:
        sel = [r for r in rows if r["clip"] == clip]
        sel = sorted({(r["start"], r["end"]): r for r in sel}.values(), key=lambda r: r["start"])
        for i in range(len(sel) - 3):
            run = sel[i:i + 4]
            span = run[-1]["end"] - run[0]["start"]
            if not (1.5 < span < 3.2):
                continue
            off = sum(abs((r["start"] - r["h_start"]) - a_shift)
                      + abs((r["h_end"] - r["end"]) - b_shift) for r in run) / len(run)
            if best is None or off < best[0]:
                best = (off, clip, run)
    return best[1], best[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ivrit-ai"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    bf = load_bf()
    rows = bf.build(args.run, "mms", {"probe"})
    a_shift, b_shift = bf.fit(rows, "shift")
    a_frac, b_frac = bf.fit(rows, "fraction")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a in ax:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=SUB, labelsize=9)

    # ---- 1. a real clip
    clip, run = pick(rows, bf, a_shift, b_shift)
    entry = next(json.loads(l) for l in (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["id"] == clip)
    wav, rate = sf.read(args.dataset / entry["audio"], dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    t0, t1 = run[0]["start"] - 0.25, run[-1]["end"] + 0.25
    seg = wav[int(t0 * rate):int(t1 * rate)]
    t = np.linspace(t0, t1, len(seg))
    ax[0].plot(t, seg / (abs(seg).max() or 1) * 0.30 + 0.80, color="#9ca3af", linewidth=0.5)
    lanes = [(0.52, MMS, "MMS"), (0.34, FIX, "corrected"), (0.16, HUM, "people")]
    for r in run:
        s_fix, e_fix = bf.apply(r, "shift", a_shift, b_shift)
        for (y, colour, _), (s, e) in zip(lanes, [(r["start"], r["end"]), (s_fix, e_fix),
                                                  (r["h_start"], r["h_end"])]):
            ax[0].add_patch(plt.Rectangle((s, y), e - s, 0.11, color=colour, alpha=.85))
    for y, colour, name in lanes:
        ax[0].text(t0, y + 0.125, name, color=colour, fontsize=9, va="bottom")
    ax[0].set_ylim(0, 1.15)
    ax[0].set_xlim(t0, t1)
    ax[0].get_yaxis().set_visible(False)
    ax[0].spines["left"].set_visible(False)
    ax[0].set_xlabel("seconds", color=SUB, fontsize=10)
    ax[0].set_title("A typical run: MMS starts late, ends early", color=INK, fontsize=12, loc="left")

    # ---- 2. why a fraction of the gap is the wrong thing to take
    buckets = [(0, .025), (.025, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 9)]
    xs, taken, would = [], [], []
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r["gap_before"] < hi and r["gap_before"] > 0.001]
        if len(sel) < 15:
            continue
        xs.append(f"{lo * 1000:.0f}-{hi * 1000:.0f}" if hi < 9 else "400+")
        taken.append(statistics.median([(r["start"] - r["h_start"]) * 1000 for r in sel]))
        would.append(a_frac * statistics.median([r["gap_before"] * 1000 for r in sel]))
    pos = range(len(xs))
    ax[1].bar([p - .2 for p in pos], taken, width=.4, color=HUM, label="what people actually take")
    ax[1].bar([p + .2 for p in pos], would, width=.4, color=MMS,
              label=f"what {a_frac:.0%} of the gap would take")
    ax[1].set_xticks(list(pos))
    ax[1].set_xticklabels(xs, fontsize=9)
    ax[1].set_xlabel("free space before the word (ms)", color=SUB, fontsize=10)
    ax[1].set_ylabel("ms moved earlier", color=SUB, fontsize=10)
    ax[1].grid(True, axis="y", color=GRID, linewidth=.8)
    ax[1].set_axisbelow(True)
    ax[1].legend(frameon=False, fontsize=9, labelcolor=SUB)
    ax[1].set_title("People move it ~30 ms however big the pause", color=INK, fontsize=12, loc="left")

    # ---- 3. the pay-off
    raw = sorted(bf.errors(rows))
    fixed = sorted(bf.two_fold(rows, "shift"))
    frac = sorted(bf.two_fold(rows, "fraction"))
    for vals, colour, name in ((raw, MMS, "MMS"), (frac, "#9ca3af", "+ fraction of the gap"),
                               (fixed, FIX, "+ 30 ms, capped at the neighbour")):
        ax[2].plot(vals, np.linspace(0, 100, len(vals)), color=colour, linewidth=2.2, label=name)
    for vals, colour in ((raw, MMS), (fixed, FIX)):
        ax[2].plot([vals[int(.9 * len(vals))]], [90], "o", color=colour, markersize=6)
    ax[2].axhline(90, color=GRID, linewidth=1, linestyle="--")
    ax[2].text(300, 91, "p90", color=SUB, fontsize=9)
    ax[2].set_xlim(0, 320)
    ax[2].set_xlabel("distance from where a person marked it (ms)", color=SUB, fontsize=10)
    ax[2].set_ylabel("% of boundaries within", color=SUB, fontsize=10)
    ax[2].grid(True, color=GRID, linewidth=.8)
    ax[2].set_axisbelow(True)
    ax[2].legend(frameon=False, fontsize=9, labelcolor=SUB, loc="lower right")
    ax[2].set_title("Tail: 90 ms to 80 ms", color=INK, fontsize=12, loc="left")

    fig.suptitle("Correcting MMS word boundaries  ·  held out, 72 clips",
                 color=INK, fontsize=13, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"shift: start -{a_shift * 1000:.0f} ms, end +{b_shift * 1000:.0f} ms -> {args.out}")


if __name__ == "__main__":
    main()
