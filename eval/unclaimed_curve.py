"""Sweep the unclaimed-speech threshold and draw what it costs.

    python eval/unclaimed_curve.py --gaps data/unclaimed/original.jsonl \
        --marks gold.jsonl --out data/unclaimed/curve

Three panels, because the choice of threshold is three questions:

  1. Are the two populations separable at all? The speech mass of stretches that sit on a
     word the transcript is missing, against every other stretch.
  2. What does each threshold catch, and what does it cost? Both rates against the
     threshold, which is the panel to read a number off.
  3. The same as one curve, catch against false alarms, with the thresholds marked.

Rows, not words, are the unit: the decision is whether to keep a row. A row counts as caught
when any of its unclaimed stretches clears the threshold, and as a false alarm when it does
that while having no missing word at all.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gaps", type=Path, required=True)
    p.add_argument("--marks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude", action="append", default=["probe"])
    args = p.parse_args()

    marks = [m for m in load_jsonl(args.marks) if m.get("annotator") not in set(args.exclude)]
    added: dict[str, list] = {}
    for m in marks:
        for w in m["words"]:
            if w.get("added"):
                added.setdefault(m["id"], []).append((w["start"], w["end"]))

    rows = load_jsonl(args.gaps)
    on_missing, elsewhere = [], []
    per_row = []
    for r in rows:
        want = added.get(r["id"].split("#")[0], [])
        best_on, best_off = 0.0, 0.0
        for g in r["gaps"]:
            hit = any(overlap(g["start"], g["end"], s, e) > 0.5 * (e - s) for s, e in want)
            (on_missing if hit else elsewhere).append(g["speech"])
            if hit:
                best_on = max(best_on, g["speech"])
            else:
                best_off = max(best_off, g["speech"])
        per_row.append({"has_missing": bool(want), "loudest": max(best_on, best_off)})

    pos = [r["loudest"] for r in per_row if r["has_missing"]]
    neg = [r["loudest"] for r in per_row if not r["has_missing"]]
    steps = [i / 1000 for i in range(0, 301, 2)]
    curve = [{"threshold_ms": t * 1000,
              "catch_pct": round(100 * sum(1 for v in pos if v >= t) / len(pos), 1),
              "false_pct": round(100 * sum(1 for v in neg if v >= t) / len(neg), 1)}
             for t in steps]

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "curve.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, sub, grid = "#1b1f24", "#6b7280", "#e5e7eb"
    hit_c, miss_c = "#b45309", "#2563eb"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a in ax:
        a.grid(True, color=grid, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(grid)
        a.tick_params(colors=sub, labelsize=9)

    bins = [i * 10 for i in range(0, 27)]
    ax[0].hist([v * 1000 for v in elsewhere], bins=bins, color=miss_c, alpha=.75,
               label=f"anywhere else (n={len(elsewhere)})")
    ax[0].hist([v * 1000 for v in on_missing], bins=bins, color=hit_c, alpha=.85,
               label=f"on a missing word (n={len(on_missing)})")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("speech in the unclaimed stretch (ms)", color=sub, fontsize=10)
    ax[0].set_ylabel("stretches (log)", color=sub, fontsize=10)
    ax[0].set_title("Are they separable?", color=ink, fontsize=12, loc="left")
    ax[0].legend(frameon=False, fontsize=9, labelcolor=sub)

    ts = [c["threshold_ms"] for c in curve]
    ax[1].plot(ts, [c["catch_pct"] for c in curve], color=hit_c, linewidth=2.2,
               label="rows with a missing word, flagged")
    ax[1].plot(ts, [c["false_pct"] for c in curve], color=miss_c, linewidth=2.2,
               label="rows without one, flagged")
    ax[1].set_xlabel("threshold (ms of unclaimed speech)", color=sub, fontsize=10)
    ax[1].set_ylabel("% of rows", color=sub, fontsize=10)
    ax[1].set_title("What each threshold costs", color=ink, fontsize=12, loc="left")
    ax[1].legend(frameon=False, fontsize=9, labelcolor=sub)

    ax[2].plot([c["false_pct"] for c in curve], [c["catch_pct"] for c in curve],
               color=ink, linewidth=2.2)
    ax[2].plot([0, 100], [0, 100], color=grid, linewidth=1, linestyle="--")
    for mark in (50, 100, 150, 200):
        c = min(curve, key=lambda c: abs(c["threshold_ms"] - mark))
        ax[2].plot(c["false_pct"], c["catch_pct"], "o", color=hit_c, markersize=6)
        ax[2].annotate(f" {mark} ms", (c["false_pct"], c["catch_pct"]),
                       color=ink, fontsize=9, va="center")
    ax[2].set_xlabel(f"false alarms, % of the {len(neg)} clean rows", color=sub, fontsize=10)
    ax[2].set_ylabel(f"caught, % of the {len(pos)} rows with a missing word", color=sub, fontsize=10)
    ax[2].set_title("Catch against cost", color=ink, fontsize=12, loc="left")

    fig.suptitle("Finding a missing word by the speech no word claims  ·  72 clips, MMS",
                 color=ink, fontsize=13, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out / "unclaimed_curve.png", dpi=150, facecolor="white")
    print(f"{len(pos)} rows with a missing word, {len(neg)} without")
    print(f"-> {args.out / 'unclaimed_curve.png'}")
    print(f"-> {args.out / 'curve.csv'}")


if __name__ == "__main__":
    main()
