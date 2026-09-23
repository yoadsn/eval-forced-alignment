"""The filter itself: one detector per fault, honestly scored.

    python eval/mismatch_filter.py --run data/mismatch/ivrit-ai

The two faults need opposite instruments. An **extra** word -- in the transcript, not in the
audio -- is placed at low confidence, so the signal is a low CTC score. A **missing** word --
spoken, absent from the transcript -- makes the score go *up*, because what remains fits the
audio better than before; there the signal is time, as the words spanning the hole stretch
to cover speech nobody is claiming.

So a row is flagged when

    min(word score) < t1      or      max(word duration / letters) > t2

Both thresholds are fitted, which on 72 rows will flatter itself if fitted and scored on the
same clips. They are therefore chosen on half the clips and measured on the other half, over
many random splits, and both numbers are reported: what the fit promises, and what it
delivers on clips it has not seen.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def features(words: list[dict]) -> tuple[float, float]:
    """(lowest confidence, longest time per letter) for one version of one clip."""
    return (min(w["score"] for w in words),
            max(w["dur"] / max(w["chars"], 1) for w in words))


def flagged(f, t1, t2) -> bool:
    return f[0] < t1 or f[1] > t2


def fit(clean, bad, budget, grid=40):
    """Thresholds catching the most bad rows while dropping at most `budget`% of clean ones.

    The grid is drawn from the clean rows' own quantiles: a threshold only matters where it
    falls between two observed values, and this keeps the search to the range that can be
    afforded at all.
    """
    s = sorted(f[0] for f in clean)
    d = sorted(f[1] for f in clean)
    best = (0.0, min(s), max(d))
    for i in range(grid + 1):
        # Never spend the whole budget on one detector: each is capped at the budget, and
        # the pair is then checked against it together.
        t1 = s[min(int(len(s) * budget / 100 * i / grid), len(s) - 1)]
        for j in range(grid + 1):
            t2 = d[max(len(d) - 1 - int(len(d) * budget / 100 * j / grid), 0)]
            lost = sum(1 for f in clean if flagged(f, t1, t2)) / len(clean) * 100
            if lost > budget:
                continue
            caught = sum(1 for f in bad if flagged(f, t1, t2)) / len(bad) * 100
            if caught > best[0]:
                best = (caught, t1, t2)
    return best[1], best[2]


def rate(rows, t1, t2):
    return round(100 * sum(1 for f in rows if flagged(f, t1, t2)) / len(rows), 1) if rows else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--budget", type=float, default=5.0, help="False-drop budget, percent of clean rows.")
    p.add_argument("--splits", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()

    words = [r for r in load_jsonl(args.run / "words.jsonl") if r.get("score") is not None]
    by: dict[tuple, list] = {}
    for r in words:
        by.setdefault((r["clip"], r["kind"]), []).append(r)
    kinds = sorted({k for _, k in by} - {"clean"})
    clips = sorted({c for c, _ in by})
    feat = {key: features(ws) for key, ws in by.items()}

    # what the fit promises: thresholds chosen and scored on everything
    clean_all = [feat[(c, "clean")] for c in clips if (c, "clean") in feat]
    bad_all = [feat[(c, k)] for c in clips for k in kinds if (c, k) in feat]
    t1, t2 = fit(clean_all, bad_all, args.budget)
    print(f"fitted on all {len(clips)} clips: score < {t1:.4f} or time/letter > {t2:.4f}")
    print(f"  false drops {rate(clean_all, t1, t2)}%   (budget {args.budget}%)")
    for kind in kinds:
        print(f"  {kind:<12} caught {rate([feat[(c, kind)] for c in clips if (c, kind) in feat], t1, t2)}%")

    # what it delivers: fitted on half the clips, measured on the other half
    rng = random.Random(args.seed)
    held: dict[str, list] = {k: [] for k in ["clean", *kinds]}
    for _ in range(args.splits):
        shuffled = clips[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        tr, te = set(shuffled[:half]), set(shuffled[half:])
        a = fit([feat[(c, "clean")] for c in tr if (c, "clean") in feat],
                [feat[(c, k)] for c in tr for k in kinds if (c, k) in feat], args.budget)
        for kind in ["clean", *kinds]:
            rows = [feat[(c, kind)] for c in te if (c, kind) in feat]
            if rows:
                held[kind].append(rate(rows, *a))
    print(f"\nheld out ({args.splits} random half/half splits, mean):")
    print(f"  false drops {sum(held['clean']) / len(held['clean']):.1f}%")
    for kind in kinds:
        print(f"  {kind:<12} caught {sum(held[kind]) / len(held[kind]):.1f}%")

    out = {"budget": args.budget, "fitted": {"t1": t1, "t2": t2},
           "held_out": {k: round(sum(v) / len(v), 1) for k, v in held.items() if v}}
    (args.run / "filter.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.run / 'filter.json'}")


if __name__ == "__main__":
    main()
