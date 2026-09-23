"""What the aligner's confidence does and does not reveal about a bad transcript.

    python eval/mismatch_report.py --run data/mismatch/ivrit-ai

Reads the word rows written by eval/mismatch.py and answers three questions:

1. Does a tampered word score lower than a genuine one at all?
2. If a sentence is bad, can we find it -- "drop the row when its worst word scores below
   t", swept over t, with the clean versions supplying the false-drop rate?
3. Where does this fail? Broken down by corruption kind, word length and position, because a
   filter that is blind to a whole category needs saying so out loud.

Writes report.json next to the input and prints the same thing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def median(vals):
    s = sorted(vals)
    if not s:
        return None
    mid = len(s) // 2
    return round(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2, 4)


def sentence_curve(pos, neg):
    """Catch rate against false-drop rate for 'drop the row when its badness exceeds t'.

    Thresholds are the observed values themselves, so every achievable operating point is
    covered and none is invented between them.
    """
    points = []
    for t in sorted({round(s, 4) for s in pos + neg}):
        points.append({"t": t,
                       "catch": pct(sum(1 for s in pos if s >= t), len(pos)),
                       "false_drop": pct(sum(1 for s in neg if s >= t), len(neg))})
    return points


# How suspicious a sentence looks, from its words alone. Higher is worse.
#
# `min` is the obvious one and the worst: take the minimum of eighteen numbers and a clean
# sentence will supply a low one too, so the honest signal in one bad word is drowned by the
# minimum-of-many. The rest are attempts to escape that -- either by judging a word against
# its own sentence, since a noisy recording drags every score down together, or by counting
# how many words are bad rather than how bad the worst one is.
STATS = {
    "min": lambda ws: -min(w["score"] for w in ws),
    "median_minus_min": lambda ws: median([w["score"] for w in ws]) - min(w["score"] for w in ws),
    "count_below_0.1": lambda ws: sum(1 for w in ws if w["score"] < 0.1),
    "count_below_0.25": lambda ws: sum(1 for w in ws if w["score"] < 0.25),
    "min_score_per_char": lambda ws: -min(w["score"] * max(w["dur"], 1e-3) / w["chars"] for w in ws),
}


def at_budget(points, budget):
    """The best catch rate whose false-drop rate stays within a budget."""
    ok = [p for p in points if p["false_drop"] is not None and p["false_drop"] <= budget]
    return max(ok, key=lambda p: p["catch"]) if ok else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    rows = [r for r in load_jsonl(args.run / "words.jsonl") if r.get("score") is not None]
    kinds = sorted({r["kind"] for r in rows} - {"clean"})

    # 1. word level
    genuine = [r["score"] for r in rows if not r["tampered"]]
    report = {"words": len(rows), "genuine_median": median(genuine), "by_kind": {}}
    print(f"{len(rows)} words, {len(genuine)} genuine (median score {median(genuine)})\n")
    print(f"{'kind':<12}{'n':>5}{'median':>9}{'worst-in-clip':>15}{'caught@5%':>11}{'caught@20%':>12}")

    # 2. sentence level -- clean versions are the negatives
    by_clip_kind: dict[tuple, list] = {}
    for r in rows:
        by_clip_kind.setdefault((r["clip"], r["kind"]), []).append(r)
    clean_sets = [ws for (c, k), ws in by_clip_kind.items() if k == "clean"]

    for kind in kinds:
        tampered = [r for r in rows if r["kind"] == kind and r["tampered"]]
        bad_sets = [ws for (c, k), ws in by_clip_kind.items() if k == kind]
        # Is the word we broke the worst word in its own sentence? That is what makes a
        # flagged row checkable by hand rather than merely suspicious.
        worst = 0
        for r in tampered:
            ws = by_clip_kind[(r["clip"], kind)]
            if r["score"] == min(x["score"] for x in ws):
                worst += 1
        stats = {}
        for name, fn in STATS.items():
            curve = sentence_curve([fn(ws) for ws in bad_sets], [fn(ws) for ws in clean_sets])
            stats[name] = {"at_5pct_false_drop": at_budget(curve, 5),
                           "at_20pct_false_drop": at_budget(curve, 20),
                           "curve": curve}
        best = max(stats, key=lambda n: (stats[n]["at_5pct_false_drop"] or {"catch": -1})["catch"])
        b5 = stats[best]["at_5pct_false_drop"]
        b20 = stats[best]["at_20pct_false_drop"]
        report["by_kind"][kind] = {
            "tampered_words": len(tampered),
            "tampered_median": median([r["score"] for r in tampered]),
            "worst_in_clip_pct": pct(worst, len(tampered)),
            "best_statistic": best, "stats": stats,
        }
        print(f"{kind:<12}{len(tampered):>5}{median([r['score'] for r in tampered]):>9}"
              f"{str(pct(worst, len(tampered))) + '%':>15}"
              f"{(str(b5['catch']) + '%') if b5 else '-':>11}"
              f"{(str(b20['catch']) + '%') if b20 else '-':>12}  via {best}")

    # 3. where it fails
    print("\nwhere it fails -- share of tampered words that are the worst in their sentence")
    cuts = {"1-3 letters": lambda r: r["chars"] <= 3, "4-6 letters": lambda r: 4 <= r["chars"] <= 6,
            "7+ letters": lambda r: r["chars"] >= 7, "first or last word": lambda r: r["edge"]}
    report["breakdown"] = {}
    print(f"{'':<20}" + "".join(f"{k:>12}" for k in kinds))
    for label, keep in cuts.items():
        cells = {}
        line = f"{label:<20}"
        for kind in kinds:
            sel = [r for r in rows if r["kind"] == kind and r["tampered"] and keep(r)]
            hits = sum(1 for r in sel
                       if r["score"] == min(x["score"] for x in by_clip_kind[(r["clip"], kind)]))
            cells[kind] = {"n": len(sel), "worst_in_clip_pct": pct(hits, len(sel))}
            line += f"{(str(pct(hits, len(sel))) + '%' if sel else '-'):>12}"
        report["breakdown"][label] = cells
        print(line)

    (args.run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n-> {args.run / 'report.json'}")


if __name__ == "__main__":
    main()
