"""Score forced aligners against human word boundaries.

One implementation, used two ways: run it as a script over an exported gold file, or import
`evaluate()` from the server so the dashboard shows the same numbers the script prints.

    python -m hebrew_training.aligner_eval \\
        --dataset data/datasets/plenum --gold gold.jsonl --out eval.json

The metric is boundary error: for every word both a human and an aligner placed, the
absolute difference of their start times and of their end times, pooled. Reported as the
median, the 90th percentile, and the share of boundaries within 10, 25, 50 and 100 ms -- the
tolerances Weber et al. (arXiv:2606.10675) report Hebrew word alignment at, so the numbers
can be set beside theirs. (They may count per word where this counts per boundary.)

The tail matters more than the median here. An aligner that is usually close and
occasionally half a second out produces training cuts that land mid-word, so the p90 and the
within-100 ms figure are the ones to rank by, not the mean.

Significance is tested by resampling *clips*, not boundaries. Boundaries inside one clip
share a speaker, a recording and an annotator, so they are not independent; treating ~1,500
of them as separate samples would call almost any difference significant. Every aligner is
scored on the same clips, so comparisons are paired: each bootstrap draw picks one set of
clips and scores both aligners on it.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import random
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

_WS = re.compile(r"\s+")
TOLERANCES_MS = (10, 25, 50, 100)
BOOTSTRAP_DRAWS = 2000
# Pairwise tests are run on these: p90 because the tail is what cuts training clips mid-word,
# within 50 ms because it is the figure the published Hebrew results lead with.
TESTED_METRICS = ("p90_ms", "within_50ms")


def norm(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def pair_words(human: list[dict], aligner: list[dict]) -> list[tuple[dict, dict]]:
    """Human and aligner words that refer to the same spoken token.

    A human may have corrected a word (its original is in `was`), inserted one the ASR
    missed (`added`), or deleted one it invented (simply absent). Pairing by position would
    then compare the wrong words from the first edit onward. So both lists are aligned as
    token sequences, using the word each side was *originally* given: an added word has no
    original and never pairs, and a deleted one has no human counterpart.
    """
    h_tokens, h_index = [], []
    for i, w in enumerate(human):
        if w.get("added"):
            continue
        h_tokens.append(norm(w.get("was") or w["word"]))
        h_index.append(i)
    a_tokens = [norm(w["word"]) for w in aligner]
    pairs = []
    matcher = SequenceMatcher(a=h_tokens, b=a_tokens, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            pairs.append((human[h_index[block.a + k]], aligner[block.b + k]))
    return pairs


def summarise(errors_ms: list[float]) -> dict:
    if not errors_ms:
        return {"boundaries": 0}
    s = sorted(errors_ms)
    out = {
        "boundaries": len(s),
        "median_ms": round(statistics.median(s), 1),
        "mean_ms": round(statistics.fmean(s), 1),
        "p90_ms": round(s[int(0.9 * (len(s) - 1))], 1),
    }
    for t in TOLERANCES_MS:
        out[f"within_{t}ms"] = round(100 * sum(e <= t for e in s) / len(s), 1)
    return out


def _stat(s: list[float], name: str) -> float | None:
    """One named statistic of an already-sorted pool.

    Takes a sorted list so a bootstrap draw sorts once and reads every statistic off it,
    rather than re-sorting per statistic -- that alone was most of a 5 s page load at 72 clips.
    """
    n = len(s)
    if not n:
        return None
    if name == "median_ms":
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if name == "p90_ms":
        return s[int(0.9 * (n - 1))]
    if name.startswith("within_"):
        t = float(name[len("within_") : -len("ms")])
        return 100 * bisect.bisect_right(s, t) / n
    raise ValueError(name)


def metric(errors_ms: list[float], name: str) -> float | None:
    """One named statistic of a pool of boundary errors."""
    return _stat(sorted(errors_ms), name)


def lower_is_better(name: str) -> bool:
    return not name.startswith("within_")


def _pool(per_clip: dict[str, list[float]], draw: list[str]) -> list[float]:
    out: list[float] = []
    for cid in draw:
        out += per_clip.get(cid, [])
    return out


def confidence_intervals(
    per_clip: dict[str, list[float]], names, rng: random.Random, draws: int
) -> dict[str, list[float]]:
    """95% interval for each statistic, from resampling this aligner's clips."""
    ids = sorted(per_clip)
    samples: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(draws):
        draw = [rng.choice(ids) for _ in ids]
        pool = sorted(_pool(per_clip, draw))
        for n in names:
            v = _stat(pool, n)
            if v is not None:
                samples[n].append(v)
    out = {}
    for n, vals in samples.items():
        vals.sort()
        if vals:
            out[n] = [round(vals[int(0.025 * (len(vals) - 1))], 1),
                      round(vals[int(0.975 * (len(vals) - 1))], 1)]
    return out


def paired_test(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
    name: str,
    rng: random.Random,
    draws: int,
) -> dict | None:
    """Is aligner a different from aligner b on `name`, beyond what the clip sample allows?

    Both are scored on the same resampled clips in every draw, so clip-to-clip variation --
    an easy recording, a mumbling speaker -- cancels out of the difference instead of
    swamping it.
    """
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    observed_a = metric(_pool(a, common), name)
    observed_b = metric(_pool(b, common), name)
    if observed_a is None or observed_b is None:
        return None
    observed = observed_a - observed_b
    diffs = []
    for _ in range(draws):
        draw = [rng.choice(common) for _ in common]
        va, vb = metric(_pool(a, draw), name), metric(_pool(b, draw), name)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    diffs.sort()
    # Two-sided: how often a redrawn sample lands on the other side of zero.
    below = sum(d <= 0 for d in diffs) / len(diffs)
    above = sum(d >= 0 for d in diffs) / len(diffs)
    p = min(1.0, 2 * min(below, above))
    return {
        "diff": round(observed, 1),
        "ci": [round(diffs[int(0.025 * (len(diffs) - 1))], 1),
               round(diffs[int(0.975 * (len(diffs) - 1))], 1)],
        "p": round(p, 4),
        "clips": len(common),
    }


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni: several pairwise tests at once would otherwise find a 'significant'
    difference by chance roughly once in every twenty."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def boundary_errors(pairs: list[tuple[dict, dict]]) -> list[float]:
    out = []
    for h, a in pairs:
        out.append(abs(float(h["start"]) - float(a["start"])) * 1000)
        out.append(abs(float(h["end"]) - float(a["end"])) * 1000)
    return out


def evaluate(
    entries: list[dict],
    gold: list[dict],
    extra_labels: dict | None = None,
    exclude: set[str] | frozenset[str] = frozenset(),
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> dict:
    """Score every aligner label against every human mark.

    entries      dataset manifest rows (id, text, labels=[{source, words}])
    gold         human marks: {id, words, annotator}; `text` is used when `id` is absent
    extra_labels {source: {id: words}} for aligners not stored in the dataset itself
    exclude      annotator names that are not people. A test row is aligner output under a
                 human-looking name: left in, it counts as "human disagreement" and makes
                 two careful people look worse than the aligners they are judging.
    """
    by_id = {e["id"]: e for e in entries}
    by_text: dict[str, list] = {}
    for e in entries:
        by_text.setdefault(norm(e["text"]), []).append(e)

    def resolve(row):
        if row.get("id") in by_id:
            return by_id[row["id"]]
        hits = by_text.get(norm(row.get("text") or row.get("transcript", "")), [])
        return hits[0] if len(hits) == 1 else None

    aligners: dict[str, dict[str, list]] = {}
    for e in entries:
        for label in e.get("labels", []):
            aligners.setdefault(label["source"], {})[e["id"]] = label["words"]
    for source, words_by_id in (extra_labels or {}).items():
        aligners.setdefault(source, {}).update(words_by_id)

    humans: dict[str, dict[str, list]] = {}
    unmatched = 0
    for row in gold:
        if row.get("annotator") in exclude:
            continue
        entry = resolve(row)
        if entry is None:
            unmatched += 1
            continue
        humans.setdefault(row.get("annotator", "?"), {})[entry["id"]] = row["words"]

    marked_ids = sorted({cid for m in humans.values() for cid in m})
    result = {
        "aligners": {},
        "clips": {},
        "human_agreement": None,
        "unmatched_marks": unmatched,
        "marked_clips": len(marked_ids),
        "annotators": {name: len(m) for name, m in humans.items()},
    }

    errors_by_aligner: dict[str, dict[str, list[float]]] = {}
    for source, words_by_id in sorted(aligners.items()):
        pooled, per_clip, paired, human_words = [], {}, 0, 0
        for marks in humans.values():
            for cid, hwords in marks.items():
                awords = words_by_id.get(cid)
                if not awords:
                    continue
                pairs = pair_words(hwords, awords)
                errs = boundary_errors(pairs)
                pooled += errs
                paired += len(pairs)
                human_words += sum(1 for w in hwords if not w.get("added"))
                per_clip.setdefault(cid, []).extend(errs)
        stats = summarise(pooled)
        stats["words_paired_pct"] = round(100 * paired / human_words, 1) if human_words else 0.0
        result["aligners"][source] = stats
        per_clip = {cid: errs for cid, errs in per_clip.items() if errs}
        errors_by_aligner[source] = per_clip
        for cid, errs in per_clip.items():
            result["clips"].setdefault(cid, {})[source] = round(statistics.median(errs), 1)

    # The reference floor: where two people marked the same clip. An aligner inside this
    # is already as good as the humans it is being judged by.
    names = sorted(humans)
    human_per_clip: dict[str, list[float]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = humans[names[i]], humans[names[j]]
            for cid in set(a) & set(b):
                human_per_clip.setdefault(cid, []).extend(
                    boundary_errors(pair_words(a[cid], b[cid]))
                )
    between = [e for errs in human_per_clip.values() for e in errs]
    if between:
        result["human_agreement"] = summarise(between)
        result["human_agreement"]["clips"] = len(human_per_clip)

    # ---- uncertainty ----
    rng = random.Random(seed)
    ci_names = ["median_ms", "p90_ms"] + [f"within_{t}ms" for t in TOLERANCES_MS]
    for source, per_clip in errors_by_aligner.items():
        if per_clip:
            result["aligners"][source]["clips"] = len(per_clip)
            result["aligners"][source]["ci"] = confidence_intervals(per_clip, ci_names, rng, draws)
    if len(human_per_clip) >= 2:
        result["human_agreement"]["ci"] = confidence_intervals(human_per_clip, ci_names, rng, draws)

    tests = []
    scored = sorted(s for s, pc in errors_by_aligner.items() if pc)
    for a, b in itertools.combinations(scored, 2):
        for name in TESTED_METRICS:
            t = paired_test(errors_by_aligner[a], errors_by_aligner[b], name, rng, draws)
            if t is None:
                continue
            better = a if (t["diff"] < 0) == lower_is_better(name) else b
            tests.append({"a": a, "b": b, "metric": name, "better": better, **t})
    for t, adj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = round(adj, 4)
        t["significant"] = adj < 0.05
    result["comparisons"] = tests
    result["draws"] = draws

    # ---- anchoring ----
    # The tool opens every clip on one aligner's boundaries and the annotator moves them from
    # there. A boundary nobody moved is then scored as that aligner being exactly right --
    # whether the person checked it or skipped it. Measured on the first 10 clips: 33% of
    # human boundaries were byte-identical to the seed, against 7% and 1% for the others.
    # So the seed is graded against a reference built partly from its own output, and any
    # comparison involving it is not a fair test.
    seeds = {e["labels"][0]["source"] for e in entries if e.get("labels")}
    seed = seeds.pop() if len(seeds) == 1 else None
    result["seed"] = seed
    for source, words_by_id in aligners.items():
        same = total = 0
        for marks in humans.values():
            for cid, hwords in marks.items():
                awords = words_by_id.get(cid)
                if not awords:
                    continue
                for h, a in pair_words(hwords, awords):
                    for k in ("start", "end"):
                        total += 1
                        same += abs(float(h[k]) - float(a[k])) < 0.0005
        if source in result["aligners"] and total:
            result["aligners"][source]["unmoved_pct"] = round(100 * same / total, 1)
    for t in tests:
        # The head start only ever helps the seed. So a comparison it wins may owe the win
        # to that, but one it loses was lost *despite* it -- the bias pushed the other way,
        # and the verdict is stronger for it, not weaker.
        involved = seed in (t["a"], t["b"])
        t["fair"] = not involved or t["better"] != seed
        t["seed_lost_anyway"] = involved and t["better"] != seed and t["significant"]

    for cid, row in result["clips"].items():
        e = by_id[cid]
        row["_text"] = e["text"]
        row["_recording"] = (e.get("metadata") or {}).get("recording")
    return result


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def legacy_label(path: Path, entries: list[dict]) -> dict[str, list]:
    """An aligner kept in the old per-source layout, keyed onto dataset ids by transcript."""
    by_text: dict[str, list] = {}
    for e in entries:
        by_text.setdefault(norm(e["text"]), []).append(e["id"])
    out = {}
    for row in load_jsonl(path):
        ids = by_text.get(norm(row.get("transcript", "")), [])
        if len(ids) == 1 and row.get("words"):
            out[ids[0]] = row["words"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, required=True, help="Folder holding manifest.jsonl.")
    p.add_argument("--gold", type=Path, required=True, help="Human marks, as /api/export writes them.")
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="An aligner outside the dataset, in the legacy per-source jsonl layout.",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="Annotator to leave out, e.g. a test account. Repeatable.",
    )
    p.add_argument("--out", type=Path, help="Write the full result as json.")
    args = p.parse_args()

    entries = load_jsonl(args.dataset / "manifest.jsonl")
    extra = {}
    for spec in args.extra:
        name, _, path = spec.partition("=")
        extra[name] = legacy_label(Path(path), entries)
    result = evaluate(entries, load_jsonl(args.gold), extra, set(args.exclude))

    print(f"{result['marked_clips']} clips marked by {result['annotators']}")
    if result["unmatched_marks"]:
        print(f"  {result['unmatched_marks']} marks matched no clip and were skipped")
    cols = "".join(f"{'<=' + str(t) + 'ms':>9}" for t in TOLERANCES_MS)
    print(f"{'aligner':<12}{'median':>9}{'p90':>9}{cols}{'paired':>9}")
    ranked = sorted(result["aligners"].items(), key=lambda kv: kv[1].get("p90_ms", 1e9))
    for source, s in ranked:
        if not s.get("boundaries"):
            print(f"{source:<12}  (no overlap with human marks)")
            continue
        within = "".join(f"{s[f'within_{t}ms']:>8}%" for t in TOLERANCES_MS)
        print(f"{source:<12}{s['median_ms']:>7}ms{s['p90_ms']:>7}ms{within}{s['words_paired_pct']:>8}%")
    if result["human_agreement"]:
        h = result["human_agreement"]
        within = "".join(f"{h[f'within_{t}ms']:>8}%" for t in TOLERANCES_MS)
        print(f"{'human-human':<12}{h['median_ms']:>7}ms{h['p90_ms']:>7}ms{within}"
              f"   ({h['boundaries']} boundaries)")

    print(f"\n95% intervals, from resampling clips ({result['draws']} draws):")
    for source, s in ranked:
        ci = s.get("ci") or {}
        if ci:
            print(f"  {source:<11} median {ci['median_ms'][0]:>5}-{ci['median_ms'][1]:<5} ms"
                  f"   p90 {ci['p90_ms'][0]:>5}-{ci['p90_ms'][1]:<6} ms"
                  f"   <=50ms {ci['within_50ms'][0]:>4}-{ci['within_50ms'][1]:<4}%"
                  f"   ({s['clips']} clips)")

    print("\nIs the difference real?  (paired, clip-level; Holm-corrected across all tests)")
    for t in result["comparisons"]:
        unit = "%" if t["metric"].startswith("within_") else " ms"
        verdict = (f"yes -- {t['better']} is better" if t["significant"]
                   else "no  -- could be chance")
        if not t.get("fair", True):
            verdict += "   [unfair: marks started from " + result["seed"] + "]"
        elif t.get("seed_lost_anyway"):
            verdict += "   [holds despite " + result["seed"] + "'s head start]"
        print(f"  {t['a']:>9} vs {t['b']:<9} {t['metric']:<12} diff {t['diff']:>+7}{unit}"
              f"  [{t['ci'][0]:>+.1f}, {t['ci'][1]:>+.1f}]  p={t['p_holm']:<6}  {verdict}")

    if result.get("seed"):
        print(f"\nANCHORING: every clip opens on {result['seed']}'s boundaries. Share of human "
              "boundaries left exactly where an aligner put them:")
        for source, s in ranked:
            if "unmoved_pct" in s:
                tag = "   <- the seed" if source == result["seed"] else ""
                print(f"  {source:<11} {s['unmoved_pct']:>5}%{tag}")
        print(f"  {result['seed']} is graded against marks partly built from its own output;"
              " its scores are flattered and comparisons with it are not a fair test.")
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
