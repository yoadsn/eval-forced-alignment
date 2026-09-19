"""Score forced aligners against human word boundaries.

One implementation, used two ways: run it as a script over an exported gold file, or import
`evaluate()` from the server so the dashboard shows the same numbers the script prints.

    python -m hebrew_training.aligner_eval \\
        --dataset data/datasets/plenum --gold gold.jsonl --out eval.json

The metric is boundary error: for every word both a human and an aligner placed, the
absolute difference of their start times and of their end times, pooled. Reported as the
median, the 90th percentile, and the share of boundaries within 20, 50 and 100 ms -- the
20 ms threshold is the strict one used to report forced-alignment accuracy.

The tail matters more than the median here. An aligner that is usually close and
occasionally half a second out produces training cuts that land mid-word, so the p90 and the
within-100 ms figure are the ones to rank by, not the mean.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

_WS = re.compile(r"\s+")
TOLERANCES_MS = (20, 50, 100)


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
        for cid, errs in per_clip.items():
            if errs:
                result["clips"].setdefault(cid, {})[source] = round(statistics.median(errs), 1)

    # The reference floor: where two people marked the same clip. An aligner inside this
    # is already as good as the humans it is being judged by.
    names = sorted(humans)
    between = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = humans[names[i]], humans[names[j]]
            for cid in set(a) & set(b):
                between += boundary_errors(pair_words(a[cid], b[cid]))
    if between:
        result["human_agreement"] = summarise(between)

    for cid, row in result["clips"].items():
        e = by_id[cid]
        row["_text"] = e["text"]
        row["_recording"] = (e.get("metadata") or {}).get("recording")
    return result


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


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
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
