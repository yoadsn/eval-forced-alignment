"""Give MMS back the time it leaves in the gaps, and measure whether that helps.

    python eval/boundary_fix.py --run data/eval_runs/ivrit-ai-corrected

MMS starts words about 25 ms late and ends them 11 ms early, and leaves a 60 ms gap before
each one. The correction is to hand that unclaimed time back: move each start earlier by a
fraction of the gap in front of it, each end later by a fraction of the gap behind it. The
fraction is the thing fitted, not a number of milliseconds, so a word with no free space
beside it is left alone without needing a rule for it.

Two corrections are compared against raw MMS, both fitted on half the clips and measured on
the other half, then swapped, so every clip is scored exactly once while unseen:

    shift      a constant number of milliseconds per edge
    fraction   a constant share of the adjacent gap per edge

Clips, not words, go into the folds: boundaries inside one clip share a speaker, a recording
and an annotator, and splitting by word would put near-duplicates on both sides.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(run: Path, label: str, exclude: set[str], phonemes: Path | None = None) -> list[dict]:
    """One row per paired word: where MMS put it, where the human did, and the free space.

    With `phonemes` (the cache eval/phonemize_words.py writes) each row also carries the
    word's IPA, so a correction can key on the sound at the edge rather than the letter.
    """
    from hebrew_training.aligner_eval import pair_words

    ipa = json.loads(phonemes.read_text(encoding="utf-8")) if phonemes and phonemes.exists() else {}

    marks = []
    for f in sorted((run / "marks").glob("*.jsonl")):
        if f.stem not in exclude:
            marks += [{**r, "annotator": f.stem} for r in load_jsonl(f)]
    words = {r["id"]: r["words"] for r in load_jsonl(run / "labels" / f"{label}.jsonl")}
    # The label the tool seeds clips from: a human boundary identical to it may never have
    # been looked at, so it is not independent evidence and is reported separately.
    seed = {}
    for r in load_jsonl(run / "dataset" / "manifest.jsonl"):
        for lb in r["labels"]:
            if lb["source"] == "ivrit-ai":
                seed[r["id"]] = lb["words"]

    rows = []
    for m in marks:
        ws = words.get(f"{m['id']}#{m['annotator']}")
        if not ws:
            continue
        at = {id(a): i for i, a in enumerate(ws)}
        seeded = {(round(w["start"], 3), round(w["end"], 3)) for w in seed.get(m["id"], [])}
        for h, a in pair_words(m["words"], ws):
            i = at[id(a)]
            rows.append({
                "clip": m["id"], "who": m.get("annotator", "anon"),
                "word": a["word"], "score": a.get("score"),
                "ipa": (ipa.get(f"{m['id']}#{m.get('annotator', 'anon')}") or [""] * (i + 1))[i]
                if i < len(ipa.get(f"{m['id']}#{m.get('annotator', 'anon')}") or []) else "",
                "first_word": i == 0, "last_word": i == len(ws) - 1,
                "start": a["start"], "end": a["end"],
                "h_start": h["start"], "h_end": h["end"],
                "gap_before": a["start"] - (ws[i - 1]["end"] if i else 0.0),
                "gap_after": (ws[i + 1]["start"] if i + 1 < len(ws) else a["end"] + 0.0) - a["end"],
                "prev_end": ws[i - 1]["end"] if i else 0.0,
                "next_start": ws[i + 1]["start"] if i + 1 < len(ws) else None,
                "unmoved": (round(h["start"], 3), round(h["end"], 3)) in seeded,
            })
    return rows


def fit(rows: list[dict], kind: str) -> tuple[float, float]:
    """The median correction on these rows: milliseconds, or a share of the adjacent gap."""
    if kind == "shift":
        return (statistics.median([r["start"] - r["h_start"] for r in rows]),
                statistics.median([r["h_end"] - r["end"] for r in rows]))
    starts = [(r["start"] - r["h_start"]) / r["gap_before"] for r in rows if r["gap_before"] > 0.001]
    ends = [(r["h_end"] - r["end"]) / r["gap_after"] for r in rows if r["gap_after"] > 0.001]
    return statistics.median(starts), statistics.median(ends)


def apply(row: dict, kind: str, a: float, b: float) -> tuple[float, float]:
    """The corrected boundaries, never crossing a neighbour or inverting the word."""
    if kind == "shift":
        start, end = row["start"] - a, row["end"] + b
    else:
        start = row["start"] - a * max(row["gap_before"], 0.0)
        end = row["end"] + b * max(row["gap_after"], 0.0)
    start = max(start, row["prev_end"])
    if row["next_start"] is not None:
        end = min(end, row["next_start"])
    if end <= start:
        start, end = row["start"], row["end"]
    return start, end


# Hebrew letters by how sharp an edge they give the aligner to find. A plosive is a burst
# after a closure, which is about as findable as a boundary gets; a glide or a glottal letter
# blends into whatever is beside it. If the 25 ms of lateness is acoustic rather than a
# quirk of the corpus, it should differ across these and transfer to other Hebrew audio.
LETTER_CLASS = {
    **{c: "plosive" for c in "בגדכפתטקץ"},
    **{c: "fricative" for c in "וזחסשצשׂ"},
    **{c: "nasal" for c in "מנםן"},
    **{c: "liquid" for c in "לר"},
    **{c: "glottal" for c in "אהעי"},
}


def letter_class(ch: str) -> str:
    return LETTER_CLASS.get(ch, "other")


def edge_letter(word: str, edge: str) -> str:
    """The first or last letter the model actually had to align.

    Not word[0] / word[-1]: a sixth of the words end in a full stop or a comma, which MMS
    cannot spell and never aligned, so keying on it bucketed 131 word-ends by a character
    that played no part in placing them.
    """
    letters = [c for c in word if c in LETTER_CLASS]
    if not letters:
        return ""
    return letters[0] if edge == "start" else letters[-1]


# Hebrew IPA from Phonikud, by manner. The point of going through phonemes at all: the
# letter at the edge of a written word is often not the sound at the edge of the spoken one.
# המצאה ends in the letter ה and the sound /a/; ו is a consonant in one word and a vowel in
# the next; and stress, which doubles a vowel's length, is never written down.
PHONE_CLASS = {
    **{c: "vowel" for c in "aeiou"},
    **{c: "plosive" for c in "pbtdkɡʔ"},
    **{c: "fricative" for c in "fvszʃχhʒ"},
    **{c: "nasal" for c in "mn"},
    **{c: "liquid" for c in "lʁr"},
    **{c: "glide" for c in "jw"},
}
STRESS = "ˈ"     # the mark Phonikud puts before the stressed vowel
PREFIX = "|"


def ipa_letters(ipa: str) -> str:
    """Just the sounds: stress marks and prefix bars removed."""
    return "".join(c for c in ipa if c not in (STRESS, PREFIX) and not c.isspace())


def phone_class(ipa: str, edge: str) -> str:
    letters = ipa_letters(ipa)
    if not letters:
        return "?"
    # An affricate is written as two characters; at a word edge it behaves as the stop.
    ch = letters[0] if edge == "start" else letters[-1]
    return PHONE_CLASS.get(ch, "other")


def final_stressed(ipa: str) -> bool:
    """Whether the stress falls on the last vowel -- Hebrew's default, and the case where
    the final vowel is long."""
    core = ipa.replace(PREFIX, "")
    at = core.rfind(STRESS)
    if at < 0:
        return True     # unmarked means final stress, which is the Hebrew default
    return not any(c in PHONE_CLASS and PHONE_CLASS[c] == "vowel" for c in core[at + 2:])


PAUSE_EDGES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80)


def pause_bucket(gap: float) -> int:
    for i, edge in enumerate(PAUSE_EDGES):
        if gap < edge:
            return i
    return len(PAUSE_EDGES)


# Candidate features. Each takes a row and which edge is being corrected, and returns the
# bucket that boundary belongs to; a single constant is the "none" bucket everything shares.
FEATURES = {
    "none": lambda r, edge: "all",
    "letter class": lambda r, edge: letter_class(edge_letter(r["word"], edge)),
    "letter": lambda r, edge: edge_letter(r["word"], edge),
    # How much silence sits beside the boundary. This is a proxy for prosodic boundary
    # strength, which is what governs pre-boundary lengthening -- and the lengthening keeps
    # growing with it, so the old version of this feature (50 ms steps, everything past
    # 200 ms in one bucket) collapsed exactly the range where the effect is largest.
    "pause": lambda r, edge: pause_bucket(r["gap_before"] if edge == "start" else r["gap_after"]),
    "ctc score": lambda r, edge: min(int((r["score"] or 0) * 4), 3),
    "phone class": lambda r, edge: phone_class(r.get("ipa", ""), edge),
    "phone + pause": lambda r, edge: (phone_class(r.get("ipa", ""), edge),
                                      0 if (r["gap_before"] if edge == "start" else r["gap_after"]) < 0.05
                                      else (1 if (r["gap_before"] if edge == "start" else r["gap_after"]) < 0.2 else 2)),
    "final stress": lambda r, edge: (phone_class(r.get("ipa", ""), edge), final_stressed(r.get("ipa", ""))),
    "clip edge": lambda r, edge: (edge == "start" and r["first_word"]) or (edge == "end" and r["last_word"]),
    "word length": lambda r, edge: min(len(r["word"]), 7),
}
MIN_BUCKET = 25


def fit_buckets(rows, feature) -> dict:
    """A median shift per (edge, bucket), falling back to the overall median when a bucket
    is too thin to trust. Without that fallback a rare letter would be fitted on a handful of
    boundaries and would make the feature look worse than it is."""
    keyfn = FEATURES[feature]
    out = {}
    for edge in ("start", "end"):
        vals = {}
        for r in rows:
            d = (r["start"] - r["h_start"]) if edge == "start" else (r["h_end"] - r["end"])
            vals.setdefault(keyfn(r, edge), []).append(d)
        overall = statistics.median([v for vs in vals.values() for v in vs])
        out[edge] = {"_": overall}
        for bucket, vs in vals.items():
            out[edge][bucket] = statistics.median(vs) if len(vs) >= MIN_BUCKET else overall
    return out


def apply_buckets(row, feature, table) -> tuple[float, float]:
    keyfn = FEATURES[feature]
    a = table["start"].get(keyfn(row, "start"), table["start"]["_"])
    b = table["end"].get(keyfn(row, "end"), table["end"]["_"])
    start = max(row["start"] - a, row["prev_end"])
    end = row["end"] + b
    if row["next_start"] is not None:
        end = min(end, row["next_start"])
    if end <= start:
        start, end = row["start"], row["end"]
    return start, end


def two_fold_feature(rows, feature) -> list[float]:
    clips = sorted({r["clip"] for r in rows})
    half = set(clips[::2])
    out = []
    for train_in_half in (True, False):
        train = [r for r in rows if (r["clip"] in half) == train_in_half]
        table = fit_buckets(train, feature)
        for r in [r for r in rows if (r["clip"] in half) != train_in_half]:
            s, e = apply_buckets(r, feature, table)
            out += [abs(s - r["h_start"]) * 1000, abs(e - r["h_end"]) * 1000]
    return out


def errors(rows, kind=None, a=0.0, b=0.0) -> list[float]:
    out = []
    for r in rows:
        s, e = (r["start"], r["end"]) if kind is None else apply(r, kind, a, b)
        out += [abs(s - r["h_start"]) * 1000, abs(e - r["h_end"]) * 1000]
    return out


def two_fold(rows, kind) -> list[float]:
    """Fit on one half of the clips, score the other, and swap."""
    clips = sorted({r["clip"] for r in rows})
    half = set(clips[::2])
    out = []
    for train_in_half in (True, False):
        train = [r for r in rows if (r["clip"] in half) == train_in_half]
        test = [r for r in rows if (r["clip"] in half) != train_in_half]
        a, b = fit(train, kind)
        out += errors(test, kind, a, b)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--label", default="mms")
    p.add_argument("--exclude", action="append", default=["probe"])
    p.add_argument("--phonemes", type=Path, default=Path("data/unclaimed/phonemes.json"),
                   help="Cache from eval/phonemize_words.py; without it the phoneme features "
                        "fall back to one bucket and are simply the flat shift.")
    args = p.parse_args()

    from hebrew_training.aligner_eval import summarise

    rows = build(args.run, args.label, set(args.exclude), args.phonemes)
    for title, subset in (("all boundaries", rows),
                          ("dropping the ones never moved off the seed",
                           [r for r in rows if not r["unmoved"]])):
        print(f"\n{title}: {len(subset)} words, {len(subset) * 2} boundaries")
        a, b = fit(subset, "fraction")
        print(f"  fitted fraction of the adjacent gap: start {a:.2f}, end {b:.2f}")
        print(f"{'':<14}{'median':>9}{'p90':>9}{'<=10ms':>9}{'<=25ms':>9}{'<=50ms':>9}{'<=100ms':>9}")
        named = [("mms, raw", errors(subset)),
                 ("+ fraction", two_fold(subset, "fraction"))]
        named += [(f"+ by {f}" if f != "none" else "+ one shift", two_fold_feature(subset, f))
                  for f in FEATURES]
        for name, errs in named:
            s = summarise(errs)
            print(f"  {name:<16}{s['median_ms']:>9}{s['p90_ms']:>9}{s['within_10ms']:>9}"
                  f"{s['within_25ms']:>9}{s['within_50ms']:>9}{s['within_100ms']:>9}")


if __name__ == "__main__":
    main()
