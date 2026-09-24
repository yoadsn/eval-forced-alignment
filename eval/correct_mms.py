"""The MMS boundary correction, as a rule rather than a model.

    python eval/correct_mms.py --run data/eval_runs/ivrit-ai-corrected        # measure it
    python eval/correct_mms.py --run <run> --label mms --write mms-corrected  # apply it

Two corrections, both applied to MMS's output. Nothing is fed back into the aligner.

  1. **Every boundary gets a shift that depends on the letter at it.** MMS starts words late
     and ends them early, by an amount set by how sharp an edge the letter gives it: a word
     beginning with a plosive needs no correction at all, one beginning with a fricative
     needs about 45 ms.

  2. **A word end followed by a pause is extended to where the sound stops.** This is the
     case MMS gets worst -- it cuts about 60 ms early -- and its own posterior cannot help,
     because CTC blank means "no new letter", not "no sound". The waveform can: the end
     moves forward while the envelope stays above a fraction of that word's own peak.

The fitted numbers live in a JSON beside the run, so they can be read, argued with and
reused without refitting. Correcting a run needs only numpy and soundfile.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOP = 0.005


def load_bf():
    spec = importlib.util.spec_from_file_location("bf", ROOT / "eval" / "boundary_fix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def envelope(path: Path):
    import numpy as np
    import soundfile as sf

    wav, rate = sf.read(path, dtype="float32", always_2d=True)
    wav = np.abs(wav.mean(axis=1))
    win = max(1, int(HOP * rate))
    return wav[: len(wav) // win * win].reshape(-1, win).max(axis=1)


def decay_end(env, start: float, end: float, limit: float, quiet: float) -> float:
    """Where the sound after a word actually stops.

    Walks forward while the envelope stays above `quiet` of the word's own peak, so a loud
    speaker and a quiet one are treated alike, and stops at `limit` -- the next word, or the
    cap. Using the word's own peak rather than an absolute level is what makes one threshold
    work across a corpus of hundreds of speakers.
    """
    a, b = int(start / HOP), int(end / HOP)
    inside = env[a:b]
    if not len(inside):
        return end
    peak = float(inside.max())
    if peak <= 0:
        return end
    i, stop = b, int(limit / HOP)
    while i < min(stop, len(env)) and env[i] > quiet * peak:
        i += 1
    return max(end, i * HOP)


def no_overlap(clips: list[list[dict]]) -> None:
    """Refuse to write words that sit on top of each other.

    Every measure in this work is taken per boundary against a human mark, so an overlap is
    invisible to all of them -- two words can cross while both move closer to their targets,
    which is how 222 overlapping pairs once went out to the live site unnoticed. A constraint
    the scoring cannot see has to be asserted, not hoped for.

    Each list is one clip's words in time order.
    """
    bad = [(a["word"], b["word"], (a["end"] - b["start"]) * 1000)
           for words in clips for a, b in zip(words, words[1:])
           if a["end"] - b["start"] > 1e-9]
    if bad:
        worst = max(x[2] for x in bad)
        raise SystemExit(
            f"refusing to write: {len(bad)} pairs of corrected words overlap, worst "
            f"{worst:.0f} ms, e.g. {bad[0][0]!r} over {bad[0][1]!r}. That is a bug in the "
            f"correction, not in the data."
        )


def corrected(bf, rows, model) -> dict:
    """Where every one of these words ends up, keyed by the row's identity.

    Rows are grouped back into the clips they came from and ordered by time, because the
    correction is only well defined for a whole clip at once -- one word's end is the same
    boundary as the next word's start.
    """
    by: dict[tuple, list] = {}
    for r in rows:
        by.setdefault((r["clip"], r.get("who")), []).append(r)
    out = {}
    for ws in by.values():
        ws.sort(key=lambda r: r["start"])
        placed = correct_clip(bf, ws, model["shifts"], model["quiet"], model["cap"],
                              model["pause_min"])
        for r, se in zip(ws, placed):
            out[id(r)] = se
    return out


def fit(bf, rows, quiets=(0.05, 0.10, 0.15, 0.20, 0.30), caps=(0.1, 0.15, 0.2, 0.3, 0.5),
        pause_min=0.1) -> dict:
    """Shifts per letter class, then the energy rule's two numbers, in that order.

    The rule is fitted on top of the shifted ends rather than beside them, because it is a
    correction to a correction: whatever the shift leaves behind before a pause is what the
    envelope is asked to explain.
    """
    shifts = {"start": {}, "end": {}}
    for edge in ("start", "end"):
        by: dict[str, list] = {}
        for r in rows:
            d = (r["start"] - r["h_start"]) if edge == "start" else (r["h_end"] - r["end"])
            by.setdefault(bf.letter_class(bf.edge_letter(r["word"], edge)), []).append(d)
        overall = statistics.median([v for vs in by.values() for v in vs])
        shifts[edge]["_"] = round(overall, 4)
        for cls, vs in by.items():
            shifts[edge][cls] = round(statistics.median(vs) if len(vs) >= 25 else overall, 4)

    pre = [r for r in rows if r["gap_after"] >= pause_min and r.get("env") is not None]
    best = (None, None, None)
    for quiet in quiets:
        for cap in caps:
            # Scored through the clip-wide resolution, not word by word: a setting that
            # reaches further than the gap allows must be judged on what it is finally
            # given, not on what it asked for.
            placed = corrected(bf, rows, {"shifts": shifts, "quiet": quiet, "cap": cap,
                                          "pause_min": pause_min})
            err = [abs(placed[id(r)][1] - r["h_end"]) for r in pre]
            score = statistics.fmean(err)
            if best[0] is None or score < best[0]:
                best = (score, quiet, cap)
    return {"shifts": shifts, "quiet": best[1], "cap": best[2], "pause_min": pause_min,
            "fitted_on_words": len(rows), "fitted_on_pre_pause_ends": len(pre)}


def wanted(bf, r, shifts, quiet, cap, pause_min) -> tuple[float, float]:
    """How far this word would like each edge moved outward, before any neighbour is asked.

    Nothing is clamped here. Clamping a word against its neighbour's *original* position is
    what put 22% of pairs on top of each other: with a 16 ms gap, an end pushed out 8 ms and
    the next start pulled back 8 ms each stayed inside the original gap, and still met in
    the middle. A boundary belongs to two words, so it has to be settled once, by
    correct_clip, and not twice.
    """
    a = shifts["start"].get(bf.letter_class(bf.edge_letter(r["word"], "start")), shifts["start"]["_"])
    b = shifts["end"].get(bf.letter_class(bf.edge_letter(r["word"], "end")), shifts["end"]["_"])
    if r["gap_after"] >= pause_min and r.get("env") is not None:
        grown = decay_end(r["env"], r["start"], r["end"], r["end"] + cap, quiet) - r["end"]
        b = max(b, grown)
    return max(a, 0.0), max(b, 0.0)


def correct_clip(bf, rows, shifts, quiet, cap, pause_min) -> list[tuple[float, float]]:
    """Correct every word of one clip together, so no two of them can cross.

    `rows` are that clip's words in order. Each gap between two words is shared: the word on
    the left wants to grow right into it, the word on the right wants to grow left. When
    they want more than there is, both are scaled by the same factor so they meet exactly
    and never pass. A gap of zero therefore moves nothing, which is the rule the user asked
    for at the outset -- a word may move, but not across its neighbour.
    """
    ask = [wanted(bf, r, shifts, quiet, cap, pause_min) for r in rows]
    starts = [r["start"] - ask[i][0] for i, r in enumerate(rows)]
    ends = [r["end"] + ask[i][1] for i, r in enumerate(rows)]

    # the clip's own edges
    if rows:
        starts[0] = max(starts[0], 0.0)
        env = rows[-1].get("env")
        if env is not None:
            ends[-1] = min(ends[-1], len(env) * HOP)

    for i in range(len(rows) - 1):
        gap = rows[i + 1]["start"] - rows[i]["end"]
        if gap <= 0:                       # already touching: neither may take anything
            ends[i], starts[i + 1] = rows[i]["end"], rows[i + 1]["start"]
            continue
        want = ask[i][1] + ask[i + 1][0]
        if want > gap:
            scale = gap / want
            ends[i] = rows[i]["end"] + ask[i][1] * scale
            starts[i + 1] = rows[i + 1]["start"] - ask[i + 1][0] * scale
    return [(s, max(e, s + 0.01)) for s, e in zip(starts, ends)]


def attach_envelopes(rows, dataset: Path):
    entries = {}
    for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            entries[e["id"]] = e
    cache: dict[str, object] = {}
    for r in rows:
        cid = r["clip"]
        if cid not in cache:
            cache[cid] = envelope(dataset / entries[cid]["audio"]) if cid in entries else None
        r["env"] = cache[cid]
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ivrit-ai"))
    p.add_argument("--label", default="mms")
    p.add_argument("--write", help="Also write the corrected labels under this name.")
    args = p.parse_args()

    from hebrew_training.aligner_eval import summarise

    bf = load_bf()
    rows = attach_envelopes(bf.build(args.run, args.label, {"probe"}), args.dataset)
    clips = sorted({r["clip"] for r in rows})
    half = set(clips[::2])

    # Measured the only way that means anything: fitted on half the clips, scored on the
    # other half, then swapped.
    held, held_end, held_pre = [], [], []
    for train_in_half in (True, False):
        model = fit(bf, [r for r in rows if (r["clip"] in half) == train_in_half])
        test = [r for r in rows if (r["clip"] in half) != train_in_half]
        placed = corrected(bf, test, model)
        for r in test:
            s, e = placed[id(r)]
            held += [abs(s - r["h_start"]) * 1000, abs(e - r["h_end"]) * 1000]
            held_end.append(abs(e - r["h_end"]) * 1000)
            if r["gap_after"] >= 0.3:
                held_pre.append(abs(e - r["h_end"]) * 1000)

    raw_end = [abs(r["end"] - r["h_end"]) * 1000 for r in rows]
    raw_pre = [abs(r["end"] - r["h_end"]) * 1000 for r in rows if r["gap_after"] >= 0.3]
    print(f"{'':<26}{'median':>9}{'p90':>9}{'<=25ms':>9}{'<=50ms':>9}")
    for name, vals in (("mms, raw (all)", bf.errors(rows)), ("corrected (all)", held),
                       ("mms ends, raw", raw_end), ("corrected ends", held_end),
                       ("mms ends before a pause", raw_pre), ("corrected, those ends", held_pre)):
        s = summarise(vals)
        print(f"  {name:<24}{s['median_ms']:>9}{s['p90_ms']:>9}{s['within_25ms']:>9}{s['within_50ms']:>9}")

    # The model to keep is fitted on everything; the numbers to quote are the held-out ones
    # above. Writing both into the file keeps that distinction where it cannot be lost.
    model = fit(bf, rows)
    model["held_out"] = {"all": summarise(held), "ends": summarise(held_end),
                         "ends_before_a_pause": summarise(held_pre)}
    out = args.run / "correction.json"
    out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nshifts (ms, positive = move outward):")
    for cls in sorted(set(model["shifts"]["start"]) | set(model["shifts"]["end"])):
        if cls == "_":
            continue
        print(f"  {cls:<12} start {model['shifts']['start'].get(cls, 0) * 1000:>6.0f}   "
              f"end {model['shifts']['end'].get(cls, 0) * 1000:>6.0f}")
    print(f"  energy rule: extend while above {model['quiet']:.0%} of the word's peak, "
          f"up to {model['cap'] * 1000:.0f} ms, when the gap after is {model['pause_min'] * 1000:.0f} ms or more")
    print(f"-> {out}")

    if args.write:
        placed = corrected(bf, rows, model)
        by_key: dict[str, list] = {}
        for r in rows:
            key = f"{r['clip']}#{r['who']}"
            by_key.setdefault(key, []).append(r)
        written = args.run / "labels" / f"{args.write}.jsonl"
        lines, per_clip = [], []
        for key, rs in by_key.items():
            words = [{"word": r["word"],
                      "start": round(placed[id(r)][0], 4),
                      "end": round(placed[id(r)][1], 4),
                      **({"score": r["score"]} if r.get("score") is not None else {})}
                     for r in sorted(rs, key=lambda r: r["start"])]
            per_clip.append(words)
            lines.append(json.dumps({"id": key, "words": words}, ensure_ascii=False))
        # Checked on the rounded values that go to disk, not on the floats behind them.
        no_overlap(per_clip)
        written.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
        print(f"-> {written}  ({sum(len(w) for w in per_clip)} words, no overlaps)")


if __name__ == "__main__":
    main()
