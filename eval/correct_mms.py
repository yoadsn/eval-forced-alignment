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
            err = []
            for r in pre:
                end = apply_end(bf, r, shifts, quiet, cap, pause_min)
                err.append(abs(end - r["h_end"]))
            score = statistics.fmean(err)
            if best[0] is None or score < best[0]:
                best = (score, quiet, cap)
    return {"shifts": shifts, "quiet": best[1], "cap": best[2], "pause_min": pause_min,
            "fitted_on_words": len(rows), "fitted_on_pre_pause_ends": len(pre)}


def apply_start(bf, r, shifts) -> float:
    a = shifts["start"].get(bf.letter_class(bf.edge_letter(r["word"], "start")), shifts["start"]["_"])
    return max(r["start"] - a, r["prev_end"])


def apply_end(bf, r, shifts, quiet, cap, pause_min) -> float:
    b = shifts["end"].get(bf.letter_class(bf.edge_letter(r["word"], "end")), shifts["end"]["_"])
    end = r["end"] + b
    if r["gap_after"] >= pause_min and r.get("env") is not None:
        limit = min(r["end"] + cap, r["next_start"] if r["next_start"] is not None else r["end"] + cap)
        end = max(end, decay_end(r["env"], r["start"], r["end"], limit, quiet))
    if r["next_start"] is not None:
        end = min(end, r["next_start"])
    return max(end, r["start"] + 0.01)


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
        for r in [r for r in rows if (r["clip"] in half) != train_in_half]:
            s = apply_start(bf, r, model["shifts"])
            e = apply_end(bf, r, model["shifts"], model["quiet"], model["cap"], model["pause_min"])
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
        by_key: dict[str, list] = {}
        for r in rows:
            key = f"{r['clip']}#{r['who']}"
            by_key.setdefault(key, []).append(r)
        written = args.run / "labels" / f"{args.write}.jsonl"
        with written.open("w", encoding="utf-8", newline="\n") as fh:
            for key, rs in by_key.items():
                words = [{"word": r["word"],
                          "start": round(apply_start(bf, r, model["shifts"]), 4),
                          "end": round(apply_end(bf, r, model["shifts"], model["quiet"],
                                                 model["cap"], model["pause_min"]), 4),
                          **({"score": r["score"]} if r.get("score") is not None else {})}
                         for r in sorted(rs, key=lambda r: r["start"])]
                fh.write(json.dumps({"id": key, "words": words}, ensure_ascii=False) + "\n")
        print(f"-> {written}")


if __name__ == "__main__":
    main()
