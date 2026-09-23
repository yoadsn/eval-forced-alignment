"""Add the `mms-corrected` label to a dataset's manifest, from its own `mms` label.

    python eval/push_corrected.py --manifest live.jsonl --audio data/datasets/ivrit-ai \
        --correction data/eval_runs/ivrit-ai-corrected/correction.json --out merged.jsonl

Reads each clip's existing `mms` timings, applies the correction (a shift per letter class
at every boundary, plus the energy extension on word ends that have a pause after them) and
writes the result back as a second label beside it. Clips without an `mms` label are copied
through untouched, so a manifest where only some clips have been aligned stays valid.

Nothing is uploaded here; the caller does that, after looking at the diff.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True, help="Dataset folder holding audio/.")
    p.add_argument("--correction", type=Path, required=True)
    p.add_argument("--source", default="mms", help="Label to correct.")
    p.add_argument("--name", default="mms-corrected", help="Label to write.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    bf = load("bf", ROOT / "eval" / "boundary_fix.py")
    cm = load("cm", ROOT / "eval" / "correct_mms.py")
    model = json.loads(args.correction.read_text(encoding="utf-8"))

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = skipped = moved = 0
    total_shift = 0.0
    for entry in rows:
        src = next((lb for lb in entry["labels"] if lb["source"] == args.source), None)
        if not src or not src["words"]:
            skipped += 1
            continue
        env = cm.envelope(args.audio / entry["audio"])
        ws = src["words"]
        out = []
        for i, w in enumerate(ws):
            r = {"word": w["word"], "start": w["start"], "end": w["end"],
                 "prev_end": ws[i - 1]["end"] if i else 0.0,
                 "next_start": ws[i + 1]["start"] if i + 1 < len(ws) else None,
                 "gap_after": (ws[i + 1]["start"] if i + 1 < len(ws) else w["end"]) - w["end"],
                 "env": env}
            start = cm.apply_start(bf, r, model["shifts"])
            end = cm.apply_end(bf, r, model["shifts"], model["quiet"], model["cap"], model["pause_min"])
            total_shift += abs(start - w["start"]) + abs(end - w["end"])
            moved += 1
            out.append({**{k: v for k, v in w.items() if k not in ("start", "end")},
                        "start": round(start, 4), "end": round(end, 4)})
        entry["labels"] = [lb for lb in entry["labels"] if lb["source"] != args.name]
        entry["labels"].append({"source": args.name, "words": out})
        done += 1

    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} clips: {done} corrected, {skipped} had no {args.source!r} label and were left alone")
    print(f"{moved} words moved, {total_shift / max(moved, 1) * 1000:.1f} ms per boundary on average")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
