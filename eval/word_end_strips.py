"""Every word end that sits before a pause, drawn: the audio, MMS's end, the human's.

    python eval/word_end_strips.py --run data/eval_runs/ivrit-ai-corrected --out ends.png

One strip per word, 300 ms of audio before MMS's end and 500 ms after, so the pause is
visible. Blue is where MMS ended the word, green is where a person did, and the band between
them is what the disagreement actually looks like against the waveform.

Sorted by how far apart they are, so the top of the list is where MMS stops earliest.

Hebrew is written into the labels reversed, because matplotlib lays glyphs out left to right
with no bidi support: reversing makes a pure-Hebrew word read correctly on the page.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BEFORE, AFTER = 0.30, 0.50
MMS_C, HUM_C, FIX_C, WAVE_C = "#2563eb", "#15803d", "#b45309", "#c7cbd1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ivrit-ai"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-pause", type=float, default=0.3)
    p.add_argument("--columns", type=int, default=3)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    spec = importlib.util.spec_from_file_location("bf", ROOT / "eval" / "boundary_fix.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)

    # The correction is recomputed here rather than read from the written labels, so the
    # picture cannot drift from what eval/correct_mms.py actually does.
    cm_spec = importlib.util.spec_from_file_location("cm", ROOT / "eval" / "correct_mms.py")
    cm = importlib.util.module_from_spec(cm_spec)
    cm_spec.loader.exec_module(cm)
    all_rows = cm.attach_envelopes(bf.build(args.run, "mms", {"probe"}), args.dataset)
    model = json.loads((args.run / "correction.json").read_text(encoding="utf-8"))
    placed = cm.corrected(bf, all_rows, model)
    for r in all_rows:
        r["fixed_end"] = placed[id(r)][1]
    rows = [r for r in all_rows if r["gap_after"] >= args.min_pause]
    rows.sort(key=lambda r: -(abs(r["h_end"] - r["end"]) - abs(r["h_end"] - r["fixed_end"])))
    entries = {}
    for line in (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            entries[e["id"]] = e

    audio: dict[str, tuple] = {}

    def clip_audio(cid):
        if cid not in audio:
            wav, rate = sf.read(args.dataset / entries[cid]["audio"], dtype="float32", always_2d=True)
            audio[cid] = (wav.mean(axis=1), rate)
        return audio[cid]

    per = -(-len(rows) // args.columns)
    fig, axes = plt.subplots(per, args.columns, figsize=(5.4 * args.columns, 0.42 * per))
    axes = np.atleast_2d(axes)
    for n, r in enumerate(rows):
        ax = axes[n % per][n // per]
        wav, rate = clip_audio(r["clip"])
        t0, t1 = r["end"] - BEFORE, r["end"] + AFTER
        seg = wav[max(0, int(t0 * rate)):int(t1 * rate)]
        if len(seg) < 10:
            continue
        # An envelope, not the raw samples: at this height the waveform is a smear, while the
        # envelope shows where the sound actually stops, which is the whole question here.
        win = max(1, int(0.005 * rate))
        env = np.abs(seg[: len(seg) // win * win].reshape(-1, win)).max(axis=1)
        env = env / (env.max() or 1)
        t = np.linspace(t0, t0 + len(env) * win / rate, len(env))
        ax.fill_between(t, 0, env, color=WAVE_C, linewidth=0)
        ax.axvline(r["end"], color=MMS_C, linewidth=1.4)
        ax.axvline(r["fixed_end"], color=FIX_C, linewidth=1.8)
        ax.axvline(r["h_end"], color=HUM_C, linewidth=1.4)
        lo, hi = sorted((r["end"], r["h_end"]))
        ax.axvspan(lo, hi, color=HUM_C, alpha=.10, linewidth=0)
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_xticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        was = abs(r["h_end"] - r["end"]) * 1000
        now = abs(r["h_end"] - r["fixed_end"]) * 1000
        ax.text(t0, 1.0, r["word"][::-1], fontsize=8, va="top", ha="left", color="#1b1f24")
        ax.text(t1, 1.0, f"{was:.0f} -> {now:.0f} ms", fontsize=8, va="top", ha="right",
                color=FIX_C if now < was - 1 else ("#9ca3af" if now < was + 1 else MMS_C))
    for n in range(len(rows), per * args.columns):
        axes[n % per][n // per].set_visible(False)

    fig.suptitle(f"Word ends before a pause of {args.min_pause * 1000:.0f} ms or more  ·  "
                 f"{len(rows)} words  ·  blue = MMS, orange = corrected, green = person",
                 fontsize=13, x=0.005, ha="left", color="#1b1f24")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor="white")
    print(f"{len(rows)} words -> {args.out}")


if __name__ == "__main__":
    main()
