"""Cut out the audio no word claims, loudest first, so it can be listened to.

    python eval/unclaimed_export.py --gaps data/unclaimed/corrected.jsonl --out data/unclaimed/clips

Aligned on a transcript people have corrected, every word in the text is right. So whatever
speech is left over is, by construction, not a word of the transcript: a hesitation, a
breath, another speaker, or a word the correction itself missed. Before treating unclaimed
speech as evidence of a missing word, we need to know which of those it usually is -- and
the only way to know is to hear them.

Each cut carries 300 ms of context either side, so the stretch can be heard in place. The
index lists what the model would have decoded there, which is the quickest way to tell a
filler from a word: "eh" comes out as a vowel or two, a word as a word.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PAD = 0.3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gaps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-speech", type=float, default=0.05,
                   help="Seconds of speech in the stretch, below which it is silence and not "
                        "worth hearing (default 0.05).")
    p.add_argument("--limit", type=int, default=60)
    args = p.parse_args()

    import soundfile as sf

    rows = [json.loads(line) for line in args.gaps.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = []
    for r in rows:
        for g in r["gaps"]:
            if g["speech"] >= args.min_speech:
                picked.append((r["id"], r["audio"], g))
    picked.sort(key=lambda t: -t[2]["speech"])
    picked = picked[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for n, (cid, audio, g) in enumerate(picked, 1):
        wav, rate = sf.read(audio, dtype="float32", always_2d=True)
        a = max(0, int((g["start"] - PAD) * rate))
        b = min(len(wav), int((g["end"] + PAD) * rate))
        name = f"{n:02d}_{round(g['speech'] * 1000)}ms_{cid.split('#')[0]}.wav"
        sf.write(args.out / name, wav[a:b].mean(axis=1), rate)
        index.append({"file": name, "clip": cid, "start": g["start"], "end": g["end"],
                      "gap_ms": round(g["dur"] * 1000), "speech_ms": round(g["speech"] * 1000),
                      "speech_pct": round(g["speech_frac"] * 100), "decoded": g["letters"]})
    with (args.out / "index.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"{len(picked)} stretches -> {args.out}  (context {PAD * 1000:.0f} ms either side)")


if __name__ == "__main__":
    main()
