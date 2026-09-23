"""Phonemize every word of the gold clips with Phonikud, once, into a cache.

    python eval/phonemize_words.py --run data/eval_runs/ivrit-ai-corrected \
        --out data/unclaimed/phonemes.json

Hebrew orthography hides the things a boundary correction needs. The letter ה ends המצאה but
the sound is the vowel /a/; ו is a consonant in some words and a vowel in others; stress is
never written. Phonikud (arXiv 2506.12311) resolves all three, so the correction can key on
the phoneme that was actually spoken instead of on the letter that was written.

Words are phonemized a sentence at a time, because diacritization is context-dependent, and
the result is only kept when it comes back with the same number of words. The model runs on
CPU through ONNX; see eval/README.md for the install.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    from phonikud import phonemize
    from phonikud_onnx import Phonikud

    model = Phonikud(hf_hub_download("thewh1teagle/phonikud-onnx", "phonikud-1.0.int8.onnx"))

    clips = [json.loads(line) for line in (args.run / "clips.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    out: dict[str, list] = {}
    mismatched = 0
    for c in clips:
        words = c["words"]
        try:
            ipa = phonemize(model.add_diacritics(" ".join(words))).split()
        except Exception:  # noqa: BLE001 -- one clip must not stop the rest
            ipa = []
        if len(ipa) != len(words):
            # Context-dependent diacritization can merge or split; fall back to one word at a
            # time, which loses context but keeps the mapping honest.
            mismatched += 1
            ipa = []
            for w in words:
                try:
                    ipa.append(phonemize(model.add_diacritics(w)).replace(" ", ""))
                except Exception:  # noqa: BLE001
                    ipa.append("")
        out[c["id"]] = ipa
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} clips phonemized ({mismatched} needed the word-by-word fallback) -> {args.out}")


if __name__ == "__main__":
    main()
