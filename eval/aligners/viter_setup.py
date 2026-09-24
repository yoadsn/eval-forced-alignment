"""Lay out a corpus and a pronunciation dictionary for viter (the MFA recipe).

    python eval/aligners/viter_setup.py --dataset data/datasets/ivrit-ai --out data/viter

viter wants a directory of audio with a same-named `.txt` beside each file, and a
dictionary mapping every word to its phones. Without a dictionary it treats each whole
token as a single phone, which for Hebrew means one phone per word and no alignment worth
having -- so the dictionary is the part that matters.

Hebrew spelling does not carry its vowels, so the dictionary is built with Phonikud
(arXiv:2506.12311), which turns unvocalized Hebrew into IPA. This is the one place in this
project where phonemes are the right tool: MFA-style training learns an acoustic model *of
the phones it is given*, unlike MMS, which was trained on a romanization of the spelling and
must be fed that same romanization.

Needs the ctc-env interpreter (it has phonikud); see eval/README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

STRESS, PREFIX = "ˈ", "|"
# Two-character phones first, so they are not split into their halves.
DIGRAPHS = ("ts", "tʃ", "dʒ")
_PUNCT = re.compile(r"[^\w֐-׿']+", re.UNICODE)


def split_phones(ipa: str) -> list[str]:
    """IPA string to a list of phones, dropping the stress mark and the prefix bar."""
    s = "".join(c for c in ipa if c not in (STRESS, PREFIX) and not c.isspace())
    out, i = [], 0
    while i < len(s):
        two = s[i:i + 2]
        if two in DIGRAPHS:
            out.append(two)
            i += 2
        else:
            out.append(s[i])
            i += 1
    return out


def clean_word(w: str) -> str:
    """The token as it will appear in both the transcript and the dictionary.

    They have to agree exactly or the word is out of vocabulary and gets the OOV phone,
    which silently ruins the alignment around it.
    """
    return _PUNCT.sub("", w).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", type=Path, help="Optional jsonl of clip ids to include (marks).")
    args = p.parse_args()

    rows = [json.loads(l) for l in (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.only:
        keep = {json.loads(l)["id"] for l in args.only.read_text(encoding="utf-8").splitlines() if l.strip()}
        rows = [r for r in rows if r["id"] in keep]

    corpus = args.out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    vocab: set[str] = set()
    for r in rows:
        src = args.dataset / r["audio"]
        # One directory per clip, because viter reads the parent directory as the speaker
        # id. Flat, the whole corpus is a single speaker and one fMLLR transform is fitted
        # across hundreds of different Knesset voices -- wrong for every one of them, and it
        # failed a third of the alignments outright.
        room = corpus / r["id"]
        room.mkdir(parents=True, exist_ok=True)
        dst = room / f"{r['id']}{src.suffix}"
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            shutil.copyfile(src, dst)
        words = [clean_word(w) for w in r["text"].split()]
        words = [w for w in words if w]
        vocab.update(words)
        (room / f"{r['id']}.txt").write_text(" ".join(words) + "\n", encoding="utf-8")

    from huggingface_hub import hf_hub_download
    from phonikud import phonemize
    from phonikud_onnx import Phonikud

    model = Phonikud(hf_hub_download("thewh1teagle/phonikud-onnx", "phonikud-1.0.int8.onnx"))
    lines, failed = [], 0
    for w in sorted(vocab):
        try:
            phones = split_phones(phonemize(model.add_diacritics(w)))
        except Exception:  # noqa: BLE001 -- one word must not stop the dictionary
            phones = []
        if not phones:
            failed += 1
            continue
        lines.append(f"{w}\t{' '.join(phones)}")
    dictionary = args.out / "dict.txt"
    dictionary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    phones = sorted({p for line in lines for p in line.split("\t")[1].split()})
    print(f"{len(rows)} clips -> {corpus}")
    print(f"{len(lines)} words in {dictionary} ({failed} could not be phonemized)")
    print(f"{len(phones)} distinct phones: {' '.join(phones)}")


if __name__ == "__main__":
    main()
