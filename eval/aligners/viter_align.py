"""MFA-style alignment with viter, the Montreal Forced Aligner recipe in one binary.

    python viter_align.py --manifest clips.jsonl --out mfa-viter.jsonl \\
        --model data/viter/hebrew.viter --dict data/viter/dict.txt

MFA itself was never in this comparison because it needs a pronunciation dictionary and an
acoustic model per language and there is no official Hebrew one -- the MWA paper hits the
same wall and drops MFA for Hebrew. viter is that recipe reimplemented, and it will train
its own model, so the gap is closable: `viter_setup.py` builds the corpus and a Phonikud
dictionary, `viter train` fits the model, and this runs the alignment.

Any word the dictionary is missing is added here rather than left to the OOV phone, which
would silently wreck the alignment around it. Words that cannot be phonemized at all are
reported and their clip is skipped.

Input rows: {id, audio, duration, text, words}. Output rows: {id, words}. Clips already in
--out are skipped, so a re-run after more tagging aligns only the new ones.

Needs the viter-env interpreter (viter + phonikud); see eval/README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from viter_setup import clean_word, split_phones  # noqa: E402


def done_ids(out: Path) -> set[str]:
    ids = set()
    for path in (out, failed_path(out)):
        if path.exists():
            ids |= {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    return ids


def failed_path(out: Path) -> Path:
    return out.with_suffix(".failed.jsonl")


def read_textgrid_words(path: Path) -> list[tuple[float, float, str]]:
    """The intervals of the `words` tier, silences dropped.

    Parsed rather than pulled in with a library: the file is a handful of numbers and this
    keeps the aligner's environment down to viter itself.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find('name = "words"')
    if start < 0:
        return []
    chunk = text[start:]
    nxt = chunk.find('name = "', 14)
    if nxt > 0:
        chunk = chunk[:nxt]
    out = []
    for m in re.finditer(r"xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*\"(.*?)\"",
                         chunk, re.S):
        word = m.group(3).strip()
        if word:
            out.append((float(m.group(1)), float(m.group(2)), word))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dict", dest="dictionary", type=Path, required=True)
    p.add_argument("--viter", default=str(HERE.parent.parent / "viter-env" / "Scripts" / "viter.exe"))
    # The defaults (10/40) leave a fifth of these clips unaligned. Widening costs almost
    # nothing at this size -- the whole 300-clip corpus aligns in seconds either way -- and
    # at 100/800 every one of them goes through. A clip dropped for a narrow beam is a clip
    # silently missing from the comparison, which is worse than a slow one.
    p.add_argument("--beam", default="100")
    p.add_argument("--retry-beam", dest="retry_beam", default="800")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    skip = done_ids(args.out)
    rows = [r for r in rows if r["id"] not in skip]
    print(f"viter: {len(rows)} clips to align ({len(skip)} already done)", flush=True)
    if not rows:
        return

    # 1. the words this manifest needs, and whatever the dictionary is missing
    entries = {}
    for line in args.dictionary.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            entries[line.split("\t")[0]] = line.split("\t")[1]
    needed = {clean_word(w) for r in rows for w in (r.get("words") or r["text"].split())}
    needed = {w for w in needed if w}
    missing = sorted(needed - set(entries))
    if missing:
        from huggingface_hub import hf_hub_download
        from phonikud import phonemize
        from phonikud_onnx import Phonikud

        model = Phonikud(hf_hub_download("thewh1teagle/phonikud-onnx", "phonikud-1.0.int8.onnx"))
        added = 0
        for w in missing:
            try:
                phones = split_phones(phonemize(model.add_diacritics(w)))
            except Exception:  # noqa: BLE001
                phones = []
            if phones:
                entries[w] = " ".join(phones)
                added += 1
        args.dictionary.write_text(
            "\n".join(f"{w}\t{p}" for w, p in sorted(entries.items())) + "\n", encoding="utf-8")
        print(f"  dictionary: {added} of {len(missing)} missing words added", flush=True)

    # 2. a corpus of just these clips. '#' separates clip from annotator in the id and is
    #    not wanted in a filename, so it is swapped for a marker and swapped back after.
    work = Path(tempfile.mkdtemp(prefix="viter_"))
    corpus, aligned = work / "in", work / "out"
    corpus.mkdir()
    names, failed = {}, []
    for r in rows:
        words = [clean_word(w) for w in (r.get("words") or r["text"].split())]
        words = [w for w in words if w]
        if not words or any(w not in entries for w in words):
            failed.append((r["id"], "a word is not in the dictionary"))
            continue
        name = r["id"].replace("#", "--")
        names[name] = r
        src = Path(r["audio"])
        # Its own directory, so viter treats it as its own speaker: these are all different
        # people, and one pooled fMLLR transform failed a third of them outright.
        room = corpus / name
        room.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, room / f"{name}{src.suffix}")
        (room / f"{name}.txt").write_text(" ".join(words) + "\n", encoding="utf-8")

    if names:
        subprocess.run([args.viter, "align", str(corpus), str(args.model),
                        "--dict", str(args.dictionary), "--beam", args.beam,
                        "--retry-beam", args.retry_beam, "-o", str(aligned)], check=False)

    # 3. back onto the manifest's own words, by position
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with args.out.open("a", encoding="utf-8", newline="\n") as handle:
        for name, r in names.items():
            grid = next(aligned.rglob(f"{name}.TextGrid"), None) if aligned.exists() else None
            if grid is None:
                failed.append((r["id"], "viter produced no TextGrid"))
                continue
            spans = read_textgrid_words(grid)
            original = [w for w in (r.get("words") or r["text"].split()) if clean_word(w)]
            if len(spans) != len(original):
                failed.append((r["id"], f"{len(spans)} aligned words for {len(original)}"))
                continue
            timed = [{"word": original[i], "start": round(s, 4), "end": round(e, 4)}
                     for i, (s, e, _) in enumerate(spans)]
            handle.write(json.dumps({"id": r["id"], "words": timed}, ensure_ascii=False) + "\n")
            ok += 1

    if failed:
        with failed_path(args.out).open("a", encoding="utf-8", newline="\n") as fh:
            for cid, why in failed:
                fh.write(json.dumps({"id": cid, "why": why}, ensure_ascii=False) + "\n")
    print(f"  aligned {ok}, failed {len(failed)}", flush=True)
    for cid, why in failed[:10]:
        print(f"  FAILED {cid}: {why}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
