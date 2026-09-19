"""Whisper-based forced alignment: stable-ts align() on faster-whisper.

    python stable_ts.py --manifest clips.jsonl --out stable_ts_turbo.jsonl \\
        --model ivrit-ai/whisper-large-v3-turbo-ct2

align() times the transcript we already have rather than transcribing, so it does the same
job as the other aligners. ivrit.ai's Hebrew fine-tunes are published in faster-whisper
(CTranslate2) format; `ivrit-ai/whisper-large-v3-ct2` is the full-size alternative.

Whisper splits the text into words its own way, so its words are checked against the
dataset's word list after stripping punctuation, and a clip is only kept if they agree word
for word. Checking the count alone would accept a merge in one place cancelled by a split in
another, and pin every word after it to the wrong time.

On Windows, torch and ctranslate2 ship different OpenMP runtimes and Intel's refuses to share
a process. Run with KMP_DUPLICATE_LIB_OK=TRUE and OMP_NUM_THREADS=1: the second removes the
competing thread pools that make the first unsafe, and the GPU does the work anyway.
run_eval.py sets both. Needs numpy<2 alongside torch 2.4.

Input rows: {id, audio, duration, text, words}. Output rows: {id, words}. Clips already in
--out are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def bare(word: str) -> str:
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFC", word))


def done_ids(out: Path) -> set[str]:
    """Clips already aligned, plus ones that failed before. A failure here is deterministic
    (digits, an unspellable word), so retrying it every run only costs time; delete the
    .failed file next to --out to try them again."""
    ids = set()
    for path in (out, failed_path(out)):
        if path.exists():
            ids |= {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return ids


def failed_path(out: Path) -> Path:
    return out.with_suffix(".failed.jsonl")


def record_failures(out: Path, failed) -> None:
    if failed:
        with failed_path(out).open("a", encoding="utf-8", newline="\n") as handle:
            for cid, why in failed:
                handle.write(json.dumps({"id": cid, "why": why}, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="ivrit-ai/whisper-large-v3-turbo-ct2")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip = done_ids(args.out)
    rows = [r for r in rows if r["id"] not in skip]
    print(f"{args.model}: {len(rows)} clips to align ({len(skip)} already done)", flush=True)
    if not rows:
        return

    import stable_whisper

    model = stable_whisper.load_faster_whisper(
        args.model, device=args.device, compute_type="float16" if args.device == "cuda" else "int8"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok, disagreed, failed = 0, [], []
    with args.out.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            tokens = row.get("words") or row["text"].split()
            wanted = [(t, bare(t)) for t in tokens if bare(t)]
            try:
                result = model.align(row["audio"], row["text"], language="he", verbose=None)
            except Exception as exc:  # noqa: BLE001 -- one clip must not stop the rest
                failed.append((row["id"], f"{type(exc).__name__}: {exc}"[:100]))
                continue
            got = [w for seg in result.segments for w in seg.words if bare(w.word)]
            if [b for _, b in wanted] != [bare(w.word) for w in got]:
                disagreed.append((row["id"], len(wanted), len(got)))
                continue
            timed = [{"word": t, "start": round(float(w.start), 4), "end": round(float(w.end), 4)}
                     for (t, _), w in zip(wanted, got)]
            handle.write(json.dumps({"id": row["id"], "words": timed}, ensure_ascii=False) + "\n")
            ok += 1

    record_failures(args.out, failed + [(c, f'disagreed {a} vs {b}') for c, a, b in disagreed])
    print(f"  aligned {ok}, word lists disagreed {len(disagreed)}, failed {len(failed)}", flush=True)
    for cid, n_want, n_got in disagreed:
        print(f"  DISAGREED {cid}: dataset {n_want} words, whisper {n_got}")
    for cid, why in failed:
        print(f"  FAILED {cid}: {why}")


if __name__ == "__main__":
    main()
