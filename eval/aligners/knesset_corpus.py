"""Build a viter training corpus from the ivrit-ai Knesset plenum shards on HuggingFace.

    python eval/aligners/knesset_corpus.py --shards 1 --out D:/knesset-train \\
        --exclude-recordings data/datasets/ivrit-ai/manifest.jsonl

One shard is about a thousand 30-second slices, so roughly 8.6 hours -- seventeen times the
31 minutes the first honest MFA model got, which is the whole point of fetching them.

Three things have to happen to the rows before viter can use them:

- **the transcript carries Whisper timestamp tokens** (`<|0.24|> text <|5.46|>`), which are
  not words and would be trained as if they were;
- **the audio is MP3 inside the parquet**, and the corpus wants files on disk;
- **our own evaluation clips may be in here.** They came from these same recordings, so any
  row whose `entry_id` matches a `recording` id in our gold set is dropped. Without that the
  acoustic model hears the test audio again, only at a larger scale and less visibly.

Needs the ctc-env interpreter (soundfile, pyarrow, huggingface_hub, phonikud) and acceptance
of the ivrit.ai licence on the dataset page.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

REPO = "ivrit-ai/knesset-plenums-whisper-training"
TIMESTAMP = re.compile(r"<\|[\d.]+\|>")


def clean_transcript(text: str) -> str:
    """The words alone: timestamp tokens out, whitespace normalised."""
    return " ".join(TIMESTAMP.sub(" ", text or "").split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--first-shard", type=int, default=0)
    p.add_argument("--exclude-recordings", type=Path,
                   help="A dataset manifest whose metadata.recording ids must not appear.")
    p.add_argument("--min-quality", type=float, default=0.8)
    p.add_argument("--min-words", type=int, default=4)
    args = p.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    banned: set[str] = set()
    if args.exclude_recordings:
        for line in args.exclude_recordings.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line).get("metadata", {}).get("recording")
                if rec is not None:
                    banned.add(str(rec))
        print(f"excluding {len(banned)} recording ids that our gold set uses")

    corpus = args.out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    kept = dropped_ours = dropped_quality = 0
    seconds = 0.0
    vocab: set[str] = set()

    for n in range(args.first_shard, args.first_shard + args.shards):
        name = f"data/train-{n:05d}-of-00834.parquet"
        path = hf_hub_download(REPO, name, repo_type="dataset")
        table = pq.read_table(path)
        print(f"{name}: {table.num_rows} rows", flush=True)
        cols = {c: table.column(c) for c in table.column_names}
        for i in range(table.num_rows):
            meta = cols["metadata"][i].as_py() or {}
            entry = str(meta.get("entry_id", ""))
            if entry in banned:
                dropped_ours += 1
                continue
            if float(meta.get("quality_score") or 0) < args.min_quality:
                dropped_quality += 1
                continue
            words = clean_transcript(cols["transcript"][i].as_py()).split()
            if len(words) < args.min_words:
                dropped_quality += 1
                continue
            blob = cols["audio"][i].as_py()
            data, rate = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=True)
            utt = f"{entry}-{i:05d}"
            # One directory per utterance: viter reads the parent directory as the speaker,
            # and these are all different people.
            room = corpus / utt
            room.mkdir(exist_ok=True)
            sf.write(room / f"{utt}.wav", data.mean(axis=1), rate)
            (room / f"{utt}.txt").write_text(" ".join(words) + "\n", encoding="utf-8")
            vocab.update(words)
            seconds += len(data) / rate
            kept += 1

    print(f"{kept} utterances, {seconds / 3600:.2f} hours -> {corpus}")
    print(f"  {dropped_ours} dropped as ours, {dropped_quality} dropped as short or low quality")
    (args.out / "vocab.json").write_text(json.dumps(sorted(vocab), ensure_ascii=False), encoding="utf-8")
    print(f"  {len(vocab)} distinct words -> {args.out / 'vocab.json'}")


if __name__ == "__main__":
    main()
