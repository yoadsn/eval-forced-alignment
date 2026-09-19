"""MWA -- Multilingual Word Aligner (Weber et al., arXiv:2606.10675), the align button's model.

    python mwa.py --manifest clips.jsonl --out mwa.jsonl \\
        --mwa-repo path/to/Multilingual-Word-Aligner --model buckeye

Runs MWA's own pipeline from a clone of github.com/MLSpeech/Multilingual-Word-Aligner, in that
repo's environment. Its checkpoints download from Hugging Face on first use: `buckeye` is
trained on conversational speech, `timit` on read speech.

Three things its own entry point (align_wav.py) does not handle, done here:
- It stops the whole batch at the first clip it cannot align. Here each clip is caught.
- MMS's alphabet has no punctuation and no digits. Punctuation is stripped from each word
  before it is sent; a clip containing digits fails and is reported, not guessed at.
- It returns words romanized ("vshvtp"), so rows are put back onto the dataset's words by
  position -- same number in as out, or the clip is left out.

On Windows it opens transcripts with no encoding: run with PYTHONUTF8=1 (run_eval.py does).

Input rows: {id, audio, duration, text, words}. Output rows: {id, words}. Clips already in
--out are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


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
    p.add_argument("--mwa-repo", type=Path, required=True)
    p.add_argument("--model", default="buckeye", choices=["buckeye", "timit"])
    args = p.parse_args()
    out = args.out.resolve()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip = done_ids(out)
    rows = [r for r in rows if r["id"] not in skip]
    print(f"mwa-{args.model}: {len(rows)} clips to align ({len(skip)} already done)", flush=True)
    if not rows:
        return

    import librosa
    import soundfile as sf

    work = Path(tempfile.mkdtemp(prefix="mwa_"))
    wav_dir, csv_dir = work / "in", work / "out"
    wav_dir.mkdir()
    csv_dir.mkdir()
    kept: dict[str, list[str]] = {}
    for row in rows:
        wav, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sf.write(wav_dir / f"{row['id']}.wav", wav, 16000, subtype="PCM_16")
        tokens = row.get("words") or row["text"].split()
        kept[row["id"]] = [t for t in tokens if re.sub(r"[^\w]", "", t)]
        (wav_dir / f"{row['id']}.txt").write_text(
            " ".join(re.sub(r"[^\w]", "", t) for t in kept[row["id"]]), encoding="utf-8"
        )

    # MWA resolves its checkpoints and configs relative to its own repo.
    sys.path.insert(0, str(args.mwa_repo.resolve()))
    os.chdir(args.mwa_repo)
    sys.argv = ["align_wav.py", "--wav_input", str(wav_dir), "--transcript_input", str(wav_dir),
                "--language", "heb", "--model_name", args.model, "--output_folder", str(csv_dir)]
    import inference.configuration.constants as constants
    from inference.configuration.get_models_configuration import get_models_configurations
    from inference.configuration.validate_input import UserInput, get_input_parser
    from inference.models.dp_algorithm.extract_features import Features_DP
    from inference.models.mms.mms import load_mms_model
    from inference.models.predict import get_file_prediction
    from inference.models.preprocess import prepare_dataset
    from inference.models.unsupSeg.unsupseg_classifier import load_unsupseg_model
    from inference.models.utils import find_fit_transcript, load_model, prepare_sentence
    from inference.results_utils.graphs import save_results_to_csv

    config = get_models_configurations(user_parameters=UserInput(**get_input_parser().__dict__))
    model = load_model(**config)
    config["mms_bundle"], config["mms_model"] = load_mms_model(config["device"])
    config["unsupseg_model"] = load_unsupseg_model(
        os.path.join(constants.INFERENCE_PART_DIR, config["unsupseg_ckpt"]), config["device"])
    config["_dp_features_obj"] = Features_DP(config["dp_features"])
    config["_w_floats"] = list(config["w"])

    failed = []
    for wav_file in config["wav_input"]:
        cid = Path(str(wav_file)).stem
        try:
            config["wav_file"] = str(wav_file)
            config["transcript_file"] = find_fit_transcript(wav_file, config["transcript_input"])
            sentence = prepare_sentence(config["transcript_file"], language="heb")
            batches, masks = prepare_dataset(**config)
            _, _, _, times = get_file_prediction(model, batches, masks, sentence, **config)
            save_results_to_csv(sentence, times, **config)
        except Exception as exc:  # noqa: BLE001 -- one clip must not stop the rest
            has_digit = bool(re.search(r"\d", " ".join(kept.get(cid, []))))
            failed.append((cid, "has digits" if has_digit else f"{type(exc).__name__}: {exc}"[:90]))

    ok, mismatched = 0, []
    with out.open("a", encoding="utf-8", newline="\n") as handle:
        for cid, tokens in kept.items():
            path = csv_dir / f"{cid}.csv"
            if not path.exists():
                continue
            got = list(csv.DictReader(path.open(encoding="utf-8")))
            if len(got) != len(tokens):
                mismatched.append(cid)
                continue
            timed = [{"word": t, "start": float(r["Start_Time"]), "end": float(r["End_Time"])}
                     for t, r in zip(tokens, got)]
            handle.write(json.dumps({"id": cid, "words": timed}, ensure_ascii=False) + "\n")
            ok += 1
    record_failures(out, failed + [(c, 'row-count mismatch') for c in mismatched])
    print(f"  aligned {ok}, failed {len(failed)}, row-count mismatches {len(mismatched)}", flush=True)
    for cid, why in failed:
        print(f"  FAILED {cid}: {why}")


if __name__ == "__main__":
    main()
