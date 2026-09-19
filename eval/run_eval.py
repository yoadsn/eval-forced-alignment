"""Score every aligner against the human marks, end to end, re-runnably.

    python eval/run_eval.py --marks gold.jsonl --dataset data/datasets/ivrit-ai

  --marks    the human marks, as the "download marks" button (GET /api/export) saves them
  --dataset  a dataset folder (DATASET_SPEC.md); fetch one with
             python -m hebrew_training.download_dataset --dataset ivrit-ai

For every clip someone has marked it runs each aligner, attaches their word timings as labels
next to the dataset's own, scores all of them against the humans (hebrew_training/
aligner_eval.py -- the same code behind the dashboard), and writes a run folder:

    <run>/clips.jsonl          the marked clips handed to the aligners
    <run>/labels/<name>.jsonl  each aligner's output, one row per clip
    <run>/dataset/             a dataset of just the marked clips, every label attached
    <run>/marks/<name>.jsonl   the marks split per annotator
    <run>/eval.json            the full result, significance tests included

Re-running is cheap: each aligner skips clips it already has, so after more tagging only the
new clips are aligned. Delete a labels file to redo that aligner from scratch.

The aligners need incompatible environments, so each runs in its own interpreter, set once in
.env (see eval/README.md):
    EVAL_PY_CTC        torch + transformers + uroman           -> hebrew, mms
    EVAL_PY_STABLE_TS  torch 2.4 + stable-ts + faster-whisper   -> stable_ts_turbo
    EVAL_PY_MWA        the MWA repo's own environment           -> mwa
    EVAL_MWA_REPO      path to a clone of MLSpeech/Multilingual-Word-Aligner
An aligner whose interpreter is not set is skipped, with a note -- the rest still run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# name -> (runner script, interpreter variable, extra args, extra environment)
ALIGNERS = {
    "hebrew": ("ctc.py", "EVAL_PY_CTC", ["--model", "imvladikon/wav2vec2-xls-r-300m-hebrew"], {}),
    "mms": ("ctc.py", "EVAL_PY_CTC", ["--model", "MahmoudAshraf/mms-300m-1130-forced-aligner"], {}),
    "stable_ts_turbo": ("stable_ts.py", "EVAL_PY_STABLE_TS",
                        ["--model", "ivrit-ai/whisper-large-v3-turbo-ct2"],
                        # see stable_ts.py: two OpenMP runtimes, made safe by one thread each
                        {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1", "PYTHONUTF8": "1"}),
    "mwa": ("mwa.py", "EVAL_PY_MWA", ["--model", "buckeye"], {"PYTHONUTF8": "1"}),
}


def per_annotator_counts(marks) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for m in marks:
        who = m.get("annotator", "anon")
        counts[who] = counts.get(who, 0) + 1
    return list(counts.items())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--marks", type=Path, required=True, help="Human marks from /api/export.")
    p.add_argument("--dataset", type=Path, required=True, help="Dataset folder (manifest.jsonl, audio/).")
    p.add_argument("--run", type=Path, help="Run folder (default: data/eval_runs/<dataset name>).")
    p.add_argument("--aligners", default=",".join(ALIGNERS),
                   help=f"Comma-separated subset of: {', '.join(ALIGNERS)}.")
    p.add_argument(
        "--text",
        choices=["dataset", "corrected"],
        default="dataset",
        help="Which transcript the aligners are given. 'dataset' is the original one, the "
        "realistic case -- it is what they would get over a whole untagged corpus, missing "
        "words and all. 'corrected' is each annotator's edited word list, which measures "
        "timing skill alone; a clip is then aligned once per annotator, since two people do "
        "not always correct it the same way.",
    )
    p.add_argument("--exclude", action="append", default=[],
                   help="Annotator to leave out of scoring, e.g. a test account. Repeatable.")
    args = p.parse_args()

    dataset = args.dataset.resolve()
    run = (args.run or ROOT / "data" / "eval_runs" / dataset.name).resolve()
    entries = {e["id"]: e for e in load_jsonl(dataset / "manifest.jsonl")}
    marks = load_jsonl(args.marks)
    marked = sorted({m["id"] for m in marks if m.get("id") in entries})
    unknown = sorted({m.get("id") for m in marks} - set(entries))
    print(f"{len(marks)} marks on {len(marked)} clips of {dataset.name} "
          f"({len(entries)} clips in the dataset)")
    if unknown:
        print(f"  {len(unknown)} marked clip ids are not in this dataset -- wrong dataset? skipped")
    if not marked:
        raise SystemExit("nothing to score")

    # 1. the clips the aligners get
    def clip_row(key, cid, words):
        return {"id": key, "audio": str((dataset / entries[cid]["audio"]).resolve()),
                "duration": float(entries[cid]["duration"]), "text": " ".join(words),
                "words": words}

    if args.text == "corrected":
        # One alignment per (clip, annotator): corrected word lists differ between people, so
        # a single alignment per clip would time one person's words against another's marks.
        clips = [clip_row(f"{m['id']}#{m.get('annotator', 'anon')}", m["id"],
                          [w["word"] for w in m["words"]])
                 for m in marks if m.get("id") in entries]
    else:
        clips = [clip_row(cid, cid, [w["word"] for w in entries[cid]["labels"][0]["words"]])
                 for cid in marked]
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "clips.jsonl", clips)
    print(f"aligning on the {args.text} transcript: {len(clips)} alignments")

    # 2. every aligner, each in its own interpreter; each skips clips it has already done
    labels = {}
    for name in [a.strip() for a in args.aligners.split(",") if a.strip()]:
        if name not in ALIGNERS:
            raise SystemExit(f"unknown aligner {name!r}; choose from {', '.join(ALIGNERS)}")
        script, var, extra, env_extra = ALIGNERS[name]
        py = os.environ.get(var)
        out = run / "labels" / f"{name}.jsonl"
        if not py:
            print(f"\n[{name}] skipped: set {var} in .env to the interpreter of its environment")
        else:
            cmd = [py, str(HERE / "aligners" / script), "--manifest", str(run / "clips.jsonl"),
                   "--out", str(out), *extra]
            if name == "mwa":
                repo = os.environ.get("EVAL_MWA_REPO")
                if not repo:
                    print(f"\n[{name}] skipped: set EVAL_MWA_REPO to the Multilingual-Word-Aligner clone")
                    cmd = None
                else:
                    cmd += ["--mwa-repo", repo]
            if cmd:
                print(f"\n[{name}]", flush=True)
                env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_extra}
                subprocess.run(cmd, env=env, check=False)
        if out.exists():
            labels[name] = {r["id"]: r["words"] for r in load_jsonl(out)}

    # With corrected text the keys are "<clip>#<annotator>"; split them back apart.
    pair_labels: dict = {}
    if args.text == "corrected":
        for name, by_key in labels.items():
            pairs = {}
            for key, words in by_key.items():
                cid, _, who = key.partition("#")
                pairs[(cid, who)] = words
            pair_labels[name] = pairs
        labels = {name: {} for name in labels}

    # 3. a dataset of just the marked clips, with every aligner attached as a label. The
    #    dataset's own labels stay first, so the tool still opens clips on the same seed.
    scored = run / "dataset"
    (scored / "audio").mkdir(parents=True, exist_ok=True)
    # With corrected text a clip can have one alignment per annotator, but a dataset row
    # holds one label per source. Where people corrected a clip differently, the lanes show
    # the alignment of whoever marked the most clips; eval.json is scored against each
    # annotator's own.
    order = [name for name, _ in sorted(per_annotator_counts(marks), key=lambda kv: -kv[1])]
    divergent = set()

    def label_for(name, cid):
        if args.text != "corrected":
            return labels.get(name, {}).get(cid)
        pairs = pair_labels.get(name, {})
        options = [(who, words) for (c, who), words in pairs.items() if c == cid]
        if not options:
            return None
        if len({json.dumps(w, ensure_ascii=False) for _, w in options}) > 1:
            divergent.add(cid)
        options.sort(key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
        return options[0][1]

    rows = []
    for cid in marked:
        e = json.loads(json.dumps(entries[cid]))
        for name in labels:
            words = label_for(name, cid)
            if words:
                e["labels"] = [lb for lb in e["labels"] if lb["source"] != name]
                e["labels"].append({"source": name, "words": words})
        src = dataset / e["audio"]
        dst = scored / e["audio"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            shutil.copyfile(src, dst)
        rows.append(e)
    write_jsonl(scored / "manifest.jsonl", rows)
    (scored / "metadata.json").write_text(json.dumps({"name": "dataset"}), encoding="utf-8")

    # 4. the marks, split per annotator, for serving the dashboard from files
    shutil.rmtree(run / "marks", ignore_errors=True)
    per: dict[str, list] = {}
    for m in marks:
        if m.get("id") in entries:
            per.setdefault(m.get("annotator", "anon"), []).append(
                {k: v for k, v in m.items() if k != "annotator"})
    for name, rs in per.items():
        write_jsonl(run / "marks" / f"{name}.jsonl", rs)

    if divergent:
        print(f"  {len(divergent)} clips were corrected differently by different people; "
              "their lanes show one person's alignment, the scores use each person's own")

    # 5. score -- the same module the dashboard runs
    print("\n" + "=" * 70)
    sys.path.insert(0, str(ROOT))
    from hebrew_training.aligner_eval import evaluate, report

    result = evaluate(rows, marks, exclude=set(args.exclude), pair_labels=pair_labels or None)
    result["text_source"] = args.text
    report(result)
    (run / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {run / 'eval.json'}")

    print("\nTo see it on the dashboard:")
    print(f"  python -m hebrew_training.align_tag_server --datasets-folder \"{run}\" --dataset dataset "
          f"--out \"{run / 'marks'}\" --port 8091")
    print("  then open http://localhost:8091/?view=eval")


if __name__ == "__main__":
    main()
