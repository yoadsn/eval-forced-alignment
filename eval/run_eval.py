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

    # 1. the clips the aligners get: the dataset's own word list, which is what people edited
    clips = [{"id": cid, "audio": str((dataset / entries[cid]["audio"]).resolve()),
              "duration": float(entries[cid]["duration"]), "text": entries[cid]["text"],
              "words": [w["word"] for w in entries[cid]["labels"][0]["words"]]}
             for cid in marked]
    write_jsonl(run / "clips.jsonl", clips)

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

    # 3. a dataset of just the marked clips, with every aligner attached as a label. The
    #    dataset's own labels stay first, so the tool still opens clips on the same seed.
    scored = run / "dataset"
    (scored / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in marked:
        e = json.loads(json.dumps(entries[cid]))
        for name, by_id in labels.items():
            if cid in by_id:
                e["labels"] = [lb for lb in e["labels"] if lb["source"] != name]
                e["labels"].append({"source": name, "words": by_id[cid]})
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

    # 5. score -- the same module the dashboard runs
    print("\n" + "=" * 70)
    cmd = [sys.executable, "-m", "hebrew_training.aligner_eval", "--dataset", str(scored),
           "--gold", str(args.marks), "--out", str(run / "eval.json")]
    for name in args.exclude:
        cmd += ["--exclude", name]
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=True)

    print("\nTo see it on the dashboard:")
    print(f"  python -m hebrew_training.align_tag_server --datasets-folder \"{run}\" --dataset dataset "
          f"--out \"{run / 'marks'}\" --port 8091")
    print("  then open http://localhost:8091/?view=eval")


if __name__ == "__main__":
    main()
