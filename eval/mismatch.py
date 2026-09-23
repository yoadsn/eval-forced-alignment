"""Can a forced aligner tell us that a transcript does not match its audio?

    python eval/mismatch.py --marks gold.jsonl --dataset data/datasets/ivrit-ai

A forced aligner is obliged to place every word of the transcript somewhere, whether or not
it was spoken. It cannot refuse. So a word that is not there has to be squeezed in -- a short
span, at low probability, stealing time from a neighbour. This measures how loudly that
shows, and where it does not show at all.

The clips whose word lists people have corrected are transcripts known to match their audio.
From each we build five versions -- clean, plus one insertion, deletion, substitution and
duplication at a recorded position -- align them all, and record for every word the aligner's
own confidence, its duration and the gaps around it.

    <run>/clips.jsonl     every version handed to the aligner
    <run>/labels/         the aligner's output
    <run>/words.jsonl     one row per word: what it is, whether it was tampered with, what
                          the aligner did with it
    <run>/report.json     detection curves, and the breakdown of where it fails

Re-runnable like eval/run_eval.py: alignments already done are skipped. Writes nothing
outside its own run folder -- not the gold set, not the dataset, not the live site.

Needs EVAL_PY_CTC, the same interpreter eval/run_eval.py uses for the CTC aligners.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
KINDS = ("clean", "insert", "delete", "substitute", "duplicate")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def corrupt(words: list[str], kind: str, rng: random.Random, pool: list[str]):
    """One corruption of a known kind at a known place.

    Returns (new words, index touched, the word involved). For `delete` the index is where
    the missing word *was*, so the two words now on either side of the hole can be checked:
    they are the ones that must absorb the speech nobody is claiming any more.
    """
    words = list(words)
    if kind == "clean":
        return words, None, None
    # Never the first or last word: they have only one neighbour to steal from, which is a
    # different situation, and the breakdown reports it separately anyway.
    if len(words) < 4:
        return None, None, None
    i = rng.randrange(1, len(words) - 1)
    if kind == "insert":
        other = rng.choice(pool)
        words.insert(i, other)
        return words, i, other
    if kind == "delete":
        removed = words.pop(i)
        return words, i, removed
    if kind == "substitute":
        original = words[i]
        # From another clip, so it is a real Hebrew word but the wrong one -- an ASR
        # substitution, not a typo.
        other = rng.choice([w for w in pool if w != original])
        words[i] = other
        return words, i, other
    if kind == "duplicate":
        words.insert(i, words[i])
        return words, i, words[i]
    raise ValueError(kind)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--marks", type=Path, required=True, help="Human marks from /api/export.")
    p.add_argument("--dataset", type=Path, required=True, help="Dataset folder (manifest.jsonl, audio/).")
    p.add_argument("--run", type=Path, help="Run folder (default: data/mismatch/<dataset name>).")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--device", default="cuda", help="'cpu' if the CTC environment has no CUDA "
                   "build; the Viterbi runs on CPU either way, so only the forward pass is slower.")
    p.add_argument("--exclude", action="append", default=[], help="Annotator to leave out.")
    args = p.parse_args()

    dataset = args.dataset.resolve()
    run = (args.run or ROOT / "data" / "mismatch" / dataset.name).resolve()
    entries = {e["id"]: e for e in load_jsonl(dataset / "manifest.jsonl")}
    marks = [m for m in load_jsonl(args.marks)
             if m.get("id") in entries and m.get("annotator") not in set(args.exclude)]
    if not marks:
        raise SystemExit("no marks for this dataset")

    # One transcript per clip, preferring the annotator who marked the most -- this
    # experiment needs a correct transcript per clip, not one per person.
    counts: dict[str, int] = {}
    for m in marks:
        counts[m.get("annotator", "anon")] = counts.get(m.get("annotator", "anon"), 0) + 1
    best = {}
    for m in marks:
        who = m.get("annotator", "anon")
        if m["id"] not in best or counts[who] > counts[best[m["id"]].get("annotator", "anon")]:
            best[m["id"]] = m
    truth = {cid: [w["word"] for w in m["words"]] for cid, m in best.items()}
    pool = sorted({w for ws in truth.values() for w in ws if len(w) > 2})
    print(f"{len(truth)} clips with a corrected transcript, {len(pool)} words in the pool")

    rng = random.Random(args.seed)
    clips, index = [], []
    for cid in sorted(truth):
        for kind in KINDS:
            words, at, involved = corrupt(truth[cid], kind, rng, pool)
            if words is None:
                continue
            key = f"{cid}#{kind}"
            clips.append({"id": key,
                          "audio": str((dataset / entries[cid]["audio"]).resolve()),
                          "duration": float(entries[cid]["duration"]),
                          "text": " ".join(words), "words": words})
            index.append({"key": key, "clip": cid, "kind": kind, "at": at,
                          "involved": involved, "n_words": len(words),
                          "source": entries[cid]["metadata"].get("source")})
    run.mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "clips.jsonl", clips)
    write_jsonl(run / "index.jsonl", index)
    print(f"{len(clips)} versions to align ({len(KINDS)} per clip)")

    py = os.environ.get("EVAL_PY_CTC")
    if not py:
        raise SystemExit("set EVAL_PY_CTC to the interpreter of the CTC environment (see eval/README.md)")
    out = run / "labels" / "mms.jsonl"
    cmd = [py, str(HERE / "aligners" / "ctc.py"), "--manifest", str(run / "clips.jsonl"),
           "--out", str(out), "--model", MODEL, "--device", args.device]
    subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False)
    if not out.exists():
        raise SystemExit("the aligner produced nothing")

    # One row per word, carrying everything needed to analyse without aligning again.
    aligned = {r["id"]: r["words"] for r in load_jsonl(out)}
    by_key = {r["key"]: r for r in index}
    rows = []
    for key, words in aligned.items():
        meta = by_key[key]
        for i, w in enumerate(words):
            prev_end = words[i - 1]["end"] if i else None
            next_start = words[i + 1]["start"] if i + 1 < len(words) else None
            # For a deletion nothing was inserted; the words that must absorb the orphaned
            # speech are the two now touching across the hole.
            if meta["kind"] == "delete":
                tampered = meta["at"] is not None and i in (meta["at"] - 1, meta["at"])
            else:
                tampered = meta["at"] is not None and i == meta["at"]
            rows.append({
                "key": key, "clip": meta["clip"], "kind": meta["kind"],
                "source": meta["source"], "i": i, "word": w["word"],
                "chars": len(w["word"]), "tampered": bool(tampered),
                "edge": i == 0 or i == len(words) - 1,
                "score": w.get("score"), "dur": round(w["end"] - w["start"], 4),
                "gap_before": None if prev_end is None else round(w["start"] - prev_end, 4),
                "gap_after": None if next_start is None else round(next_start - w["end"], 4),
            })
    write_jsonl(run / "words.jsonl", rows)
    print(f"{len(rows)} word rows -> {run / 'words.jsonl'}")
    print(f"\nNow analyse: python eval/mismatch_report.py --run {run}")


if __name__ == "__main__":
    main()
