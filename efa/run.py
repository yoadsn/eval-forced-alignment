"""Score every aligner against a published gold set, end to end, re-runnably.

    python -m efa.run --input yoad/eval-forced-alignment          # a Hub dataset
    python -m efa.run --input ../audio_alignment_tool/data/publish # the same, as a folder

The input is what the tagging tool's publish_labeled_dataset.py writes: one row per tagged
clip, the human-corrected words and their timings. Every aligner is given exactly those
words, so each aligner word pairs with a human one and the score is timing, not transcript.

Writes a run folder:

    <run>/clips.jsonl          what the aligners are handed, one row per (clip, annotator)
    <run>/labels/<name>.jsonl  each aligner's output; <name>.failed.jsonl, <name>.meta.json
    <run>/marks/<who>.jsonl    the human marks per annotator
    <run>/dataset/manifest.jsonl  the clips with every label attached (correction/ reads it)
    <run>/result.json          the result: self-contained, everything but the audio

result.json is what the tagging tool's eval viewer uploads and shows.

Re-running is cheap: each aligner skips what it has done, so after more tagging only the
new clips are aligned. A clip whose corrected words changed is aligned again.

Aligners that cannot run here -- no environment, no GPU, not configured -- are listed in the
result as unavailable, with the reason. To score one anyway, run its runner on a machine
that can and copy its labels/<name>.jsonl and <name>.meta.json into this run folder; the
next run picks them up as imported.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

from efa import inputs, registry, scoring

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "efa-result/1"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def git_sha() -> str | None:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


def default_run(source: dict) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(source["ref"]).name if source["kind"] == "local"
                  else source["ref"]).strip("-")
    rev = source["revision"].split(":")[-1][:8]
    return ROOT / "runs" / f"{slug}-{rev}"


def prune_stale(run: Path, clips: list[dict]) -> int:
    """Drop alignments of a (clip, annotator) whose words have changed since they were made.

    Keys stay the same when someone re-corrects a clip, so without this an old alignment of
    the old words would be scored against the new marks.
    """
    before = {r["id"]: r["words"] for r in load_jsonl(run / "clips.jsonl")}
    changed = {c["id"] for c in clips if c["id"] in before and before[c["id"]] != c["words"]}
    if changed:
        for f in (run / "labels").glob("*.jsonl"):
            rows = load_jsonl(f)
            kept = [r for r in rows if r.get("id") not in changed]
            if len(kept) != len(rows):
                write_jsonl(f, kept)
    return len(changed)


def run_aligner(a: registry.Aligner, how: dict, run: Path) -> int:
    out = run / "labels" / f"{a.name}.jsonl"
    cmd = [str(how["python"]), str(a.runner), "--manifest", str(run / "clips.jsonl"),
           "--out", str(out), "--device", how["device"], *how["args"]]
    print(f"\n[{a.name}] {how['device']} -- {' '.join(cmd[1:])}", flush=True)
    code = subprocess.run(cmd, env=how["env"], check=False).returncode
    meta = {
        "aligner": a.name, "runner": a.runner.name, "args": how["args"], "device": how["device"],
        "env": a.env, "python": how["probe"].get("python"), "packages": how["probe"].get("packages", {}),
        "platform": how["probe"].get("platform"), "host": socket.gethostname(),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "exit_code": code,
    }
    (run / "labels" / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return code


def apply_derived(d: registry.Derived, run: Path, keys: dict[str, dict]) -> dict:
    """A label computed from another one, not from audio by a model: mms-corrected."""
    src = run / "labels" / f"{d.source}.jsonl"
    if not src.exists():
        return {"status": "unavailable", "reason": f"needs {d.source!r}, which has no output here"}
    # A correction fitted on this run's own clips wins over the repo's committed one.
    path = next((c for c in (run / "correction.json", d.correction) if c.exists()), None)
    if path is None:
        return {"status": "unavailable",
                "reason": f"no correction fitted: python correction/correct_mms.py --run {run} "
                f"--dataset {run / 'dataset'}"}
    from efa import derive

    model = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in load_jsonl(src) if r["id"] in keys and r.get("words")]
    out = derive.correct(rows, {k: Path(c["audio"]) for k, c in keys.items()}, model)
    write_jsonl(run / "labels" / f"{d.name}.jsonl", out)
    return {"status": "ok", "from": d.source, "correction": {
        "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "in_sample": path.parent == run,
        "fitted_on_words": model.get("fitted_on_words"),
        "fitted_on": model.get("fitted_on"),
        # The numbers to quote: the correction scored on clips it was not fitted on. The
        # table's own row for it is in-sample whenever these clips are the ones it was fitted
        # on, which is flattering.
        "held_out": model.get("held_out"),
    }}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True,
                   help="A Hub dataset id (owner/name) or a local folder with metadata.jsonl + audio/.")
    p.add_argument("--revision", help="Hub revision (branch, tag or commit). Default: main.")
    p.add_argument("--run", type=Path, help="Run folder. Default: runs/<input>-<revision>.")
    p.add_argument("--aligners", help="Comma-separated subset to *align*. Everything with output "
                   "in the run folder is still scored. Default: all in aligners.toml.")
    p.add_argument("--device", help="Force one device (cpu, cuda, mps) instead of each aligner's best.")
    p.add_argument("--no-align", action="store_true", help="Only rescore what is already in the run.")
    p.add_argument("--exclude", action="append", default=["probe"],
                   help="Annotator to leave out of scoring, e.g. a test account. Repeatable. "
                   "Default: probe.")
    p.add_argument("--title", help="A name for the result, shown in the viewer.")
    p.add_argument("--draws", type=int, default=scoring.BOOTSTRAP_DRAWS)
    args = p.parse_args()

    gold = inputs.load(args.input, args.revision)
    for w in gold.warnings:
        print(f"  note: {w}")
    run = (args.run or default_run(gold.source)).resolve()
    run.mkdir(parents=True, exist_ok=True)
    people = sorted({r.annotator for r in gold.rows})
    print(f"{gold.source['rows']} marks on {gold.source['clips']} clips from {gold.source['ref']} "
          f"@ {gold.source['revision'][:12]}, annotators: {', '.join(people)}")
    print(f"run folder: {run}")

    # 1. what every aligner is given
    clips = [{"id": r.key, "audio": str(r.audio), "duration": r.duration, "text": r.text,
              "words": [w["word"] for w in r.words]} for r in gold.rows]
    stale = prune_stale(run, clips)
    if stale:
        print(f"  {stale} clips were re-corrected since the last run; aligning them again")
    write_jsonl(run / "clips.jsonl", clips)
    keys = {c["id"]: c for c in clips}

    import shutil

    shutil.rmtree(run / "marks", ignore_errors=True)
    per: dict[str, list] = {}
    for r in gold.rows:
        per.setdefault(r.annotator, []).append({"id": r.id, "words": r.words})
    for who, rs in per.items():
        write_jsonl(run / "marks" / f"{who}.jsonl", rs)

    # 2. align
    aligners, derived = registry.load()
    wanted = [a.strip() for a in args.aligners.split(",")] if args.aligners else list(aligners)
    for name in wanted:
        if name not in aligners:
            raise SystemExit(f"unknown aligner {name!r}; choose from {', '.join(aligners)}")
    status: dict[str, dict] = {}
    for name, a in aligners.items():
        how = registry.resolve(a, args.device)
        if not how["ok"]:
            status[name] = {"status": "unavailable", "reason": how["reason"]}
            if name in wanted and not args.no_align:
                print(f"\n[{name}] skipped: {how['reason']}")
            continue
        if name in wanted and not args.no_align:
            code = run_aligner(a, how, run)
            if code:
                status[name] = {"status": "error", "reason": f"runner exited with {code}"}

    for name, d in derived.items():
        status[name] = apply_derived(d, run, keys)
        if status[name]["status"] != "ok":
            print(f"\n[{name}] skipped: {status[name]['reason']}")

    # 3. collect what exists, whether made now, earlier, or on another machine
    pair_labels: dict[str, dict] = {}
    for name in [*aligners, *derived]:
        rows = [r for r in load_jsonl(run / "labels" / f"{name}.jsonl") if r["id"] in keys]
        failed = [r for r in load_jsonl(run / "labels" / f"{name}.failed.jsonl") if r["id"] in keys]
        meta_path = run / "labels" / f"{name}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        st = status.setdefault(name, {})
        if rows:
            if st.get("status") == "unavailable" and name in aligners:
                # Not runnable here, but output is present: made elsewhere, or here earlier.
                st.update(status="imported", reason_not_run_here=st.pop("reason"))
            elif st.get("status") != "error":
                st["status"] = "ok"
            pair_labels[name] = {}
            for r in rows:
                cid, _, who = r["id"].partition("#")
                pair_labels[name][(cid, who)] = r["words"]
        elif st.get("status") in (None, "ok"):
            st.update(status="failed" if failed else "no-output",
                      reason="every clip failed" if failed else "ran, but wrote nothing")
        st["description"] = (aligners.get(name) or derived.get(name)).description
        st["aligned"] = len(rows)
        st["missing"] = len(keys) - len(rows) - len(failed)
        st["failed"] = [{"id": r["id"], "why": r.get("why", "")} for r in failed]
        if meta:
            st["provenance"] = {k: meta.get(k) for k in
                                ("device", "env", "python", "packages", "platform", "host",
                                 "updated_at", "args", "runner")}

    # 4. the dataset view correction/ reads: every clip once, every label attached
    by_clip: dict[str, dict] = {}
    for r in gold.rows:
        e = by_clip.setdefault(r.id, {"id": r.id, "audio": str(r.audio), "duration": r.duration,
                                      "text": r.text, "metadata": r.metadata, "labels": []})
        for name, pairs in pair_labels.items():
            if (r.id, r.annotator) in pairs and not any(lb["source"] == name for lb in e["labels"]):
                e["labels"].append({"source": name, "words": pairs[(r.id, r.annotator)]})
    write_jsonl(run / "dataset" / "manifest.jsonl", by_clip.values())

    # 5. score. Entries carry no labels of their own: every alignment is per (clip, annotator),
    #    and there is no seed -- the aligners never showed their output to the annotators.
    entries = [{"id": cid, "text": e["text"], "metadata": e["metadata"], "labels": []}
               for cid, e in by_clip.items()]
    marks = [{"id": r.id, "annotator": r.annotator, "words": r.words} for r in gold.rows]
    print("\n" + "=" * 70)
    result = scoring.evaluate(entries, marks, exclude=set(args.exclude),
                              pair_labels=pair_labels, draws=args.draws)
    scoring.report(result)

    # 6. the self-contained result
    result = {
        "schema": SCHEMA,
        "title": args.title or f"{gold.source['ref']} @ {gold.source['revision'][:12]}",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "producer": {"repo": "eval-forced-alignment", "git": git_sha(), "host": socket.gethostname(),
                     "python": platform.python_version()},
        "input": gold.source,
        "text_source": "corrected",
        "excluded": sorted(set(args.exclude)),
        "aligner_status": status,
        **result,
        "clip_data": {
            cid: {
                "text": e["text"], "duration": e["duration"], "metadata": e["metadata"],
                "human": {r.annotator: r.words for r in gold.rows if r.id == cid},
                "labels": {name: {who: words for (c, who), words in pairs.items() if c == cid}
                           for name, pairs in pair_labels.items()
                           if any(c == cid for c, _ in pairs)},
            }
            for cid, e in by_clip.items()
        },
    }
    out = run / "result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}  ({out.stat().st_size // 1024} KB)")
    print("Upload it on the tagging tool's aligner eval screen to view it there.")
    unavailable = {n: s["reason"] for n, s in status.items() if s["status"] == "unavailable"}
    if unavailable:
        print("\nNot in this result:")
        for n, why in unavailable.items():
            print(f"  {n:<18} {why}")


if __name__ == "__main__":
    sys.exit(main())
