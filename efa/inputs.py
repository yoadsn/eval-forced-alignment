"""The gold set, read from what the tagging tool publishes.

`publish_labeled_dataset.py` in the tagging repo writes an AudioFolder: `metadata.jsonl`
beside an `audio/` folder, one row per tagged clip. The same layout is used whether it went
to the Hub or stayed in a local folder, so both are read the same way once the Hub copy is
on disk.

    {"audio_file_name": "audio/<id>.wav", "id": "...", "metadata": {...},
     "text": "...", "words": [{"word", "start", "end"}, ...], "annotator": "yoad"}

`words` are the human-corrected words and `text` is them joined, so the words the aligners
are asked to time are exactly the ones the human timed. `annotator` may be missing or empty
in older exports; such rows are kept and named `unknown`. One clip may appear more than once
with different annotators -- that is what the human-agreement floor is computed from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

UNKNOWN = "unknown"


@dataclass
class Row:
    id: str
    annotator: str
    audio: Path
    duration: float
    text: str
    words: list[dict]
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """What the aligners are keyed on: one alignment per (clip, annotator), because two
        people do not always correct a clip's words the same way."""
        return f"{self.id}#{self.annotator}"


@dataclass
class GoldSet:
    rows: list[Row]
    source: dict  # {"kind", "ref", "revision"} -- enough to fetch exactly this input again
    warnings: list[str] = field(default_factory=list)


def _is_hub_ref(ref: str) -> bool:
    return not Path(ref).exists() and ref.count("/") == 1 and not ref.startswith((".", "/"))


def fetch(ref: str, revision: str | None = None) -> tuple[Path, dict]:
    """A local folder holding metadata.jsonl + audio/, and where it came from."""
    if not _is_hub_ref(ref):
        folder = Path(ref).expanduser().resolve()
        meta = folder / "metadata.jsonl"
        if not meta.exists():
            raise SystemExit(f"{folder} has no metadata.jsonl -- not a published gold set")
        digest = hashlib.sha256(meta.read_bytes()).hexdigest()[:16]
        return folder, {"kind": "local", "ref": str(folder), "revision": f"sha256:{digest}"}

    from huggingface_hub import HfApi, snapshot_download

    # Resolved to a commit first, so the result names the exact snapshot it scored even when
    # the caller asked for a branch -- the gold set grows, and "main" will not mean this later.
    sha = HfApi().dataset_info(ref, revision=revision).sha
    folder = Path(snapshot_download(ref, repo_type="dataset", revision=sha))
    return folder, {"kind": "hf", "ref": ref, "revision": sha}


def load(ref: str, revision: str | None = None) -> GoldSet:
    import soundfile

    folder, source = fetch(ref, revision)
    rows: list[Row] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate((folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        words = [w for w in r.get("words") or [] if w.get("start") is not None and w.get("end") is not None]
        if not words:
            warnings.append(f"line {n}: clip {r.get('id')} has no timed words, skipped")
            continue
        audio = folder / r["audio_file_name"]
        if not audio.exists():
            warnings.append(f"line {n}: {audio} missing, skipped")
            continue
        who = (r.get("annotator") or "").strip() or UNKNOWN
        # Two rows for the same clip under the same name cannot be told apart in the
        # scoring, which keys on (clip, annotator); keep both, as distinct people.
        base, k = who, 2
        while f"{r['id']}#{who}" in seen:
            who = f"{base}~{k}"
            k += 1
        if who != base:
            warnings.append(f"clip {r['id']}: a second row under {base!r}, kept as {who!r}")
        seen.add(f"{r['id']}#{who}")
        rows.append(Row(
            id=r["id"], annotator=who, audio=audio,
            duration=round(soundfile.info(str(audio)).duration, 3),
            text=r.get("text") or " ".join(w["word"] for w in words),
            words=[{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in words],
            metadata=r.get("metadata") or {},
        ))
    if not rows:
        raise SystemExit(f"no usable rows in {source['ref']}")
    source["rows"] = len(rows)
    source["clips"] = len({r.id for r in rows})
    return GoldSet(rows=rows, source=source, warnings=warnings)
