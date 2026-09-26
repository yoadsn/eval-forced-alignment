"""Labels made from other labels: `mms-corrected` from `mms`, with correction/'s own code.

Loaded from correction/ rather than copied, so the rule the viewer shows is the rule that
was measured. Needs numpy and soundfile, which the orchestrator has; no model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "correction" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def correct(rows: list[dict], audio: dict[str, Path], model: dict) -> list[dict]:
    """rows: {id, words} from the source aligner. Returns the same rows, corrected."""
    bf, cm = _load("boundary_fix"), _load("correct_mms")
    out, clips = [], []
    envelopes: dict[Path, object] = {}
    for row in rows:
        ws = sorted(row["words"], key=lambda w: w["start"])
        path = audio[row["id"]]
        if path not in envelopes:
            envelopes[path] = cm.envelope(path)
        env = envelopes[path]
        # The whole clip at once: a boundary is shared by two words, and correcting each
        # against the other's original position lets them cross.
        made = [{"word": w["word"], "start": w["start"], "end": w["end"],
                 "clip": row["id"], "who": None,
                 "prev_end": ws[i - 1]["end"] if i else 0.0,
                 "next_start": ws[i + 1]["start"] if i + 1 < len(ws) else None,
                 "gap_after": (ws[i + 1]["start"] if i + 1 < len(ws) else w["end"]) - w["end"],
                 "env": env}
                for i, w in enumerate(ws)]
        placed = cm.corrected(bf, made, model)
        words = []
        for w, r in zip(ws, made):
            start, end = placed[id(r)]
            words.append({**{k: v for k, v in w.items() if k not in ("start", "end")},
                          "start": round(start, 4), "end": round(end, 4)})
        clips.append(words)
        out.append({"id": row["id"], "words": words})
    cm.no_overlap(clips)
    return out
