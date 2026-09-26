"""aligners.toml, resolved against this machine: which interpreter, which device, and why not."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# What is worth recording about an aligner's environment. Everything else in it is noise
# next to these, and the full freeze of a torch environment is a few hundred lines.
PACKAGES = ("torch", "torchaudio", "transformers", "uroman", "stable-ts", "faster-whisper",
            "ctranslate2", "mwa", "librosa", "phonikud", "phonikud-onnx", "numpy")

_PROBE = r"""
import json, importlib.metadata as m, platform
out = {"python": platform.python_version(), "platform": platform.platform(), "devices": ["cpu"],
       "packages": {d.metadata["Name"].lower(): d.version for d in m.distributions()}}
try:
    import torch
    if torch.cuda.is_available():
        out["devices"].insert(0, "cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out["devices"].insert(0 if "cuda" not in out["devices"] else 1, "mps")
except Exception:
    pass
print(json.dumps(out))
"""


@dataclass
class Aligner:
    name: str
    runner: Path
    env: str
    args: list[str]
    devices: list[str]
    env_vars: dict[str, str] = field(default_factory=dict)
    needs: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Derived:
    name: str
    source: str
    correction: Path
    description: str = ""


def load(path: Path = ROOT / "aligners.toml") -> tuple[dict[str, Aligner], dict[str, Derived]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    aligners = {
        name: Aligner(name=name, runner=ROOT / "aligners" / a["runner"], env=a["env"],
                      args=list(a.get("args", [])), devices=list(a.get("devices", ["cpu"])),
                      env_vars=dict(a.get("env_vars", {})), needs=list(a.get("needs", [])),
                      description=a.get("description", ""))
        for name, a in raw.get("aligners", {}).items()
    }
    derived = {
        name: Derived(name=name, source=d["from"], correction=ROOT / d["correction"],
                      description=d.get("description", ""))
        for name, d in raw.get("derived", {}).items()
    }
    return aligners, derived


def interpreter(env: str) -> Path | None:
    """$EFA_PY_<ENV> if set, otherwise the env's own uv project under envs/."""
    override = os.environ.get(f"EFA_PY_{env.upper()}")
    if override:
        return Path(override) if Path(override).exists() else None
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        p = ROOT / "envs" / env / rel
        if p.exists():
            return p
    return None


_PROBES: dict[str, dict] = {}


def probe(python: Path) -> dict:
    """The interpreter's version, its packages, and the torch devices it can see."""
    key = str(python)
    if key not in _PROBES:
        try:
            out = subprocess.run([str(python), "-c", _PROBE], capture_output=True, text=True,
                                 timeout=180, check=True).stdout
            info = json.loads(out.strip().splitlines()[-1])
        except Exception as exc:  # noqa: BLE001 -- an env that cannot even start is reported
            info = {"error": f"{type(exc).__name__}: {exc}"[:300], "devices": []}
        pk = info.get("packages", {})
        info["packages"] = {k: pk[k] for k in PACKAGES if k in pk}
        _PROBES[key] = info
    return _PROBES[key]


def resolve(a: Aligner, device: str | None = None) -> dict:
    """How to run this aligner here, or why it cannot be.

    {"ok": True, "python", "device", "args", "env", "probe"} or {"ok": False, "reason"}
    """
    py = interpreter(a.env)
    if py is None:
        return {"ok": False, "reason": f"no interpreter for env {a.env!r}: build envs/{a.env} "
                "(see README.md) or set EFA_PY_" + a.env.upper()}
    missing = [n for n in a.needs if not os.environ.get(n)]
    if missing:
        return {"ok": False, "reason": "not configured: set " + ", ".join(missing)}
    info = probe(py)
    if info.get("error"):
        return {"ok": False, "reason": f"env {a.env!r} does not start: {info['error']}"}
    have = info.get("devices", ["cpu"])
    if device:
        if device not in a.devices:
            return {"ok": False, "reason": f"{a.name} does not run on {device} "
                    f"(supports {', '.join(a.devices)})"}
        if device not in have:
            return {"ok": False, "reason": f"{device} is not available in env {a.env!r}"}
        chosen = device
    else:
        chosen = next((d for d in a.devices if d in have), None)
        if chosen is None:
            return {"ok": False, "reason": f"needs one of {', '.join(a.devices)}; "
                    f"this machine has {', '.join(have)}"}
    args = [s.format(**{n: os.environ[n] for n in a.needs}) for s in a.args]
    return {"ok": True, "python": py, "device": chosen, "args": args,
            "env": {**os.environ, "PYTHONIOENCODING": "utf-8", **a.env_vars}, "probe": info}
