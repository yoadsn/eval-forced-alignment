"""A correction model that sits outside the aligner and never touches it.

    python eval/boundary_model.py --run data/eval_runs/ivrit-ai-corrected

MMS's boundaries are taken as given. Every feature here is read either from its output or
from the waveform -- nothing is fed back into the model, and its input is left alone.

The target is the *signed* correction for each edge, because the error is a bias and a bias
has a direction; an absolute error would throw that away. Trees are fitted with an absolute
error criterion, since absolute error is also how the result is scored, and predicting the
conditional median is what minimises it.

Folds are by clip, never by word: boundaries inside one clip share a speaker, a recording
and an annotator.

Audio features are the point of the exercise. The CTC posterior says nothing after a word
ends -- measured, 0.02 -- but the waveform plainly does, which is visible in
`eval/word_end_strips.py`. So the envelope is measured directly: how loud it still is where
MMS stopped, and how long until it falls away.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOP = 0.005          # 5 ms envelope
QUIET = 0.10         # "the sound has gone" as a share of the word's own peak
LOOK = 0.50          # how far past the boundary to look


def load_bf():
    spec = importlib.util.spec_from_file_location("bf", ROOT / "eval" / "boundary_fix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audio_features(rows, dataset: Path):
    """What the waveform says around each boundary, which the aligner's own output cannot."""
    import numpy as np
    import soundfile as sf

    entries = {}
    for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            entries[e["id"]] = e
    cache: dict[str, tuple] = {}

    def envelope(cid):
        if cid not in cache:
            wav, rate = sf.read(dataset / entries[cid]["audio"], dtype="float32", always_2d=True)
            wav = np.abs(wav.mean(axis=1))
            win = max(1, int(HOP * rate))
            env = wav[: len(wav) // win * win].reshape(-1, win).max(axis=1)
            cache[cid] = (env, HOP)
        return cache[cid]

    for r in rows:
        env, hop = envelope(r["clip"])
        a, b = int(r["start"] / hop), int(r["end"] / hop)
        inside = env[a:b]
        peak = float(inside.max()) if len(inside) else 0.0
        r["peak"] = peak
        if peak <= 0:
            r.update(loud_at_end=0.0, decay_ms=0.0, loud_before_start=0.0, rise_ms=0.0)
            continue
        # How much sound is still there where MMS stopped: near 1 means it cut mid-word.
        r["loud_at_end"] = float(env[min(b, len(env) - 1)]) / peak
        limit = int(min(r["end"] + LOOK, r["end"] + max(r["gap_after"], 0.0)) / hop)
        i = b
        while i < min(limit, len(env)) and env[i] > QUIET * peak:
            i += 1
        r["decay_ms"] = (i - b) * hop * 1000
        r["loud_before_start"] = float(env[max(a - 1, 0)]) / peak
        limit0 = int(max(r["start"] - LOOK, r["start"] - max(r["gap_before"], 0.0)) / hop)
        j = a
        while j > max(limit0, 0) and env[j - 1] > QUIET * peak:
            j -= 1
        r["rise_ms"] = (a - j) * hop * 1000
    return rows


def features(bf, r, edge):
    """One row of numbers per boundary. Categoricals are one-hot so a tree can split cleanly
    on a class without inventing an order between manners of articulation."""
    letter = bf.letter_class(bf.edge_letter(r["word"], edge))
    phone = bf.phone_class(r.get("ipa", ""), edge)
    gap = r["gap_before"] if edge == "start" else r["gap_after"]
    out = [gap, min(gap, 0.3), len(r["word"]), r["score"] or 0.0,
           float(r["first_word"] if edge == "start" else r["last_word"]),
           float(bf.final_stressed(r.get("ipa", ""))),
           r["loud_at_end"] if edge == "end" else r["loud_before_start"],
           r["decay_ms"] if edge == "end" else r["rise_ms"]]
    for name in ("plosive", "fricative", "nasal", "liquid", "glottal", "other"):
        out.append(float(letter == name))
    for name in ("vowel", "plosive", "fricative", "nasal", "liquid", "glide", "other"):
        out.append(float(phone == name))
    return out


NAMES = ["gap", "gap capped", "word length", "ctc score", "at clip edge", "final stress",
         "loudness at the boundary", "ms until quiet"] + \
        [f"letter {n}" for n in ("plosive", "fricative", "nasal", "liquid", "glottal", "other")] + \
        [f"phone {n}" for n in ("vowel", "plosive", "fricative", "nasal", "liquid", "glide", "other")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ivrit-ai"))
    p.add_argument("--phonemes", type=Path, default=Path("data/unclaimed/phonemes.json"))
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--leaf", type=int, default=40)
    args = p.parse_args()

    import numpy as np
    from sklearn.tree import DecisionTreeRegressor

    from hebrew_training.aligner_eval import summarise

    bf = load_bf()
    rows = audio_features(bf.build(args.run, "mms", {"probe"}, args.phonemes), args.dataset)
    clips = sorted({r["clip"] for r in rows})
    half = set(clips[::2])

    held, importances = [], np.zeros(len(NAMES))
    for train_in_half in (True, False):
        train = [r for r in rows if (r["clip"] in half) == train_in_half]
        test = [r for r in rows if (r["clip"] in half) != train_in_half]
        models = {}
        for edge in ("start", "end"):
            X = np.array([features(bf, r, edge) for r in train])
            y = np.array([(r["start"] - r["h_start"]) if edge == "start" else (r["h_end"] - r["end"])
                          for r in train])
            tree = DecisionTreeRegressor(criterion="absolute_error", max_depth=args.depth,
                                         min_samples_leaf=args.leaf, random_state=0)
            tree.fit(X, y)
            models[edge] = tree
            importances += tree.feature_importances_
        for r in test:
            a = float(models["start"].predict([features(bf, r, "start")])[0])
            b = float(models["end"].predict([features(bf, r, "end")])[0])
            s, e = bf.apply(r, "shift", a, b)
            held += [abs(s - r["h_start"]) * 1000, abs(e - r["h_end"]) * 1000]

    print(f"{'':<22}{'median':>9}{'p90':>9}{'<=10ms':>9}{'<=25ms':>9}{'<=50ms':>9}")
    named = [("mms, raw", bf.errors(rows)),
             ("one shift", bf.two_fold_feature(rows, "none")),
             ("by letter class", bf.two_fold_feature(rows, "letter class")),
             ("by pause", bf.two_fold_feature(rows, "pause")),
             ("tree, all features", held)]
    for name, errs in named:
        s = summarise(errs)
        print(f"  {name:<20}{s['median_ms']:>9}{s['p90_ms']:>9}{s['within_10ms']:>9}"
              f"{s['within_25ms']:>9}{s['within_50ms']:>9}")

    print("\nwhat the trees actually used:")
    order = np.argsort(-importances)
    for i in order[:8]:
        if importances[i] > 0.005:
            print(f"  {NAMES[i]:<26}{importances[i] / 4:.2f}")


if __name__ == "__main__":
    main()
