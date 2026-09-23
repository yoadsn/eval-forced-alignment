"""CTC forced alignment: the `hebrew` and `mms` aligners.

    python ctc.py --manifest clips.jsonl --out hebrew.jsonl --model imvladikon/wav2vec2-xls-r-300m-hebrew
    python ctc.py --manifest clips.jsonl --out mms.jsonl    --model MahmoudAshraf/mms-300m-1130-forced-aligner

Same algorithm for both; they differ only in model. A wav2vec2 Hebrew head works on Hebrew
letters directly; MMS's alphabet is romanized Latin, so its text goes through uroman first
(`--romanize auto` decides from the model's vocabulary). A word whose letters the model has
no symbol for -- digits, mostly -- is left unplaced rather than failing the clip.

Input rows: {id, audio (absolute path), duration, text, words}. `words` is the dataset's own
word list -- the one annotators were shown and edited -- and is what gets timed, so every
aligner's words pair one-to-one with the humans'. Splitting `text` on spaces is only the
fallback: it disagrees exactly where it matters, e.g. the dataset keeps "- - - במגזר" as one
word where a space split makes four. Output rows: {id, words}. Clips already in --out are
skipped, so a re-run after the dataset grows aligns only the new ones.

Needs: torch, torchaudio, transformers, uroman, soundfile, scipy (see eval/README.md).
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from math import gcd
from pathlib import Path

SAMPLE_RATE = 16000
_WS = re.compile(r"\s+")
_HEBREW = re.compile(r"[֐-׿]")


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def load_audio(path: str, duration: float):
    """Mono float32 at 16 kHz. Decoded, averaged and resampled with scipy's resample_poly
    exactly as the training loader does, so these timings reproduce the ones this aligner
    was first scored with."""
    import numpy as np
    import soundfile
    from scipy.signal import resample_poly

    wav, rate = soundfile.read(path, dtype="float32", always_2d=True)
    wav = wav[: int(duration * rate)].mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(int(rate), SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
    return wav


def blank_id(tokenizer) -> int:
    """The CTC blank. Models name it differently -- MMS <blank>, a wav2vec2 head [PAD] -- and
    the wrong one silently ruins every alignment."""
    vocab = tokenizer.get_vocab()
    for name in ("<blank>", "[PAD]", "<pad>"):
        if name in vocab:
            return vocab[name]
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


def gap_rows(timed, blank_prob, ratio, duration, emission_ids, id_to_char, min_gap=0.08):
    """Every stretch of audio no word claims, and how much speech is in it.

    The point of the measure. A gap's *length* says nothing: a pause and a word the
    transcript is missing are the same length. But the model reports, per frame, the
    probability that nothing is being said -- the CTC blank. Summing 1 - P(blank) across a
    gap gives the seconds of speech lying in audio that no word accounts for. Silence comes
    out near zero however long it is; a spoken word that the transcript lacks comes out like
    a word.

    `letters` is what the model would emit there if asked, which is how a hesitation gives
    itself away: "eh" decodes to a vowel or two, a real word to a word's worth of letters.
    """
    edges = [(0.0, timed[0]["start"])] if timed else []
    edges += [(timed[i]["end"], timed[i + 1]["start"]) for i in range(len(timed) - 1)]
    edges += [(timed[-1]["end"], duration)] if timed else []
    out = []
    for start, end in edges:
        if end - start < min_gap:
            continue
        a, b = int(round(start / ratio)), int(round(end / ratio))
        b = min(b, len(blank_prob))
        if b <= a:
            continue
        speech = sum(1.0 - p for p in blank_prob[a:b]) * ratio
        letters = "".join(
            id_to_char.get(t, "") for t in emission_ids[a:b] if t in id_to_char
        )
        # Collapse CTC's repeats, the way a greedy decode would.
        squeezed = "".join(c for i, c in enumerate(letters) if i == 0 or c != letters[i - 1])
        out.append({"start": round(start, 4), "end": round(end, 4),
                    "dur": round(end - start, 4), "speech": round(speech, 4),
                    "speech_frac": round(speech / (end - start), 4),
                    "letters": squeezed[:40]})
    return out


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
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--romanize", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--gaps", type=Path, help="Also write, per clip, every stretch of audio no "
                   "word claims, with how much speech is in it -- see gap_rows().")
    p.add_argument("--frames", type=Path, help="Also write, per clip, the model's per-frame "
                   "probability that nothing is being said. Lets a word end be extended to "
                   "where the sound actually stops rather than by a fixed amount.")
    args = p.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip = done_ids(args.out)
    rows = [r for r in rows if r["id"] not in skip]
    print(f"{args.model}: {len(rows)} clips to align ({len(skip)} already done)", flush=True)
    if not rows:
        return

    import torch
    import torchaudio.functional as AF
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(args.device).eval()
    vocab = {k: v for k, v in processor.tokenizer.get_vocab().items() if len(k) == 1}
    blank = blank_id(processor.tokenizer)
    has_hebrew = any(_HEBREW.match(c) for c in vocab)
    romanize = args.romanize == "yes" or (args.romanize == "auto" and not has_hebrew)
    romanizer = None
    if romanize:
        import uroman

        romanizer = uroman.Uroman()
    token = processor.tokenizer.word_delimiter_token
    delimiter = vocab.get(token) if token else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    id_to_char = {v: k for k, v in vocab.items()}
    gaps_handle = None
    if args.gaps:
        args.gaps.parent.mkdir(parents=True, exist_ok=True)
        gaps_handle = args.gaps.open("a", encoding="utf-8", newline="\n")
    frames_handle = None
    if args.frames:
        args.frames.parent.mkdir(parents=True, exist_ok=True)
        frames_handle = args.frames.open("a", encoding="utf-8", newline="\n")
    ok, failed = 0, []
    with args.out.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            words = [clean(w) for w in row["words"]] if row.get("words") else clean(row["text"]).split()
            ids = []
            for w in words:
                text = romanizer.romanize_string(w).lower() if romanizer else w
                ids.append([vocab[c] for c in text if c in vocab])
            keep = [i for i, t in enumerate(ids) if t]
            if not keep:
                failed.append((row["id"], "no word the model can spell"))
                continue
            audio = torch.from_numpy(load_audio(row["audio"], float(row["duration"])))[None].to(args.device)
            with torch.inference_mode():
                emission = torch.log_softmax(model(audio).logits, dim=-1)
            # A wav2vec2 head trained with a word delimiter expects one between words;
            # without it the path drifts. MMS has none and must not get one.
            flat = []
            for pos, i in enumerate(keep):
                if pos and delimiter is not None:
                    flat.append(delimiter)
                flat.extend(ids[i])
            # forced_align has no CUDA kernel; the Viterbi runs on CPU.
            emission = emission.cpu()
            try:
                aligned, scores = AF.forced_align(emission, torch.tensor([flat], dtype=torch.int32), blank=blank)
            except Exception as exc:  # noqa: BLE001 -- one clip must not stop the rest
                failed.append((row["id"], f"{type(exc).__name__}: {exc}"[:100]))
                continue
            # merge_tokens defaults to blank=0, right for MMS and wrong for a head whose
            # blank is elsewhere -- pass it explicitly or every word slides early.
            spans = AF.merge_tokens(aligned[0], scores[0].exp(), blank=blank)
            if len(spans) != len(flat):
                failed.append((row["id"], f"{len(spans)} spans for {len(flat)} targets"))
                continue
            ratio = audio.shape[-1] / emission.shape[1] / SAMPLE_RATE
            timed, cursor = [], 0
            for pos, i in enumerate(keep):
                if pos and delimiter is not None:
                    cursor += 1
                chunk = spans[cursor : cursor + len(ids[i])]
                cursor += len(ids[i])
                if chunk:
                    # How sure the model was of the letters it placed here, averaged over
                    # the frames they occupy. A word that was never spoken still has to go
                    # somewhere: forced alignment cannot refuse one, it can only place it
                    # cheaply. This is where that shows.
                    frames = sum(s.end - s.start for s in chunk)
                    score = sum(s.score * (s.end - s.start) for s in chunk) / max(frames, 1)
                    timed.append({"word": words[i], "start": round(chunk[0].start * ratio, 4),
                                  "end": round(chunk[-1].end * ratio, 4),
                                  "score": round(float(score), 4)})
            handle.write(json.dumps({"id": row["id"], "words": timed}, ensure_ascii=False) + "\n")
            probs = emission[0].exp() if (gaps_handle or frames_handle) else None
            if frames_handle is not None:
                frames_handle.write(json.dumps(
                    {"id": row["id"], "ratio": round(ratio, 6),
                     "blank": [round(v, 3) for v in probs[:, blank].tolist()]},
                    ensure_ascii=False) + "\n")
            if gaps_handle is not None:
                gaps = gap_rows(timed, probs[:, blank].tolist(), ratio,
                                float(row["duration"]),
                                emission[0].argmax(dim=-1).tolist(), id_to_char)
                gaps_handle.write(json.dumps(
                    {"id": row["id"], "audio": row["audio"], "gaps": gaps},
                    ensure_ascii=False) + "\n")
            ok += 1

    for closing in (gaps_handle, frames_handle):
        if closing is not None:
            closing.close()
    record_failures(args.out, failed)
    print(f"  aligned {ok}, failed {len(failed)}", flush=True)
    for cid, why in failed:
        print(f"  FAILED {cid}: {why}")


if __name__ == "__main__":
    main()
