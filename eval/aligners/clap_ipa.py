"""Word alignment with IPA-Aligner (CLAP-IPA, Zhu et al., NAACL 2024).

    python clap_ipa.py --manifest clips.jsonl --out clap-ipa.jsonl

A dual encoder: one tower embeds speech frames, the other embeds IPA, and the two are
matched by cosine similarity and decoded with DTW. It takes IPA rather than text, which is
what makes it usable for Hebrew at all -- and we already produce Hebrew IPA with Phonikud,
so the input it wants is a pipeline we have.

Unlike MMS, this is the right place for phonemes: IPA is the model's native input, not a
transliteration of spelling it was never trained on.

Reimplemented from the repository's `forced_alignment_example.ipynb`, word mode. Two things
about the method are worth knowing before reading the numbers:

- **It returns starts, not ends.** DTW assigns every speech frame to exactly one unit, so a
  word's end is the next word's start. The alignment tiles the audio with no gaps, the same
  property that hurt MWA on word ends here.
- **Frames are 20 ms**, so 20 ms is the floor on its resolution.

Input rows: {id, audio, duration, text, words}. Output rows: {id, words}. Clips already in
--out are skipped.

Needs the clap-env interpreter (torch, transformers, librosa, the clap package, phonikud).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

STRESS, PREFIX = "ˈ", "|"
SEP = "[SEP]"
FRAME = 0.02
_PUNCT = re.compile(r"[^\w֐-׿']+", re.UNICODE)


def done_ids(out: Path) -> set[str]:
    ids = set()
    for path in (out, out.with_suffix(".failed.jsonl")):
        if path.exists():
            ids |= {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    return ids


def phone_mask(lengths, torch):
    """Average the tokens of each word into one vector, as the paper's code does."""
    mask = torch.zeros(len(lengths), sum(lengths))
    at = 0
    for i, n in enumerate(lengths):
        mask[i, at:at + n] = 1 / n + 1e-8
        at += n
    return mask


def align_units(cost, np):
    """DTW through the similarity matrix; the last frame each unit holds.

    `cost` is (frames, units). The step pattern is the paper's: a unit may take many frames,
    but the sequence never goes backwards, which is what makes this a forced alignment
    rather than a search.
    """
    from librosa.sequence import dtw

    _, path = dtw(C=cost.T, step_sizes_sigma=np.array([[1, 1], [0, 1]]))
    last = [-1] * (int(path[:, 0].max()) + 1)
    for unit, frame in path:
        if last[unit] < frame:
            last[unit] = int(frame)
    return last


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", default="base", choices=["tiny", "base", "small"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    skip = done_ids(args.out)
    rows = [r for r in rows if r["id"] not in skip]
    print(f"clap-ipa ({args.size}): {len(rows)} clips to align ({len(skip)} already done)", flush=True)
    if not rows:
        return

    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from clap.encoders import PhoneEncoder, SpeechEncoder
    from huggingface_hub import hf_hub_download
    from phonikud import phonemize
    from phonikud_onnx import Phonikud
    from transformers import AutoProcessor, DebertaV2Tokenizer

    device = torch.device(args.device)
    speech_encoder = SpeechEncoder.from_pretrained(f"anyspeech/ipa-align-{args.size}-speech").eval().to(device)
    phone_encoder = PhoneEncoder.from_pretrained(f"anyspeech/ipa-align-{args.size}-phone").eval().to(device)
    tokenizer = DebertaV2Tokenizer.from_pretrained("charsiu/IPATokenizer")
    processor = AutoProcessor.from_pretrained(f"openai/whisper-{args.size}")
    g2p = Phonikud(hf_hub_download("thewh1teagle/phonikud-onnx", "phonikud-1.0.int8.onnx"))

    def to_ipa(word: str) -> str:
        bare = _PUNCT.sub("", word).strip()
        if not bare:
            return ""
        try:
            out = phonemize(g2p.add_diacritics(bare))
        except Exception:  # noqa: BLE001
            return ""
        return "".join(c for c in out if c not in (STRESS, PREFIX) and not c.isspace())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    with args.out.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            words = row.get("words") or row["text"].split()
            ipa = [to_ipa(w) for w in words]
            keep = [i for i, s in enumerate(ipa) if s]
            if not keep:
                failed.append((row["id"], "no word could be turned into IPA"))
                continue

            wav, rate = sf.read(row["audio"], dtype="float32", always_2d=True)
            wav = wav[: int(float(row["duration"]) * rate)].mean(axis=1)
            if rate != 16000:
                import librosa

                wav = librosa.resample(wav, orig_sr=rate, target_sr=16000)
            # The mel input stays at Whisper's full 30 s and the mask stays its full
            # length: clap's encoder subsamples the mask itself (`attention_mask[:, ::2]`)
            # to match the conv stride, so trimming either one first makes the two
            # disagree. The encoder always returns 1500 frames; the ones past the end of
            # the audio are dropped afterwards instead.
            batch = processor([wav], sampling_rate=16000, return_attention_mask=True,
                              return_tensors="pt")
            frames = max(1, int(round(len(wav) / 16000 / FRAME)))

            # A boundary either side, so the first word has something to start after and the
            # last has something to end before.
            units = [SEP] + [ipa[i] for i in keep] + [SEP]
            tokens = tokenizer(units, return_attention_mask=False, return_length=True,
                               return_token_type_ids=False, add_special_tokens=False)
            ids = torch.tensor(list(itertools.chain.from_iterable(tokens["input_ids"]))).long().unsqueeze(0)
            try:
                with torch.no_grad():
                    speech = speech_encoder(
                        input_features=batch["input_features"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).last_hidden_state.squeeze(0)[:frames]
                    phones = phone_encoder(ids.to(device)).last_hidden_state.squeeze(0)
                per_unit = torch.matmul(phone_mask(tokens["length"], torch).to(device), phones)
                sim = torch.matmul(F.normalize(speech, dim=-1), F.normalize(per_unit, dim=-1).t())
                last = align_units(-sim.cpu().numpy(), np)
            except Exception as exc:  # noqa: BLE001 -- one clip must not stop the rest
                failed.append((row["id"], f"{type(exc).__name__}: {exc}"[:110]))
                continue
            if len(last) < len(units):
                failed.append((row["id"], f"{len(last)} boundaries for {len(units)} units"))
                continue

            # Unit u's last frame is where unit u+1 begins, so word k runs from the end of
            # unit k to the end of unit k+1.
            timed = []
            for k, i in enumerate(keep):
                start, end = last[k] * FRAME, last[k + 1] * FRAME
                timed.append({"word": words[i], "start": round(start, 4),
                              "end": round(max(end, start + FRAME), 4)})
            handle.write(json.dumps({"id": row["id"], "words": timed}, ensure_ascii=False) + "\n")
            ok += 1
            if ok % 10 == 0:
                print(f"  {ok} aligned", flush=True)

    if failed:
        with args.out.with_suffix(".failed.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            for cid, why in failed:
                fh.write(json.dumps({"id": cid, "why": why}, ensure_ascii=False) + "\n")
    print(f"  aligned {ok}, failed {len(failed)}", flush=True)
    for cid, why in failed[:10]:
        print(f"  FAILED {cid}: {why}")


if __name__ == "__main__":
    main()
