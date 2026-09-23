# Aligner evaluation

Scores forced aligners against the human word boundaries marked in the tool, with
significance tests, and shows the result on the dashboard (`?view=eval`). Re-runnable: after
more tagging, run it again and only the new clips are aligned.

## The loop

1. **Get the marks.** On the dashboard, *download marks (.jsonl)* — or `GET /api/export`
   while signed in.
2. **Get the dataset** the marks belong to (audio + manifest):
   ```
   python -m hebrew_training.download_dataset --dataset ivrit-ai
   ```
   Needs the bucket credentials in `.env`, same as `upload_dataset.py`.
3. **Run:**
   ```
   python eval/run_eval.py --marks gold.jsonl --dataset data/datasets/ivrit-ai
   ```
   Aligns every marked clip with every aligner, scores them, and prints the command to open
   the result on a local dashboard. Everything lands in `data/eval_runs/<dataset>/`.

## What is scored

| name | what | environment |
|---|---|---|
| *(the dataset's own label)* | e.g. `ivrit-ai`: the timings shipped with the data (Whisper + stable-ts) | none |
| `wav2vec2-hebrew` | CTC forced alignment — `imvladikon/wav2vec2-xls-r-300m-hebrew` | `EVAL_PY_CTC` |
| `mms` | CTC forced alignment — Meta MMS, `MahmoudAshraf/mms-300m-1130-forced-aligner` | `EVAL_PY_CTC` |
| `whisper-stable-ts` | stable-ts `align()` on faster-whisper, `ivrit-ai/whisper-large-v3-turbo-ct2` | `EVAL_PY_STABLE_TS` |
| `mwa-buckeye` | Multilingual Word Aligner (arXiv:2606.10675), `buckeye` checkpoint — the align button's model | `EVAL_PY_MWA` |

All five time **the dataset's own word list** — the one annotators saw and edited — so every
aligner word pairs with a human word. Metric: |aligner − human| on each word start and end,
reported as median, p90 and share within 10/25/50/100 ms. Significance: clip-level paired
bootstrap (2,000 draws), Holm-corrected. The dataset's first label is the one the tool opens
clips on, so it is flagged as having a head start. See the docstring of
`hebrew_training/aligner_eval.py` for why each of these choices was made.

## One-time setup

The aligners have incompatible dependencies, so each gets its own environment. Point `.env`
at their interpreters; an aligner whose variable is unset is skipped and the rest still run.

```
# .env
EVAL_PY_CTC=C:/path/to/ctc-env/Scripts/python.exe
EVAL_PY_STABLE_TS=C:/path/to/stable-ts-env/Scripts/python.exe
EVAL_PY_MWA=C:/path/to/Multilingual-Word-Aligner/.venv/Scripts/python.exe
EVAL_MWA_REPO=C:/path/to/Multilingual-Word-Aligner
```

A CUDA GPU is assumed (cu121 wheels below). On 72 clips with an RTX 4060: CTC ~1 min each,
stable-ts ~15 s, MWA a few minutes.

**CTC (wav2vec2-hebrew, mms)**
```
uv venv --python 3.11 ctc-env
uv pip install --python ctc-env torch torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install --python ctc-env transformers uroman soundfile scipy
```
The load prints *"Some weights … were not initialized … pos_conv_embed"*. It is a false
alarm: the hebrew checkpoint stores that layer under old names and transformers maps them.
Checked — the loaded weights equal the checkpoint's.

**stable-ts**
```
uv venv --python 3.11 stable-ts-env
uv pip install --python stable-ts-env torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python stable-ts-env stable-ts==2.19.1 faster-whisper==1.2.1 soundfile "numpy<2"
```
`numpy<2` because torch 2.4 is built against 1.x. On Windows torch and ctranslate2 carry
different OpenMP runtimes; `run_eval.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` with
`OMP_NUM_THREADS=1`, which is what makes that combination safe.

**MWA**
```
git clone https://github.com/MLSpeech/Multilingual-Word-Aligner
cd Multilingual-Word-Aligner
uv venv --python 3.11 .venv
uv pip install --python .venv torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv numpy==1.26.4 scipy librosa soundfile pandas matplotlib seaborn \
    praatio pydantic einops huggingface-hub safetensors boltons tqdm PyYAML uroman wandb
uv pip install --python .venv --no-deps -e .
```
Its `requirements.txt` pins Linux-only packages (triton, nvidia-*), hence the list above.
It cannot align words containing digits — those clips are reported and skipped.

## Which transcript the aligners get

`--text dataset` (default) gives them the dataset's original transcript. That is the
realistic case: it is what they would get over a whole untagged corpus, missing words and
all. On these 72 clips the humans added 15% more words than the transcript had, and an
aligner with no word for a stretch of speech must stretch some word across it.

`--text corrected` gives each annotator's own edited word list, which measures timing skill
alone. A clip is then aligned once per annotator, since two people do not always correct it
the same way. Two caveats:

- A label the dataset ships (e.g. `ivrit-ai`) is **not** re-run -- it is stored timings, made
  against the original text. Its row in a corrected run is not comparable with the rest.
- Where annotators corrected a clip differently, a dataset row can only hold one label per
  source, so the lanes show one person's alignment. `eval.json` scores each person against
  their own, so the dashboard's recomputed figures differ slightly there (3 of 72 clips
  here). `eval.json` is the one to quote.

## Viewing a run

```
python -m hebrew_training.align_tag_server --datasets-folder data/eval_runs/ivrit-ai \
    --dataset dataset --out data/eval_runs/ivrit-ai/marks --port 8091
```
then http://localhost:8091/?view=eval. Each clip in the per-clip table has an *open* link
that shows every aligner's words as toggleable lanes under the human marks.

## Correcting MMS

`mms-corrected` is not another aligner. It is MMS's own output moved, so **MMS does not have
to be run again** — the correction needs the existing timings plus the audio, and no model.

Two rules, both fitted on the 72 marked clips and measured on clips they were not fitted on:

- **every boundary** moves outward by an amount set by the letter *at that boundary*: the
  word's first letter sets its start, its last letter sets its end. Starts move much more
  than ends, because a word's onset is where CTC is latest.

        letter class            start     end
        plosive   ב ג ד כ פ ת    -2 ms    -3 ms
        nasal     מ נ           +23 ms    +9 ms
        glottal   א ה ע י       +24 ms   +24 ms
        liquid    ל ר           +30 ms    +8 ms
        fricative ש ס ז ח ו     +44 ms   +16 ms

  Text and timings only, no audio.
- **a word end with 100 ms or more of silence after it** is extended to where the sound
  actually stops: forward while the envelope stays above a fifth of that word's own peak, up
  to 100 ms. This one reads the waveform.

The second rule is the one that matters — it is where MMS is worst, cutting about 60 ms
early — so running text-only gets roughly half the benefit.

    29.5 -> 23.4 ms median, p90 90.5 -> 79.0        all boundaries
    62.8 -> 54.0 ms median, p90 187 -> 133          word ends before a pause

Measure it, and write the fitted numbers to `<run>/correction.json`:

```
python eval/correct_mms.py --run data/eval_runs/ivrit-ai
```

Apply it to a whole dataset, adding `mms-corrected` beside the existing labels (clips with
no `mms` label are left alone):

```
python eval/push_corrected.py --manifest manifest.jsonl --audio data/datasets/ivrit-ai \
    --correction data/eval_runs/ivrit-ai/correction.json --out manifest.new.jsonl
```

Needs numpy and soundfile only. `correction.json` can be reused as it is, or refitted on
more marks by re-running the first command.
