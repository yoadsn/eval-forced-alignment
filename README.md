# eval-forced-alignment

Scores Hebrew forced aligners against human word boundaries, with clip-level bootstrap
intervals and Holm-corrected paired tests. The output is one self-contained `result.json`,
which you upload to the tagging tool's **aligner eval** screen to view.

The gold set comes from the tagging tool (`audio_alignment_tool`). Its
`publish_labeled_dataset.py` exports the clips marked *done* as an AudioFolder
(`metadata.jsonl` + `audio/`), either to a Hub dataset or to a local folder.

```
uv sync
uv sync --project envs/ctc                          # at least one aligner environment
uv run python -m efa.run --input owner/gold-dataset   # or a local folder
```

The run folder is `runs/<input>-<revision>/`, and the result is `result.json` inside it.

## Input

This project reads one row per (clip, annotator):

    {"audio_file_name": "audio/<id>.wav", "id": "...", "metadata": {...},
     "text": "...", "words": [{"word", "start", "end"}], "annotator": "yoad"}

- `words` holds the human-corrected words. Every aligner is given exactly these words, so
  the score measures timing only, not transcript quality.
- `annotator` can be missing or empty (such rows count as `unknown`).
- A clip that appears under two annotators (`publish_labeled_dataset.py --all-annotators`)
  feeds the human-agreement floor. Each annotator's row is aligned separately, since two
  people can correct the same clip's words differently.

## Aligners

All aligners are listed in `aligners.toml`. Each one runs as a separate program in its own
interpreter, because their dependencies conflict. Every runner uses the same contract:

    <python> aligners/<runner>.py --manifest clips.jsonl --out labels.jsonl --device cpu|cuda|mps [args]

    in   {"id", "audio": <absolute path>, "duration", "text", "words": [str]}   per line
    out  {"id", "words": [{"word", "start", "end", ...}]}                       per line
         <out>.failed.jsonl: {"id", "why"} for clips it could not align

Rows already in `out` or `failed` are skipped, so a re-run only aligns new clips. If a clip's
words change, its old alignment is dropped and it is aligned again.

| name | runner | env | runs on this Mac |
|---|---|---|---|
| `wav2vec2-hebrew`, `mms` | `ctc.py` | `envs/ctc` | yes (mps) |
| `whisper-stable-ts` | `stable_ts.py` | `envs/stable_ts` | cpu only, slow |
| `mwa-buckeye`, `mwa-timit` | `mwa.py` | MWA's own venv (`EFA_PY_MWA`, `EFA_MWA_REPO`) | yes (cpu) |
| `clap-ipa` | `clap_ipa.py` | `envs/clap` | cpu |
| `mfa-viter` | `viter_align.py` | `envs/viter` + the viter binary and a trained model | needs `EFA_VITER_*` |
| `mms-corrected` | derived from `mms` by `correction/` | the orchestrator's own | yes |

The interpreter for env `X` is `envs/X/.venv`, built with `uv sync --project envs/X`. You can
override it with `EFA_PY_X` in `.env`. Each aligner uses the first device in its `devices`
list that its interpreter can see. Pass `--device` to force one.

**If an aligner can't run here** (missing env, no GPU, unconfigured model), it still appears
in the result as `unavailable`, with the reason. To score it anyway:

1. Run its runner on a machine that can, using this run's `clips.jsonl`.
2. Copy `labels/<name>.jsonl` (and its `.meta.json`, if any) into this run folder.
3. Run again with `--no-align`.

The aligner then shows up as `imported`, with the other machine's provenance.

## mms-corrected

This is MMS's own output, shifted per letter class. Before a pause, each word end is
extended to where the sound stops. See `correction/correct_mms.py` for the rule and how it
was measured. The correction needs fitting first, on a run that has `mms`:

```
uv run python correction/correct_mms.py --run runs/<run> --dataset runs/<run>/dataset
uv run python -m efa.run --input ... --no-align
```

A `correction.json` inside the run folder beats the committed `correction/correction.json`.
The result then records `"in_sample": true`, which means the correction's own table row is
flattering. For numbers to quote, use its `held_out` figures.

## The result

`result.json` (schema `efa-result/1`) contains everything except the audio:

- `input`: where the gold set came from, including the exact Hub commit or file hash.
- `producer`: this repo's git sha and the host it ran on.
- `aligner_status`: for every aligner, one of ok, imported, unavailable, failed, or error.
  It also records the reason, the clips that failed, and per-aligner provenance (device,
  packages, host).
- `aligners`, `human_agreement`, `comparisons`, `clips`: the scores, in the same shape the
  tagging tool's viewer has always read.
- `clip_data`: every clip's text, metadata, each annotator's words, and every aligner's
  words. The viewer uses these to draw lanes when you open a clip.

## Layout

    efa/          inputs (gold set), registry (aligners.toml), run (orchestrator), scoring, derive
    aligners/     one runner per aligner family, plus viter_setup/knesset_corpus for training viter
    envs/         one uv project per aligner environment
    correction/   the mms-corrected rule: fitting (correct_mms.py) and applying (push_corrected.py)
    research/     one-off analyses behind the write-ups; they read a run folder

`research/README.legacy.md` holds the setup notes from when this code lived in the tagging
repo, including the Windows/CUDA install lines.
