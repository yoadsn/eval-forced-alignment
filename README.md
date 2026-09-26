# eval-forced-alignment

Scores Hebrew forced aligners against human word boundaries, with clip-level bootstrap
intervals and Holm-corrected paired tests. The output is one self-contained `result.json`.

The gold set uses the "publish labeled dataset" format produced by the
[audio_alignment_tool](https://github.com/asaelbarilan/audio_alignment_tool) tagging tool:
its `publish_labeled_dataset.py` script exports the clips marked *done* as an AudioFolder
(`metadata.jsonl` + `audio/`), either to a Hub dataset or to a local folder. The two projects
can be used independently, but this project's `result.json` output can also be uploaded to
the tagging tool's **aligner eval** screen, which has a dedicated viewer for it.

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

| name | runner | env | device support |
|---|---|---|---|
| `wav2vec2-hebrew`, `mms` | `ctc.py` | `envs/ctc` | cpu, cuda, or mps (Apple Silicon Macs) |
| `whisper-stable-ts` | `stable_ts.py` | `envs/stable_ts` | cpu only, slow |
| `mwa-buckeye`, `mwa-timit` | `mwa.py` | MWA's own venv (`EFA_PY_MWA`, `EFA_MWA_REPO`) | cpu |
| `clap-ipa` | `clap_ipa.py` | `envs/clap` | cpu |
| `mfa-viter` | `viter_align.py` | `envs/viter` + the viter binary and a trained model | needs `EFA_VITER_*` |
| `mms-corrected` | derived from `mms` by `correction/` | the orchestrator's own | inherits from `mms` |

The interpreter for env `X` is `envs/X/.venv`, built with `uv sync --project envs/X`. You can
override it with `EFA_PY_X` in `.env`. Each aligner uses the first device in its `devices`
list that its interpreter can see. Pass `--device` to force one.

**If an aligner can't run here** (missing env, no GPU, unconfigured model), it still appears
in the result as `unavailable`, with the reason. To score it anyway:

1. Run its runner on a machine that can, using this run's `clips.jsonl`.
2. Copy `labels/<name>.jsonl` (and its `.meta.json`, if any) into this run folder.
3. Run again with `--no-align`.

The aligner then shows up as `imported`, with the other machine's provenance.

## Working with aligner envs

Each aligner runs in its own interpreter because their dependencies conflict (different
torch/numpy pins, or a native binary). `aligners.toml`'s `env = "<name>"` says which one a
given aligner uses. Most envs are plain `uv` projects under `envs/<name>/`; a couple point at
an external project's own environment instead.

**uv-project envs** (`ctc`, `stable_ts`, `clap`, `viter`) are built with:

    uv sync --project envs/<name>

This creates `envs/<name>/.venv`, the interpreter `efa/registry.py` looks for by default. You
only need to build the ones you plan to use:

    uv sync --project envs/ctc        # wav2vec2-hebrew, mms
    uv sync --project envs/stable_ts  # whisper-stable-ts
    uv sync --project envs/clap       # clap-ipa
    uv sync --project envs/viter      # mfa-viter (also needs the binary + a model, below)

An aligner whose env isn't built simply appears as `unavailable` in the result, with the
reason, and the rest still run.

**MWA** (`mwa-buckeye`, `mwa-timit`) has no `envs/mwa`; it runs inside a clone of MWA's own
repo (`github.com/MLSpeech/Multilingual-Word-Aligner`), in that repo's own virtualenv. Clone
it, build its venv per its own instructions (`research/README.legacy.md` has the exact
package list this project was checked against), then point `.env` at it:

    EFA_PY_MWA=/path/to/Multilingual-Word-Aligner/.venv/bin/python
    EFA_MWA_REPO=/path/to/Multilingual-Word-Aligner

**mfa-viter** needs more than `envs/viter` (which only has the Python side: phonikud, for
building a dictionary). It also needs the `viter` binary itself and a trained model, since
there is no official Hebrew MFA model:

1. Install the binary: `cargo binstall --git https://github.com/thewh1teagle/viter viter`,
   or download one from its releases page.
2. Build a corpus and dictionary: `uv run --project envs/viter python aligners/viter_setup.py
   --dataset <dataset> --out data/viter` (optionally widen the corpus first with
   `aligners/knesset_corpus.py`).
3. Train: `viter train data/viter/corpus --dict data/viter/dict.txt -o data/viter/hebrew.viter`.
   The model must not have seen the gold clips.
4. Point `.env` at all three:

       EFA_VITER_MODEL=data/viter/hebrew.viter
       EFA_VITER_DICT=data/viter/dict.txt
       EFA_VITER_BIN=/path/to/viter        # or "viter" if it's on PATH

**Overriding an interpreter.** Any env can be pointed elsewhere with `EFA_PY_<ENV>`
(uppercased) in `.env` instead of building `envs/<name>/.venv` -- this is how MWA's own venv
is wired in, and it works the same way for any of the others.

**Devices.** Each resolved interpreter is probed once per run for which of cpu/cuda/mps it
can see (`registry.probe`); the first device in the aligner's `devices` list (`aligners.toml`)
that its interpreter has wins, unless `--device` forces one.

## Adding a new aligner

1. **Write a runner** under `aligners/`, following the contract in [Aligners](#aligners):
   read `--manifest`/`--out`/`--device` (plus whatever extra `args` it needs), skip ids
   already in `--out` or `<out>.failed.jsonl`, and write one `{"id", "words": [{"word",
   "start", "end", ...}]}` row per aligned clip. An existing runner (`aligners/ctc.py` is the
   simplest) is the easiest template.
2. **Give it an interpreter.** Reuse an existing `envs/<name>` if its dependencies fit,
   otherwise add a new `uv` project: `envs/<name>/pyproject.toml` listing just that aligner's
   dependencies (`package = false` under `[tool.uv]`, like the others), then
   `uv sync --project envs/<name>`.
3. **Register it in `aligners.toml`**:

       [aligners.<name>]
       runner = "<runner>.py"
       env = "<name>"
       args = ["--model", "..."]        # anything beyond --manifest/--out/--device
       devices = ["cuda", "mps", "cpu"] # best first; the first one the interpreter has wins
       needs = ["EFA_SOME_VAR"]         # optional: env vars that must be set, substituted into args as {NAME}
       env_vars = { SOME = "value" }    # optional: extra environment for the subprocess
       description = "shown in the viewer beside the aligner's name"

4. **Run it**: `uv run python -m efa.run --input ... --aligners <name>` to align and score
   just the new one against an existing run. If the env or a `needs` variable is missing, it
   shows up as `unavailable` with the reason instead of failing the whole run.
5. If it belongs in the quick-reference table, add a row to it in [Aligners](#aligners).

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
  tagging tool's aligner-eval viewer has always read.
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
