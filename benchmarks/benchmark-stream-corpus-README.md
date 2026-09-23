# macOS `say` streaming corpus benchmark

This benchmark compares the current `160/480/4000/stability=2` streaming
window with candidate `160/320/2000/stability=2`. It makes no production
changes. The run reused the existing 12 WAV files and manifest in
`/tmp/ptt-stream-corpus`. The script checked every synthesis recipe and WAV
SHA-256 before loading the model, so the prompts and audio hashes match the
prior corpus.

The corpus has 12 labeled prompts and 46.44 seconds of speech. It covers code
terms, filenames, acronyms, numbers, punctuation, short and long commands,
five English voices, rates from 155 to 210 words per minute, and one explicit
700 ms pause. The result JSON retains the recipe, duration, and SHA-256 for
every WAV.

## Run

From the repository root:

```bash
.venv/bin/python benchmarks/benchmark-stream-corpus.py \
  --reuse-corpus \
  --output benchmarks/benchmark-stream-corpus-results.json
```

The script uses real-time 20 ms frames and queues a 320 ms silence tail when
the real PCM ends. It shares one warm model and alternates preset order for
successive prompts. `/tmp/ptt-mps-benchmark.lock` covers model load and every
MPS run, so a concurrent benchmark fails instead of sharing the GPU.

This run used macOS 15.5 on Apple silicon, Kestrel 0.8.0,
`kestrel-kernels` 0.7.0, and torch 2.14.0. It is one deterministic corpus
pass, not a statistical confidence interval.

## Results

WER and CER are pooled edit counts divided by pooled reference units. Raw
comparison preserves case and punctuation. Normalization applies Unicode
NFKC, case folding, punctuation-to-space, and whitespace collapse. It does
not apply inverse text normalization for spoken numbers. Normalized CER
excludes spaces.

| metric | current 160/480/4000/s2 | candidate 160/320/2000/s2 |
|---|---:|---:|
| exact final matches | 5/12 (41.67%) | 5/12 (41.67%) |
| normalized final matches | 9/12 (75.00%) | 9/12 (75.00%) |
| raw WER / CER | 19.23% / 9.75% | 19.23% / 9.75% |
| normalized WER / CER | 8.11% / 7.96% | 8.11% / 7.96% |
| first-stable coverage | 12/12 | 12/12 |
| first stable median / p95 | 1002 / 1213 ms | 847 / 3083 ms |
| release-final median / p95 | 338 / 1167 ms | 368 / 1080 ms |
| raw churn edits / retractions | 605 / 0 | 469 / 0 |
| stable churn edits / retractions | 527 / 0 | 372 / 0 |
| release tail missing words / word edits | 74 / 35 | 95 / 64 |
| release tail utterances with errors | 12/12 | 12/12 |
| release tail completion | 36.21% | 18.10% |
| end-of-stream tail word edits / affected | 33 / 12 | 62 / 12 |
| model calls | 285 | 297 |
| decoded input seconds | 769.80 | 602.84 |
| audio compute amplification | 15.309x | 11.989x |

`first stable` is the first text emitted by the production
`InterimTranscriptStabilizer`. P95 uses nearest rank. The candidate median was
155 ms earlier, but its p95 was 1.87 seconds later because the numbers prompt
did not become stable until 3083 ms. `release-final` starts when the final real
PCM sample is available, before the synthetic tail.

Churn is the sum of character Levenshtein edits between consecutive interim
snapshots. Retractions count characters removed after the snapshots' longest
common prefix. Final output is not appended to churn. The candidate reduced
raw churn edits by 22.5% and stable churn edits by 29.4%; neither preset
retracted characters.

Release tail loss compares the last stable snapshot at or before real-audio
release with the final result. Completion is the pooled common word prefix
divided by final words. End-of-stream tail loss instead uses the last stable
snapshot before the final result. The candidate's lower churn came with much
less complete previews: completion halved, release word errors rose from 35
to 64, and end-of-stream word errors rose from 33 to 62.

Amplification is total audio seconds presented across decoder calls divided by
source seconds including the synthetic tail. The candidate reduced decoded
audio amplification by 21.7%, despite making 4.2% more calls. The JSON retains
every decoded frame count for audit.

## Interpretation and recommendation

Both presets produced identical final text on all 12 prompts. Therefore their
exact-match rates, WER, and CER are identical. EOF performs the same exact
full-buffer replay.

The candidate improves median time to stable text and reduces decoder audio,
but it has a severe tail-completeness regression and an unstable latency tail.
The release-final median is also 30 ms slower in this pass.

**Keep the current `160/480/4000/stability=2` preset.** Do not move to
`160/320/2000/stability=2` based on this corpus pass.
