# First-visible latency and quality corpus comparison (Apple MPS)

This benchmark compares the production streaming windows and the production
`InterimTranscriptStabilizer` on the existing 12-prompt macOS `say` corpus.
It changes Kestrel's private window constants only inside the benchmark process.
There are no production changes.

- **A (baseline):** chunk/right/left `160/480/4000` ms, stability 2
- **B:** `160/320/4000` ms, stability 2
- **C:** `160/480/4000` ms, stability 1

C is not raw passthrough. Stability 1 still uses the currently coded
complete-word-boundary behavior.

## Method

```bash
.venv/bin/python benchmarks/benchmark-first-visible-corpus.py \
  --reuse-corpus --corpus-dir /tmp/ptt-stream-corpus \
  --output benchmarks/benchmark-first-visible-corpus-results.json
```

One warm model was shared by all 36 runs. Audio was fed in real time in 20 ms
frames, followed by the production 320 ms synthetic tail. Preset order rotated
`ABC`, `BCA`, `CAB` across utterances and repeated four times. The benchmark
held `/tmp/ptt-mps-benchmark.lock` across model load and every run. P95 uses the
nearest-rank definition; with 12 samples it is the observed maximum.

## Latency, coverage, compute, and final quality

| preset | first raw p50 / p95 | first stable p50 / p95 | stable coverage | release-final p50 / p95 | amplification | normalized WER | raw WER | final parity vs A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 696.59 / 743.77 ms | 1008.27 / 1225.17 ms | 12/12 (100%) | 316.45 / 684.00 ms | 15.309x | 8.108% | 19.231% | 12/12 |
| B | 534.18 / 2779.43 ms | 846.54 / 3094.68 ms | 12/12 (100%) | 314.36 / 391.14 ms | 15.283x | 8.108% | 19.231% | 12/12 |
| C | 693.15 / 755.40 ms | 1008.21 / 1176.00 ms | 12/12 (100%) | 314.53 / 503.24 ms | 15.309x | 8.108% | 19.231% | 12/12 |

All three presets also had the same exact-reference result: 5/12 exact and 9/12
normalized matches. Each B and C release-final transcript was byte-for-byte
identical to A for the same utterance.

B's latency maximum came from `numbers`: first raw was 2779.43 ms and first
stable was 3094.68 ms. The other B medians were faster, but this one-pass corpus
does not establish whether the outlier is repeatable.

## Interim churn and retractions

Counts are summed over the 12 utterances. Character edits are edit distance
between successive snapshots. Retracted characters are characters removed
past the shared character prefix.

| preset | raw snapshots / changed | raw edits / retractions | stable snapshots / changed | stable edits / retractions |
|---|---:|---:|---:|---:|
| A | 273 / 190 | 605 / 0 | 110 / 98 | 527 / 0 |
| B | 285 / 145 | 461 / 0 | 84 / 72 | 367 / 0 |
| C | 273 / 190 | 605 / 0 | 99 / 87 | 523 / 0 |

B emitted less changing text, but that did not mean more complete text: its
last stable previews covered much less of each final transcript. C reduced
stable churn only slightly relative to A and did not improve median first-stable
latency because the complete-word-boundary rule still withheld fragments.

## Tail completeness and errors

Completeness is the summed common-prefix word count divided by summed final
words after normalization. Errors are summed word edit distance from the last
stable preview to the release-final transcript. “At release” uses the last
stable preview present when source audio ended. “End of stream” also includes
any stable updates produced while processing the synthetic tail.

| preset | at-release completeness | missing words / word edits / affected | end-stream completeness | missing words / word edits / affected |
|---|---:|---:|---:|---:|
| A | 36.207% | 74 / 36 / 12 | 37.931% | 72 / 33 / 12 |
| B | 18.103% | 95 / 66 / 12 | 18.966% | 94 / 64 / 12 |
| C | 36.207% | 74 / 35 / 12 | 37.931% | 72 / 33 / 12 |

The common-prefix metric is intentionally strict: an early wrong word stops
credited completion even if later words are visible. The JSON retains every
snapshot-derived count, final transcript, error count, decoded-frame count,
and per-utterance latency.

## Result

B improved the median first-visible times by about 162 ms, but had a 2.78 s raw
and 3.09 s stable worst case, halved strict tail completeness, increased tail
word errors, and did not materially reduce compute amplification. C preserved
A's tail behavior and slightly reduced stable churn, but gave no meaningful
median first-stable improvement under the complete-word-boundary rule. All
three had identical final quality and pairwise final parity in this corpus.

For this 12-prompt, one-pass screen, neither B nor C gives a clear overall
reason to replace A. These results are comparative observations, not confidence
intervals.
