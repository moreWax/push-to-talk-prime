# Early single-model draft preview benchmark

## Decision

Do **not** change production. Keep `balanced` unchanged.

The serialized speculative variants passed the explicit authoritative p95 latency gate, but the drafts were not reliable enough to ship. Only 10 of 12 drafts were non-empty, and only 4 of those 10 were normalized whole-word prefixes of the matched authoritative final. The parallel variants are rejected because they materially worsened an authoritative p95 metric.

## Experiment

This is a benchmark-only evaluation of a candidate preset named `speculative`. The baseline preset is `balanced`, with its authoritative request unchanged:

- live chunk: 160 ms
- right context: 480 ms
- left context: 4000 ms
- `InterimTranscriptStabilizer(2)`
- `timestamps="none"`

The benchmark reuses and hash-checks the existing 12-prompt corpus at `/tmp/ptt-stream-corpus`. It feeds 20 ms frames on real-time, end-of-frame deadlines. One warm Photon/Parakeet engine is shared by all runs under `/tmp/ptt-mps-benchmark.lock`. Candidate order rotates by one position for every utterance.

Each speculative run launches exactly one independent stateless `timestamps="none"` draft on either 320 ms or 480 ms of captured PCM. Parallel variants permit the draft and live inference to overlap. Serialized variants pause before yielding the cutoff-crossing live frame, run the draft while capture continues, and then resume the authoritative stream. This prevents model-call overlap while exposing queued-audio cost.

## Aggregate results

| Candidate | Draft non-empty | Draft visible median / p95 | Whole-word prefix | First visible median / p95 | Auth first raw p95 Δ | Auth first stable p95 Δ | Release-final p95 Δ | Backlog-lag p95 Δ | Exact finals vs balanced | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `balanced` | — | — | — | 1023 / 1219 ms | — | — | — | — | 12/12 | pass |
| `speculative-parallel-320` | 10/12 | 394 / 444 ms | 4/10 | 395 / 1024 ms | -12 ms | -32 ms | +103 ms | +507 ms | 12/12 | **reject** |
| `speculative-serialized-320` | 10/12 | 393 / 397 ms | 4/10 | 393 / 1025 ms | -2 ms | -31 ms | -48 ms | -7 ms | 12/12 | pass |
| `speculative-parallel-480` | 10/12 | 557 / 596 ms | 4/10 | 558 / 1029 ms | -15 ms | -28 ms | +207 ms | -1 ms | 12/12 | **reject** |
| `speculative-serialized-480` | 10/12 | 555 / 573 ms | 4/10 | 556 / 1025 ms | +22 ms | -30 ms | -45 ms | +5 ms | 12/12 | pass |

The material-regression rule is a p95 increase greater than `max(100 ms, 10% of balanced)` for authoritative first raw, first stable, release-final, or maximum consume lag. Negative deltas are improvements/noise, not claimed speedups.

## Quality, churn, and compute

- The 320 ms drafts had normalized reference WER 0.950. The 480 ms drafts had 0.9059. This compares an intentionally incomplete prefix with the full utterance, because `timestamps="none"` provides no aligned cutoff reference; treat it only as a coverage/error indicator.
- The 320 ms draft contained 10 words across the corpus, with 4 common-prefix words. The 480 ms draft contained 15 words, with 8 common-prefix words.
- Balanced stable output retracted 0 characters. Adding drafts caused 25 retracted characters at 320 ms and 34 at 480 ms in the combined visible timeline.
- Authoritative compute was identical for every candidate: 285 live model invocations and 769.8046 decoded input seconds across the corpus.
- Exactly 12 stateless calls added 3.84 logical decoded seconds at 320 ms or 5.76 seconds at 480 ms. Median stateless wall time was about 72–75 ms. Wall time is not pure accelerator time.
- Every speculative authoritative final matched its paired balanced final exactly and after normalization: 12/12 for all four variants.

The serialized results show that an early draft can avoid a material authoritative p95 regression in this sample. They do not justify the preset because coverage, prefix correctness, and visible churn are poor. The parallel results also show a long-tail risk on the shared engine.

## Reproduce

From the repository root:

```bash
.venv/bin/python benchmarks/benchmark-early-draft-corpus.py \
  --corpus-dir /tmp/ptt-stream-corpus \
  --output benchmarks/benchmark-early-draft-corpus-results.json
```

The script requires Apple silicon with MPS. It rejects a missing, changed, or reordered corpus manifest/WAV set. The recorded report contains all 60 per-run rows, package/platform versions, exact metric definitions, candidate order, parity, and gate fields.
