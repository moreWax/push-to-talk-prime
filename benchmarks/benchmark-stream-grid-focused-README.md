# Focused streaming context grid (Apple MPS)

This follow-up holds interim stability at 2 and tests chunk 160 ms across
left context `{1000, 2000, 4000}` ms and right context `{320, 480}` ms. It
also includes the optional chunk 240 / right 480 / left 2000 point. Each point
ran twice with the second round in reverse order. One warm model was shared and
`/tmp/ptt-mps-benchmark.lock` serialized MPS access.

Command:

```bash
.venv/bin/python benchmarks/benchmark-stream-grid.py \
  --matrix focused --rounds 2 \
  --output benchmarks/benchmark-stream-grid-focused-results.json
```

Every one of the 14 final transcripts exactly matched:

> Refactor the authentication middleware and add tests for expired tokens,
> malformed tokens, and missing tokens.

## Two-round averages

| chunk/right/left (ms) | first raw | first stable | release-final | raw/stable edits | retractions | calls | amplification |
|---|---:|---:|---:|---:|---:|---:|---:|
| 160/320/1000 | 550 ms | 862 ms | 335 ms | 91 / 79 | 0 / 0 | 35 | 9.065x |
| 160/480/1000 | 696 ms | 1172 ms | 309 ms | 95 / 79 | 0 / 0 | 34 | 9.722x |
| 160/320/2000 | 534 ms | 857 ms | 439 ms | 69 / 64 | 0 / 0 | 35 | 13.233x |
| 160/480/2000 | 694 ms | 1173 ms | 878 ms | 69 / 60 | 0 / 0 | 34 | 13.717x |
| 160/320/4000 | 533 ms | 848 ms | 337 ms | 71 / 66 | 0 / 0 | 35 | 18.333x |
| 160/480/4000 | 694 ms | 1173 ms | 338 ms | 69 / 60 | 0 / 0 | 34 | 18.472x |
| 240/480/2000 | 792 ms | 1272 ms | 316 ms | 67 / 60 | 0 / 0 | 23 | 9.722x |

The JSON retains every per-round raw and stable cadence statistic, churn field,
decoded frame count, latency, transcript, and amplification input.

## Best practical Pareto candidates

- **160/320/1000** is the low-work responsive endpoint. It gives 9.065x
  amplification and about 862 ms first-stable latency, but has the most visible
  churn among the recommended points (79 stable edits) and a less clean last
  preview.
- **160/320/2000** is the balanced knee. It keeps first-stable latency near
  857 ms while cutting stable edits from 79 to 64. The cost is 13.233x
  amplification. Its release-final values were 298 and 579 ms.
- **240/480/2000** is the efficiency/stability endpoint. It ties the minimum
  stable churn (60 edits), uses 23 calls and 9.722x amplification, and has very
  consistent 314-319 ms finalization. The tradeoff is about 1272 ms to first
  stable text.

The 4000 ms left-context points do not look like useful knees: their extra work
buys little or no churn or interim-latency improvement over a 2000 ms point.
The 160/480/2000 point had a 1433 ms release-final outlier in round 2, so its
878 ms two-run average should not be treated as a stable steady-state estimate.
These are still screening results from one short clean WAV, not confidence
intervals.
