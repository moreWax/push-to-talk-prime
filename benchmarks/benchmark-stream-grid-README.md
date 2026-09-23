# Streaming context grid benchmark

`benchmark-stream-grid.py` measures the current Kestrel 0.8 live PCM path on
Apple MPS. It varies preview chunk, right context, left context, and interim
stability together in a deliberately small three-preset matrix. It does not
change `ptt_worker.py`, installed packages, or production defaults. The private
Kestrel constants are changed only in the benchmark process and restored on
exit.

## Run

From the repository root:

```bash
.venv/bin/python benchmarks/benchmark-stream-grid.py \
  --output benchmarks/benchmark-stream-grid-results.json
```

The script requires Apple silicon and MPS. It atomically creates
`/tmp/ptt-mps-benchmark.lock` with `O_CREAT|O_EXCL` before model loading. A
concurrent benchmark fails rather than sharing MPS. The lock contains its PID
and start time and is removed on exit. If the process is killed without cleanup,
verify that PID is gone before manually removing the stale lock.

The default input is `assets/demo-input.wav` (5.468 seconds). An independent
async producer supplies 20 ms frames at real-time speed. At source release it
queues the production-default 320 ms synthetic silence immediately, followed by
EOF. Thus `release_to_final_ms` measures decoder backlog plus final exact replay,
not a 320 ms endpointer wait.

## Metrics

- `first_raw_interim_ms`: first nonempty provisional hypothesis from capture
  start.
- `first_stable_interim_ms`: first hypothesis exposed by
  `InterimTranscriptStabilizer` at the preset's stability setting.
- `raw_update_cadence` and `stable_update_cadence`: counts and inter-update gap
  statistics. Stable cadence counts only visible text changes.
- `release_to_final_ms`: result availability after the source recording ends.
- `raw_churn` and `stable_churn`: character Levenshtein edits, changed
  transitions, and characters retracted past the common prefix.
- `final_equals_production`: exact normalized final-string equality against the
  production preset from the same run.
- `audio_compute_amplification`: total seconds of audio presented to all model
  invocations divided by source seconds including the synthetic tail. This is a
  deterministic work proxy, not a FLOP or energy measurement.

`decoded_frames_per_call` is retained in the JSON so amplification can be
audited rather than treated as an opaque summary.

## Initial MPS evidence

Environment: Kestrel 0.8.0, kestrel-kernels 0.7.0, torch 2.14.0, Apple MPS. One
warm model was used, with one real-time pass per preset. The complete record is
in `benchmark-stream-grid-results.json`.

| preset (chunk/right/left, stability) | first raw | first stable | raw cadence mean | release to final | calls | amplification |
|---|---:|---:|---:|---:|---:|---:|
| responsive (80/320/2000 ms, 1) | 1016 ms | 2667 ms | 124 ms | 4093 ms | 69 | 24.705x |
| production (160/480/4000 ms, 2) | 693 ms | 1171 ms | 154 ms | 345 ms | 34 | 18.472x |
| conservative (320/640/4000 ms, 3) | 1024 ms | 1975 ms | 303 ms | 273 ms | 17 | 10.040x |

All three final transcripts were exactly equal:

> Refactor the authentication middleware and add tests for expired tokens,
> malformed tokens, and missing tokens.

The responsive preset could not keep up with real-time capture on this run. Its
preview backlog moved the first stable text later and left about 4.1 seconds of
work after release despite its smaller window. The production preset stayed
near its configured cadence and finalized in 345 ms. The conservative preset
used about half as many calls and finalized in 273 ms, but delayed visible text
because of its larger chunk and stability of three.

Raw hypotheses showed 72, 69, and 98 character edits for responsive,
production, and conservative respectively. No measured transition retracted
characters past its prior common prefix. Stable visible snapshots had no
retractions either. Stability does not guarantee that an interim equals the
final result: the responsive run's last visible interim contained `experired`,
and the conservative run ended its visible preview at `Malform tokens, and`.

## Limits

This is a screening benchmark, not a statistical comparison. It uses one short,
clean WAV, one run per preset, and fixed run order. Model warm-up, MPS thermal
state, and contention can affect timing. Final equality is expected here because
Kestrel performs an exact full-buffer replay at EOF; it does not prove interim
quality on other speech. Run repeated or randomized trials before changing any
production setting.
