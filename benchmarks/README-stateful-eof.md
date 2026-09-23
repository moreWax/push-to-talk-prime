# Stateful EOF finalization experiment

This benchmark compares Kestrel 0.8's current exact full-block replay at live
PCM EOF with an experimental stateful tail decode. It is isolated from the
worker. It monkey-patches Kestrel only inside the benchmark process and restores
the function after each run. It does not change production defaults or files in
`.venv`.

Run from the repository root:

```bash
.venv/bin/python benchmarks/benchmark-stateful-eof.py \
  --device mps --rounds 3 --output /tmp/stateful-eof.json
```

The default input is the actual demo recording, `assets/demo-input.wav`. A separate async producer feeds audio at real-time speed in 20 ms frames, so preview inference cannot stretch capture time. `release_to_final_ms` starts after
the final source-audio frame is yielded and includes processing of the configured 320 ms synthetic tail and final decoding. The tail is queued immediately, as in `Recorder.finish()`; this is not a 320 ms endpointer wait. Each round runs the stock `exact_replay` first and
the experimental `stateful_eof` second. `exact_transcript_equality` uses exact
normalized string equality, including case and punctuation.

Use `--no-realtime` only for quick mechanical checks. It changes buffering and
is not representative of release latency.

## Private APIs used

The candidate replaces `kestrel.models.parakeet_tdt.longform._run_live_pcm` and
uses `_leaf_prompt`, `_LIVE_*` window constants, and
`kestrel.models.parakeet_tdt.runtime._StreamWindow`, `LiveAudioBuffer` buffer/snapshot internals, and the undocumented `EngineResult.output["_stream_state"]` transport. These APIs are private.
Kestrel may rename them or change their contracts without notice.

Stock Kestrel 0.8 discards preview decoder state at EOF and runs the complete
buffered block through its exact non-streaming path. The candidate keeps the
preview `_StreamState`, supplies bounded left context, starts at the first
uncommitted sample, and lets the runtime decode through EOF. The experiment is
limited to input below Kestrel's first stock commit threshold (180 seconds plus its 0.5 second minimum tail) so it
does not imply correctness for long-form block boundaries.

## Risks and interpretation

- Stateful tail decoding can differ from exact replay because it preserves
  decisions made with bounded preview context. Equality on one WAV is evidence,
  not a general correctness proof.
- The exact path may use pause-aligned segmentation, while `_StreamWindow`
  bypasses that segmentation. Results can differ on silence, repeated speech,
  or difficult boundaries.
- Private decoder state may retain device tensors and has no compatibility
  guarantee across Kestrel versions.
- Run order is fixed (exact then stateful). Thermal state, MPS synchronization,
  caches, and model warm-up affect timing. Use multiple rounds and treat the
  latency delta as indicative, not a controlled mode-only effect.
- The candidate is a narrow `timestamps="none"`, single-block experiment. It
  omits stock result accumulation, segment shifting, metrics aggregation, and
  progress fields, so it is not a drop-in implementation.
- A mismatch exits with status 2 and should block any production experiment.

## Demo WAV result (Apple MPS)

One real-time run in the project environment produced:

| mode | release to final | normalized transcript |
|---|---:|---|
| Kestrel 0.8 exact replay | 386.03 ms | `Refactor the authentication middleware and add tests for expired tokens, malformed tokens, and missing tokens.` |
| experimental stateful EOF | 261.32 ms | `Refactor the authentication middleware and add tests for expired tokens.` |

The candidate was 124.71 ms faster, but exact equality was **false**. It lost
the final two test cases. This result rejects this stateful finalizer as a
production replacement despite its lower latency. Re-run results can vary with
hardware and load.
