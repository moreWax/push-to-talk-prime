# Apple MPS fixed-shape `torch.compile` encoder experiment

This is an opt-in benchmark for Kestrel 0.8 Parakeet on Apple silicon. It does
not change `ptt_worker.py`, production defaults, or installed `.venv` files. The
script applies a process-local patch and restores it before exit.

Run it from the repository root:

```bash
.venv/bin/python benchmarks/benchmark-mps-compile.py \
  --eager-rounds 3 --hot-rounds 3 \
  --output /tmp/mps-compile.json
```

The default input is `assets/demo-input.wav`. The script always requests MPS.
It rejects any environment except macOS arm64 with MPS and these exact package
versions:

- `kestrel==0.8.0`
- `kestrel-kernels==0.7.0`
- `torch==2.14.0`

These guards are intentional. The experiment calls private Kestrel fields and
Torch compiler diagnostics whose behavior is not a stable API.

## Candidate path

The eager baseline is Kestrel's stock MPS encoder path. A process-local patch
to `ParakeetTdt.encode` selects the benchmark mode. The candidate keeps the
subsampler eager, pads its output to a fixed encoder-frame bucket, and calls
`torch.compile(model.encode_subsampled, backend="inductor", dynamic=False)`.
It then slices the padded result back to the logical length. The default buckets
are Kestrel 0.8's encoder graph buckets: `48 80 128 224`. Override them with, for
example, `--buckets 48 80 128 224 384`.

A segment larger than the last bucket runs eagerly and is counted as
`over_max_bucket`. A compiler exception also runs eagerly and is counted as
`compile_error`; the unique error text is saved. Any fallback makes the command
exit with status 2. Use `--fullgraph` to reject graph breaks instead of allowing
Dynamo to split the function into compiled regions.

## Measurements and correctness

The report contains:

- multiple stock eager runs;
- one first-use candidate run as `cold_compile_run`;
- multiple cached candidate runs as `hot_compile_runs`;
- total transcription and synchronized encoder latency for every run;
- exact normalized transcript equality against the final eager run;
- per-call input shape, selected bucket, latency, and fallback;
- Dynamo `frames`, `stats`, `graph_break`, `inductor`, and `aot_autograd`
  counters; and
- path fallback counts and compile errors.

`torch.mps.synchronize()` brackets each encoder call so encoder latency includes
completed device work. The cold run includes compile cost for every shape first
seen in that run. A transcript mismatch or any path fallback returns status 2.
Exact equality on one WAV is only a regression check, not a general proof.

## Benchmark serialization

The script atomically creates `/tmp/ptt-mps-benchmark.lock` with `mkdir` before
model loading and removes it in `finally`. If the directory already exists, the
script reports its owner metadata and exits without benchmarking. This prevents
overlap among cooperating push-to-talk MPS benchmarks. It cannot detect an
unrelated process using MPS, so close other GPU workloads for controlled timing.
A stale lock after an uncatchable process termination must be inspected and
removed manually.

## Interpretation

Compare hot candidate encoder latency with eager encoder latency. Do not compare
only total transcription time: feature extraction, pause segmentation, and TDT
decoding remain eager. Graph-break counters identify compiler partitioning, but
a graph break is not the same as the explicit eager fallbacks listed in
`path_fallback_counts`. Thermal state and system GPU activity can still affect
results, even with the cooperative lock.
