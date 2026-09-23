# Opt-in MPS Metal experiment: Parakeet residual + LayerNorm

This experiment is isolated from the worker and Kestrel. It compiles a shader
at run time with `torch.mps.compile_shader`; it does not monkey-patch a runtime,
change an installed package, or alter a production default.

## Why this operation

Kestrel 0.8 calls `add_scaled_layer_norm` four times in each of the 24
FastConformer blocks. The operation must return both:

```text
total = residual + alpha * update
normalized = layer_norm(total, weight, bias, eps=1e-5)
```

On MPS, Kestrel Kernels 0.7 inherits this entry point from its CPU conformer
runtime. MPS tensors fail the CPU native-kernel eligibility check and use two
torch/MPSGraph operations. The experiment uses one 256-thread Metal threadgroup
per `[1024]` Parakeet row. It writes the fp16 residual sum, reduces LayerNorm
statistics in fp32, and writes the fp16 normalized result in the same launch.
The fixed width and fp16 checks are intentional. This is a Parakeet hot-path
probe, not a general LayerNorm implementation.

This is the highest-value remaining simple fusion found in the Apple path:

- Kestrel already fuses relative attention and depthwise-convolution +
  BatchNorm + SiLU into native Metal launches. Its source says those two kernels
  replace 78% of the old encoder dispatches. Reimplementing either with
  `compile_shader` would be redundant.
- Kestrel already has a fused MPS TDT dual-argmax, reducing two host waits to
  one.
- Kestrel documents that a fused predictor LSTM elementwise kernel saved work
  but was rejected because Metal sigmoid/tanh differences compounded through
  the autoregressive state and changed transcripts.
- Kestrel measured standalone Metal LayerNorm at parity with MPSGraph. Fusing
  the preceding residual add changes that tradeoff by removing one launch and
  one intermediate read while still producing the residual needed by the next
  sublayer.

The candidate is structurally safer than the LSTM experiment because it has no
recurrent state and exactly reproduces the fp16 residual output. Its normalized
output is not bit-exact due to a different fp32 reduction order. Full-model
transcript tests are still required before any production integration.

## Run

Use the repository environment without modifying it:

```bash
.venv/bin/python benchmarks/benchmark-mps-metal-layernorm.py
```

Useful options:

```bash
.venv/bin/python benchmarks/benchmark-mps-metal-layernorm.py \
  --rows 48 80 128 224 --iterations 500 --repeats 5 \
  --output /tmp/mps-metal-layernorm.json
```

The script requires MPS, fp16, a 32-lane Apple SIMD group, and
`torch.mps.compile_shader`. It atomically acquires
`/tmp/ptt-mps-benchmark.lock` with `mkdir` and removes it in `finally`. It
refuses to run when another benchmark owns the lock.

`rows` flattens batch and time. The defaults mirror Kestrel's small encoder
bucket sizes. Timing queues many calls, synchronizes once per sample, and
reports the median of samples. `metal_allocating_us` includes the two output
allocations, as the eager torch expression does. `metal_preallocated_us` shows
the launch when a caller can reuse output storage.

## Serialized result

Measured on an Apple M1 MacBook Pro with 16 GB RAM, macOS 15.5,
`torch 2.14.0`, `kestrel 0.8.0`, and `kestrel-kernels 0.7.0`. The final run used
`--iterations 300 --warmup 20 --repeats 5` and held the atomic lock for the
whole run.

| rows | Metal, allocating | torch eager | speedup | Metal, preallocated |
|---:|---:|---:|---:|---:|
| 48 | 10.58 us | 21.89 us | 2.07x | 8.31 us |
| 80 | 12.10 us | 25.93 us | 2.14x | 11.25 us |
| 128 | 16.08 us | 29.73 us | 1.85x | 14.31 us |
| 224 | 20.97 us | 37.36 us | 1.78x | 19.15 us |

Across both Parakeet alphas (`0.5` and `1.0`) and all four shapes:

- The returned fp16 residual was bit-exact (`max_abs = 0`).
- At least 99.976% of normalized fp16 values were bit-exact in every case.
- Worst normalized mean absolute error was `6.33e-8`.
- Worst normalized maximum absolute error was `0.00390625`.
- All outputs were finite.

The isolated microbenchmark suggests about 1.8-2.1x lower latency for this
operation including allocation. It does **not** claim the same speedup for the
whole encoder. At 96 calls per encoder, the per-call medians imply only a rough
1-2 ms upper-bound opportunity for these bucket sizes, before integration and
full-model effects.

An early exploratory timing run happened before the lock requirement was
communicated and might have overlapped other MPS work. Those measurements were
discarded. Only the serialized final run is reported above.

## Production caveats

Do not wire this shader into the worker based only on this result. A production
candidate needs at least:

1. transcript and encoder-output regression tests on representative audio;
2. testing with actual model activation distributions, not only random tensors;
3. supported-shape and dtype fallback behavior matching Kestrel's runtime;
4. lifetime, stream-ordering, and output-allocation review; and
5. measurements on newer Apple GPU families.

The experiment deliberately remains opt-in so none of those open questions can
change current transcription behavior.
