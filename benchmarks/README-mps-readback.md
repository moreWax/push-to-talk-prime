# Kestrel 0.8 Apple MPS streaming readback experiment

`benchmark-mps-readback.py` is an isolated prototype for the Parakeet TDT live
preview path used by this project. It makes no worker or installed-package
changes. Its monkey patches exist only in the benchmark process and are restored
in `finally` blocks.

## Run

From the repository root:

For the 12-prompt stream corpus:

```sh
.venv/bin/python benchmarks/benchmark-mps-readback.py \
  --corpus /tmp/ptt-stream-corpus \
  --output benchmarks/benchmark-mps-readback-results.json
```

For one WAV, pass its path as the positional argument. With no input, the script
uses `assets/demo-input.wav`. Audio is supplied as 160 ms live PCM chunks. The
script requires Apple silicon MPS and exactly:

- Kestrel 0.8.0
- Kestrel Kernels 0.7.0
- torch 2.14.0

It atomically acquires `/tmp/ptt-mps-benchmark.lock` before MPS work and fails
rather than sharing MPS with another cooperating benchmark. The lock records the
PID and script path. If a killed process leaves the lock behind, check that PID
before deleting it.

## Candidate

Stock Kestrel performs these Python-visible device reads for every live preview:

1. `int(valid.sum())` in `model.py:673`;
2. one fused native token/duration decision read for every greedy step;
3. `int(generated.lengths[0])` in `runtime.py:511`;
4. separate token and duration `tolist()` calls in `runtime.py:512-513`.

The candidate copies the Kestrel 0.8 single-row decode loop into the benchmark
without changing the joint network, predictor LSTM, greedy decision kernel,
blank handling, durations, carry, or decoder state. It changes only the data
boundary:

- It derives `valid_length` from the host NumPy sample count. Feature extraction
  marks `samples // 160` mel frames valid. Every stride-2 subsampler convolution
  then computes `ceil(length / 2)`. Kestrel's `_encoder_frames` applies that
  exact integer recurrence, so no `valid.sum()` scalar read is needed.
- It retains cumulative token and duration tensors on MPS in the stream state.
- At each UI preview boundary it stacks both tensors and performs one `tolist()`.
  There is no generated-length scalar read and no separate token/duration read.
- Kestrel's exact full-buffer replay at EOF remains stock. Preview state is not
  used for the final result in Kestrel 0.8.

The benchmark requires exact equality of every preview transcript, final
transcript, cumulative token ID list, and cumulative duration list. It repeats
that parity gate once with diagnostics enabled and once with diagnostics off.
Any mismatch fails the command.

## 12-prompt corpus result

The checked report is `benchmark-mps-readback-results.json`. It used all 12
prompts in `/tmp/ptt-stream-corpus`. The first mode alternated by prompt:
stock/candidate, then candidate/stock. Each prompt had one diagnostic run and
one probe-off timing repeat in the same order.

All 12 prompts passed literal equality for:

- the complete ordered preview transcript sequence;
- the final transcript;
- every cumulative token ID snapshot;
- every cumulative duration snapshot.

The probe-off repeat passed the same gates. There were zero failures.

| Metric | Stock | Candidate | Delta |
|---|---:|---:|---:|
| Preview decode p50 (249 previews) | 59.794 ms | 59.573 ms | -0.221 ms (-0.4%) |
| Preview decode p95 | 69.632 ms | 67.919 ms | -1.713 ms (-2.5%) |
| Release-to-final p50 (12 prompts) | 102.087 ms | 102.055 ms | -0.032 ms (-0.0%) |
| Release-to-final p95 | 193.824 ms | 200.626 ms | +6.802 ms (+3.5%) |
| End-to-end p50 (12 prompts) | 1,182.827 ms | 1,177.947 ms | -4.880 ms (-0.4%) |
| End-to-end p95 | 3,251.685 ms | 3,197.990 ms | -53.695 ms (-1.7%) |
| Python-visible MPS materializations | 1,822 | 1,075 | -747 (-41.0%) |
| Explicit `torch.mps.synchronize()` calls | 0 | 0 | 0 |
| Native TDT decision readbacks | 409 | 409 | 0 |

The latency result is modest and should not be read as a broad confidence
interval. The materialization reduction is structural: three fewer host reads
per preview, or 747 fewer across 249 previews.

Latency is measured with the traceback probe removed. `release_final_ms` starts
when the input generator yields the final audio chunk. The diagnostic run alone
supplies materialization counts. Close unrelated GPU workloads for better
control.

## What “sync count” means

The report counts Python-visible MPS tensor materializations and explicit
`torch.mps.synchronize()` calls. These are host-wait boundaries, not proof of
one physical Metal synchronization each. Work can already be complete when a
boundary is reached. Waits inside native code are not generally observable by
Python monkey patching.

For this exact MPS path, Kestrel's native `tdt_greedy_argmax_into` writes a
reused two-element device tensor and immediately calls `tolist()`. Those calls
are visible to the probe. There were 108 across each complete run: 55 live
preview decisions and 53 decisions in the unchanged final replay. Use Instruments
with Metal System Trace if an exact driver/native synchronization count is
required.

## Larger Metal greedy loop feasibility

A larger native loop cannot simply wrap the existing TDT decision kernel. That
kernel accepts already-computed logits and returns only one token ID and one
duration index. Python still must use the duration to select the next encoder
frame and use a non-blank token to run the predictor LSTM before computing the
next joint output. These dependencies prevent batching future decisions.

Removing the remaining per-step host read requires a new native decoder that
owns all of the following for the whole loop:

- frame, carry, token budget, and step budget;
- joint activation and output projection;
- blank/duration branching;
- predictor LSTM hidden and cell state;
- a device output buffer and final length.

That is a new TDT runtime, not a larger dispatch around
`tdt_greedy_argmax_into`. It also crosses the exact-parity boundary. Kestrel's
source notes that a fused Metal predictor LSTM was rejected because Metal
sigmoid/tanh differences compounded through recurrence. Keeping predictor math
in PyTorch/MPSGraph preserves current math but returns control to Python on each
conditional token and therefore does not remove the wait.

Conclusion: the host-length and packed UI materialization change is feasible
and exact. A whole-loop Metal decoder is technically possible only by
reimplementing the predictor/joint control path and proving recurrent bit-level
token parity over a broad corpus. It is high effort and high regression risk;
the existing native decision kernel alone is not a sufficient building block.

## Scope and limitations

- Private Kestrel 0.8 APIs and source-equivalent logic are used only in this
  process. Any production change would need version guards and tests.
- The result covers 12 prompts with alternating first mode. Run more rounds
  before treating the small latency delta as a stable win.
- The candidate still creates per-preview output tensors from the existing
  Python decision lists because the greedy loop itself remains host-controlled.
  It then keeps the cumulative tensors on device until UI emission.
- No fallback claim is made by this script. Use `benchmark-mps-fallback.py` with
  `PYTORCH_ENABLE_MPS_FALLBACK=0` for that separate question.
