# Inference optimization benchmarks

These isolated harnesses record the Apple/MPS inference investigation. They do not change production defaults or installed dependencies. MPS jobs use `/tmp/ptt-mps-benchmark.lock` to avoid overlapping measurements.

## Accepted production change

| Change | Evidence | Decision |
|---|---|---|
| Prefer native 16 kHz capture with device-native fallback | 12-prompt corpus: WER 8.108% at 16 kHz vs 9.009% after a 48 kHz roundtrip; 27.3× less `LiveAudioBuffer` snapshot work; current microphone supports 16 kHz | Implemented |

## Retained defaults and rejected experiments

| Experiment | Result | Decision |
|---|---|---|
| Stateful EOF finalization | 124.7 ms faster, but dropped two final clauses | Rejected |
| Whole-encoder `torch.compile` | Hot encoder 222 ms vs 121 ms eager; 46 graph breaks | Rejected |
| Compile islands | 108.11 ms vs 108.95 ms eager | Rejected |
| Fused Metal residual + LayerNorm | Exact/token parity passed; about 1.1% total speedup | Not worth production complexity |
| Reduced MPS host readbacks | 12/12 corpus parity; 41% fewer Python materializations; 4.9 ms median end-to-end gain and no release median gain | Not worth private Kestrel patch |
| 160/320/1000 streaming context | 44% less amplification and earlier median interim, but worse preview-tail completeness | Rejected |
| 160/320/2000 streaming context | 21.7% less amplification, but preview completion fell from 36.2% to 18.1% | Rejected |
| Dense Core ML conversion | Would expand packed ternary projections and defeat the required ternary execution path | Rejected before artifact creation |

## Environment

The recorded Apple runs used macOS arm64, PyTorch 2.14.0, Kestrel 0.8.0, Kestrel Kernels 0.7.0, and the cached `moondream/parakeet-redux` checkpoint. Most timing runs use `assets/demo-input.wav`; corpus runs generate deterministic WAV files under `/tmp` and store recipes and hashes in their result JSON.

Run a harness with the project interpreter, for example:

```bash
.venv/bin/python benchmarks/benchmark-capture-path.py
```

Treat single-machine latency as directional. Transcript/token parity and corpus quality gates take priority over microbenchmark speedups.
