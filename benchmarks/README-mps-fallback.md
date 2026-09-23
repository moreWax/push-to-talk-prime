# MPS fallback and host-wait probe

Run from the repository root with the checked-in environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python benchmarks/benchmark-mps-fallback.py \
  --output /tmp/ptt-mps-fallback.json
```

The script holds `/tmp/ptt-mps-benchmark.lock` by atomic exclusive creation for
the complete model run. It makes no package or production source changes. If the
lock remains after an interrupted process, confirm that its recorded PID is no
longer running before removing it.

A completed transcription is evidence that no unsupported MPS operation used
PyTorch's CPU fallback on the tested input and shape. It is not proof for all
inputs. The probe also counts Python-visible MPS tensor materializations
(`tolist`, `item`, `cpu`, scalar conversion, and similar calls) plus explicit
`torch.mps.synchronize()` calls. Native extension host waits are not intercepted;
the reported Kestrel MPS TDT `Tensor.tolist` sites are the Python boundary around
its native Metal result readback.
