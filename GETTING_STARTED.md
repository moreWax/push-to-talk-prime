# Getting Started

This guide gets local voice dictation working in Prime Agent in about ten minutes.

## 1. Check compatibility

You need:

- Prime Agent 0.9.5 or a compatible pi client
- Node.js 22.19 or newer
- `uv` 0.8 or newer
- a supported 64-bit platform listed in the [README](README.md#requirements)
- a microphone with OS permission
- several GiB of free disk for Torch, native kernels, and model caches

Linux also needs PortAudio. See [Troubleshooting](TROUBLESHOOTING.md#no-microphone-is-detected).

## 2. Install

```bash
prime-agent package install https://github.com/moreWax/push-to-talk-prime
```

The first voice warm-up prepares the locked Python environment automatically. To run setup and diagnostics yourself, clone manually:

```bash
git clone https://github.com/moreWax/push-to-talk-prime.git
cd push-to-talk-prime
npm run setup
prime-agent package install "$PWD"
```

The model downloads on first warm-up. To prefetch and validate it while online:

```bash
npm run doctor:model
```

## 3. Start a new Prime client

Client-side editor decoration loads only at process startup. Close and reopen Prime; `/reload` alone is not sufficient after installing or updating the package.

```bash
prime-agent
```

## 4. Dictate

1. Hold Space until the level block appears.
2. Speak while continuing to hold Space.
3. Release Space.
4. Stable words appear progressively; the exact final transcript replaces them after release.
5. Press Enter yourself to submit.

A quick Space tap remains ordinary typing. Press Escape or type another key to cancel active dictation safely.

## 5. Configure

```text
/voice                 toggle voice and its warm singleton service
/voice status          inspect state, preset, and device
/voice preset balanced choose the recommended preset
/voice device auto     choose the best available compute backend
```

Available presets: `fast`, `balanced`, `realtime`, and `smooth`.

Available devices: `auto`, `cpu`, `gpu`, `mps`, and `cuda`. `gpu` maps to MPS on macOS and CUDA elsewhere.

## Next steps

- [Configuration and compatibility](README.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Architecture](ARCHITECTURE.md)
- [Privacy and third-party notices](THIRD_PARTY_NOTICES.md)
- [Updating and rollback](README.md#updating-and-rollback)
