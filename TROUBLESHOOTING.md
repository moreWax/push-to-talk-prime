# Troubleshooting

Run these first from the package directory:

```bash
npm run doctor
npm run doctor:model
```

Remove secrets, usernames, audio, and transcript content before sharing logs.

## Hold Space types spaces but does not record

- Make sure voice is enabled with `/voice status`.
- Enable keyboard repeat in the operating system.
- Hold through the repeat threshold instead of tapping.
- Restart the full Prime client after package updates; `/reload` does not replace client-side decoration.
- In tmux, enable extended keys/CSI-u for physical key release reporting.

## Release is delayed

The terminal may not report key release. The worker then uses its repeat-gap fallback. Enable Kitty/CSI-u keyboard events where supported.

## No microphone is detected

- Grant microphone permission to the terminal application.
- Run `npm run doctor` and choose a listed `PTT_INPUT_DEVICE` index or name.
- Linux: install PortAudio and verify PipeWire/PulseAudio exposes an input.

```bash
sudo apt install libportaudio2   # Debian/Ubuntu
sudo dnf install portaudio       # Fedora
sudo pacman -S portaudio         # Arch
```

## First use is slow or fails offline

The 178 MB model downloads during first warm-up. Run `npm run doctor:model` once while online. The full Python/Torch environment requires several GiB beyond model size.

## GPU, MPS, or CUDA fails

Use CPU to isolate accelerator problems:

```text
/voice device cpu
```

Then run `npm run doctor:model`. Restore automatic selection with `/voice device auto`.

## Provisional text is unstable

Use a more stable preset:

```text
/voice preset balanced
# or
/voice preset smooth
```

The final exact decode remains authoritative. `PTT_INTERIM_STABILITY` controls how many matching snapshots are required before provisional words appear.

## High memory with multiple Prime clients

The package uses one authenticated singleton model service shared by all clients. If multiple `ptt_worker.py` processes remain after all clients close, update to the latest version and remove stale processes only after confirming no recording is active.

Disable and unload the singleton globally with `/voice`. Run `/voice` again to start and prewarm it.

## Changes do not appear after update

Restart the entire Prime client. Worker `/reload` cannot replace JavaScript prototype decoration already loaded in the terminal process.

## Collect diagnostics

Temporarily launch with:

```bash
PTT_DEBUG_LOG=/tmp/ptt-debug.jsonl prime-agent
```

The file is created as mode `0600` and records transition metadata, not prompt text or raw key bytes. Inspect and redact it before sharing.

Include project commit/version, Prime/pi version, OS/architecture, terminal/tmux, `/voice status`, and sanitized doctor output in bug reports.
