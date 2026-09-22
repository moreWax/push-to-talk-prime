# Contributing

Thanks for improving push-to-talk-prime. Please read the [Code of Conduct](CODE_OF_CONDUCT.md), [Security Policy](SECURITY.md), and [Support guide](SUPPORT.md).

## Prerequisites

- Node.js 22.19+
- `uv`
- Python 3.11 selected by uv
- a supported OS/architecture from the README
- PortAudio on Linux

## Setup

```bash
git clone https://github.com/moreWax/push-to-talk-prime.git
cd push-to-talk-prime
npm ci
uv sync --locked --python 3.11
npm test
```

Optional hardware checks:

```bash
npm run doctor
npm run doctor:model
```

## Repository layout

- `extensions/` — Prime/pi editor integration and IPC clients
- `scripts/voice-server.mjs` — authenticated singleton broker
- `ptt_worker.py` — microphone, Photon model, streaming, and stabilization
- `scripts/` — setup, platform checks, and diagnostics
- `tests/` — Python worker tests
- `tests-ts/` — editor and singleton-service tests

## Workflow

1. Open an issue before large design changes.
2. Create a focused branch and commits.
3. Add regression tests for changed behavior.
4. Run `npm test` and locked setup checks.
5. Update README, architecture, troubleshooting, and changelog for user-visible changes.
6. Open a pull request using the template.

Do not commit model weights, recordings, transcripts, settings, debug logs, `.venv`, or `node_modules`. Do not regenerate lockfiles without dependency changes. State tested OS, terminal, Prime/pi version, and hardware in the pull request.

Contributions are submitted under the MIT License. Do not add dependencies or assets with incompatible terms.
