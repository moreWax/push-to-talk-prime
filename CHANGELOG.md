# Changelog

## 0.1.0

- Added local hold-Space and tap speech input for Prime Agent and compatible pi clients.
- Added daemon-backed Prime editor integration, progressive Parakeet transcription, worker lifecycle controls, diagnostics, and cross-platform packaging.
- Added a single authenticated warm speech service shared across daemon-backed Prime sessions.
- Added production documentation, community health files, and singleton integration coverage.
- Added a reproducible actual Prime PTY demo using real MPS transcription and generated source audio.
- Added model-native 16 kHz capture with device-native fallback and corpus-backed diagnostics.
- Added Claude-style raw/stable streaming: stable words remain normal while the speculative suffix is dimmed.
- Added an opt-in `speculative` preset with one serialized 480 ms early draft while preserving the `balanced` authoritative baseline.
