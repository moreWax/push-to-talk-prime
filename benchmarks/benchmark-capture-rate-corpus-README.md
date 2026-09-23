# Native 16 kHz capture corpus validation

This benchmark compares two real-time async input paths with the production
Kestrel streaming settings (`160/480/4000 ms`, stability 2):

1. the existing 16 kHz WAV samples fed directly at 16 kHz; and
2. the same samples upsampled to 48 kHz with a 193-tap, linear-phase
   Kaiser-windowed sinc, then resampled by Kestrel `LiveAudioBuffer` on every
   snapshot.

It makes no production changes. It reused and SHA-256 checked all 12 WAVs in
`/tmp/ptt-stream-corpus` (46.445 seconds). Input-path order alternated by
utterance. One warm model was shared. `/tmp/ptt-mps-benchmark.lock` covered
model loading and all inference.

## Run

```bash
.venv/bin/python benchmarks/benchmark-capture-rate-corpus.py   --output benchmarks/benchmark-capture-rate-corpus-results.json
```

The producer emits 20 ms frames on a real-time clock and queues the production
320 ms silence tail at release. The result JSON includes every raw preview,
every stability-filtered preview, final transcript, decoder input size,
latency, and timed `LiveAudioBuffer.append`/`snapshot` call.

## Results

Environment: macOS 15.5 arm64, Kestrel 0.8.0, `kestrel-kernels` 0.7.0, and
torch 2.14.0 on Apple MPS.

| metric | native 16 kHz | 48 kHz then Kestrel resample |
|---|---:|---:|
| first raw preview, median | 690.05 ms | 691.78 ms |
| first stable preview, median | 1007.63 ms | 929.05 ms |
| release to final, median | 323.27 ms | 328.95 ms |
| `LiveAudioBuffer.snapshot`, total | 13.82 ms | 377.32 ms |
| `LiveAudioBuffer.append`, total | 124.86 ms | 132.10 ms |
| decoder calls | 285 | 285 |
| decoded input | 769.805 s | 769.805 s |
| compute amplification | 15.308953x | 15.308953x |
| raw reference-exact finals | 5/12 | 5/12 |
| normalized reference-exact finals | 9/12 | 8/12 |
| normalized WER | 8.108% | 9.009% |
| normalized CER | 7.961% | 8.155% |

The 48 kHz path added **363.50 ms** of snapshot work across 285 calls, or
**1.275 ms per snapshot**. Snapshot time was 27.3x the native-path total.
The offline high-quality upsampling used to construct the synthetic 48 kHz
capture inputs cost another 75.23 ms in total. That construction cost would
not exist for real hardware capture, but Kestrel's per-snapshot downsampling
cost would.

The latency differences are small compared with normal single-pass inference
variation. Paired median 48 kHz-minus-16 kHz differences were +1.25 ms for the
first raw preview, -3.00 ms for the first stable preview, and -9.98 ms for
release-to-final. This one alternating pass is not a confidence interval and
does not show a latency benefit for either path.

## Transcript parity and quality

Final transcripts matched exactly on 11/12 utterances. Complete raw-preview
sequences matched on 11/12, and complete stable-preview sequences matched on
11/12.

Across all ordinal previews:

- raw text matched on 253/273 previews (92.67%);
- stability-filtered text matched on 104/110 previews (94.55%).

The `numbers` utterance caused all preview-sequence differences, although both
paths reached the same final text. The only final difference was
`paused-command`:

- native 16 kHz: `Build the release binary. Then wait for the checksum.`
- 48 kHz round trip: `Build the release binary. Then wait for the checksom.`

Thus the extra sample-rate conversion was not transcript-neutral. It introduced
one final word error in this corpus. Both paths performed identical decoder
work, so 48 kHz did not reduce model amplification.

## Device probe

`sounddevice.check_input_settings(device=0, channels=1, dtype="float32",
samplerate=16000)` succeeded for the current `MacBook Pro Microphone`. Its
reported default rate is 48 kHz. This proves PortAudio accepts a requested
16 kHz stream on this device. It does not prove the hardware ADC itself runs
at 16 kHz rather than Core Audio converting before delivery.

## Recommendation

**Request native 16 kHz capture on the current device.** It is accepted by the
input API, preserves or slightly improves transcript quality in this corpus,
and avoids Kestrel's repeated 48-to-16 kHz snapshot resampling. Keep a fallback
to a supported device rate if a different device rejects 16 kHz.

The evidence is isolated and deterministic, but it is still one clean,
synthetic 12-utterance pass. Re-run on real microphone speech and other input
devices before treating the quality delta as universal.
