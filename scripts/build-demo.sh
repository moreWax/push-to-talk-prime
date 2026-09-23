#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VHS_BIN=${VHS_BIN:-vhs}
command -v "$VHS_BIN" >/dev/null || { echo "VHS v0.11.0 is required" >&2; exit 1; }
version=$($VHS_BIN --version)
[[ "$version" == *"v0.11.0"* ]] || { echo "VHS v0.11.0 is required (v0.12.0 has a no-output regression)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
DEMO_STATE=/tmp/ptt-vhs-v2-service.json
stop_demo_brokers() {
  pkill -f "voice-server.mjs --state $DEMO_STATE" 2>/dev/null || true
  rm -f "$DEMO_STATE" "$DEMO_STATE.lock"
}
stop_demo_brokers
cleanup() {
  stop_demo_brokers
  rm -rf /tmp/ptt-demo-hud /tmp/ptt-demo-overlay.mp4
}
trap cleanup EXIT
"$VHS_BIN" demo.tape
timing=$(uv run --project . --group demo --python 3.11 python scripts/render-demo-hud.py \
  --video assets/vhs-polished-demo-silent.webm \
  --events /tmp/ptt-vhs-events.jsonl \
  --output /tmp/ptt-demo-hud \
  --fps 20)
printf '%s\n' "$timing"
audio_delay=$(python3 -c 'import json,sys; print(1000 + json.loads(sys.argv[1])["commit_ms"])' "$timing")
ffmpeg -y -loglevel error \
  -i assets/vhs-polished-demo-silent.webm \
  -framerate 20 -i /tmp/ptt-demo-hud/%05d.png \
  -filter_complex "[0:v][1:v]overlay=W-w-28:28:shortest=1[v]" \
  -map '[v]' -c:v libx264 -pix_fmt yuv420p /tmp/ptt-demo-overlay.mp4
ffmpeg -y -loglevel error \
  -i /tmp/ptt-demo-overlay.mp4 \
  -filter_complex "[0:v]fps=20,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" \
  assets/vhs-polished-demo.gif
ffmpeg -y -loglevel error \
  -i /tmp/ptt-demo-overlay.mp4 \
  -i assets/demo-input.wav \
  -filter_complex "[1:a]adelay=${audio_delay}|${audio_delay},apad[a]" \
  -map 0:v -map '[a]' -c:v copy -c:a aac -shortest \
  assets/vhs-polished-demo.mp4
rm -f assets/vhs-polished-demo-silent.webm
printf '%s
' assets/vhs-polished-demo.gif assets/vhs-polished-demo.mp4
