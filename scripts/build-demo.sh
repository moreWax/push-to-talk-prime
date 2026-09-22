#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VHS_BIN=${VHS_BIN:-vhs}
command -v "$VHS_BIN" >/dev/null || { echo "VHS v0.11.0 is required" >&2; exit 1; }
version=$($VHS_BIN --version)
[[ "$version" == *"v0.11.0"* ]] || { echo "VHS v0.11.0 is required (v0.12.0 has a no-output regression)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
cleanup() {
  if [[ -f /tmp/ptt-vhs-service.json ]]; then
    pid=$(python3 -c 'import json; print(json.load(open("/tmp/ptt-vhs-service.json"))["pid"])' 2>/dev/null || true)
    [[ -z "${pid:-}" ]] || kill "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
"$VHS_BIN" demo.tape
ffmpeg -y -loglevel error   -i assets/vhs-polished-demo-silent.webm   -i assets/demo-input.wav   -filter_complex "[1:a]adelay=1160|1160,apad[a]"   -map 0:v -map '[a]' -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest   assets/vhs-polished-demo.mp4
rm -f assets/vhs-polished-demo-silent.webm
printf '%s
' assets/vhs-polished-demo.gif assets/vhs-polished-demo.mp4
