#!/usr/bin/env python3
"""Render the video-only hold/release/final timing HUD as transparent PNG frames."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def event_times(path: Path) -> tuple[float, float, float]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    press = next(item["time"] for item in events if item.get("event") == "editor_bridge_press")
    commit = next(item["time"] for item in events if item.get("event") == "editor_bridge_commit" and item["time"] >= press)
    release = next(item["time"] for item in events if item.get("event") == "editor_bridge_release" and item["time"] >= commit)
    final = next(item["time"] for item in events if item.get("event") == "editor_bridge_final" and item["time"] >= release)
    return (commit - press) / 1000, (release - press) / 1000, (final - press) / 1000


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/SFNSMono.ttf"),
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size, index=1 if bold and candidate.suffix == ".ttc" else 0)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--press-at", type=float, default=1.0)
    args = parser.parse_args()

    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(args.video),
    ], text=True).strip())
    commit_after, release_after, final_after = event_times(args.events)
    shutil.rmtree(args.output, ignore_errors=True)
    args.output.mkdir(parents=True)

    regular = font(15)
    strong = font(16, bold=True)
    muted = (166, 173, 200, 255)
    coral = (217, 119, 87, 255)
    amber = (255, 194, 141, 255)
    panel = (24, 24, 37, 224)
    border = (88, 91, 112, 210)

    for index in range(math.ceil(duration * args.fps)):
        now = index / args.fps
        image = Image.new("RGBA", (390, 96), (0, 0, 0, 0))
        if now >= args.press_at:
            elapsed = now - args.press_at
            if elapsed < commit_after:
                state, color, key = "HOLDING", muted, "SPACE  ▼"
                shown = elapsed
            elif elapsed < release_after:
                state, color, key = "RECORDING", coral, "SPACE  ▼"
                shown = elapsed
            elif elapsed < final_after:
                state, color, key = "FINALIZING", amber, "SPACE  ▲"
                shown = elapsed
            else:
                state, color, key = "FINAL", (166, 227, 161, 255), "SPACE  ✓"
                shown = final_after
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((1, 1, 388, 94), radius=14, fill=panel, outline=border, width=1)
            draw.ellipse((18, 17, 28, 27), fill=color)
            draw.text((38, 12), state, font=strong, fill=color)
            millis = round(shown * 1000)
            draw.text((250, 12), f"{millis:04d} ms", font=regular, fill=muted)
            pressed = state == "HOLDING"
            top = 49 if pressed else 45
            draw.rounded_rectangle((18, top, 139, top + 31), radius=7, fill=(49, 50, 68, 255), outline=color, width=2)
            draw.text((31, top + 6), key, font=regular, fill=(238, 241, 250, 255))
            start_x, end_x = 164, 365
            draw.rounded_rectangle((start_x, 57, end_x, 63), radius=3, fill=(65, 67, 86, 255))
            progress = min(elapsed / max(final_after, 0.001), 1)
            draw.rounded_rectangle((start_x, 57, start_x + max(6, int((end_x - start_x) * progress)), 63), radius=3, fill=color)
        image.save(args.output / f"{index:05d}.png")

    print(json.dumps({
        "commit_ms": round(commit_after * 1000),
        "release_ms": round(release_after * 1000),
        "final_ms": round(final_after * 1000),
        "duration": duration,
    }))


if __name__ == "__main__":
    main()
