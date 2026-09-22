#!/usr/bin/env python3
"""Render the scripted README demo as MP4 and GIF."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
WIDTH, HEIGHT = 960, 540
FPS, DURATION = 20, 15.0
FONT_PATH = Path("/System/Library/Fonts/Menlo.ttc")
if not FONT_PATH.exists():
    FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT = ImageFont.truetype(str(FONT_PATH), 22)
SMALL = ImageFont.truetype(str(FONT_PATH), 15)
TITLE = ImageFont.truetype(str(FONT_PATH), 30)
BG, PANEL, BORDER = "#11111b", "#181825", "#45475a"
TEXT, DIM, GREEN, PURPLE, YELLOW = "#cdd6f4", "#7f849c", "#a6e3a1", "#cba6f7", "#f9e2af"


def reveal(text: str, progress: float) -> str:
    return text[: max(0, min(len(text), int(len(text) * progress)))]


def state_at(t: float) -> tuple[list[tuple[str, str]], str, str]:
    lines: list[tuple[str, str]] = []
    footer = "hold Space to speak"
    badge = "LOCAL · parakeet-redux"
    if t < 1.2:
        return [("Push to Talk for Prime Agent", PURPLE), ("Hold Space · speak · release", DIM)], "", badge
    lines.append(("$ prime-agent", GREEN))
    if t < 2.0:
        return lines, "", badge
    base = "Update the auth middleware"
    if t < 3.5:
        prompt = reveal(base + " ", (t - 2.0) / 1.2)
    elif t < 4.2:
        prompt = base + " " + "▁▂▃▄▅▆▇█"[int((t * 10) % 8)]
        footer = "recording locally"
    elif t < 4.9:
        prompt = base + " with"
        footer = "stable live transcript"
    elif t < 5.6:
        prompt = base + " with the new"
        footer = "stable live transcript"
    elif t < 6.3:
        prompt = base + " with the new token validation"
        footer = "stable live transcript"
    elif t < 7.1:
        prompt = base + " with the new token validation helper"
        footer = "stable live transcript"
    elif t < 8.2:
        prompt = "Update the auth middleware with the new token-validation helper."
        footer = "exact final correction"
    elif t < 9.5:
        lines.append(("> /voice status", TEXT))
        lines.append(("Voice: hold · Preset: balanced · Device: gpu (mps)", GREEN))
        prompt = ""
        footer = "one command namespace"
    elif t < 10.8:
        lines.append(("> /voice preset realtime", TEXT))
        lines.append(("Preset changed · singleton worker restarting", YELLOW))
        prompt = ""
        footer = "named presets · no numeric tuning"
    elif t < 12.0:
        lines.append(("> /voice device gpu", TEXT))
        lines.append(("Device: gpu (mps) · model prewarmed", GREEN))
        prompt = ""
        footer = "auto maps MPS on macOS · CUDA elsewhere"
    else:
        lines.extend([
            ("Session A ─┐", PURPLE),
            ("Session B ─┼── 127.0.0.1 authenticated broker", PURPLE),
            ("Session C ─┘              │", PURPLE),
            ("                         └── one warm Parakeet model", GREEN),
        ])
        prompt = ""
        footer = "one service shared across every Prime session"
    return lines, prompt, "LOCAL · " + footer


def render_frame(t: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((38, 34, WIDTH - 38, HEIGHT - 34), radius=15, fill=PANEL, outline=BORDER, width=2)
    draw.ellipse((59, 54, 73, 68), fill="#f38ba8")
    draw.ellipse((82, 54, 96, 68), fill="#f9e2af")
    draw.ellipse((105, 54, 119, 68), fill="#a6e3a1")
    draw.text((140, 51), "prime-agent · push-to-talk", font=SMALL, fill=DIM)
    draw.text((WIDTH - 60, 51), "SCRIPTED DEMO", font=SMALL, fill=YELLOW, anchor="ra")
    draw.line((58, 82, WIDTH - 58, 82), fill=BORDER, width=1)

    lines, prompt, footer = state_at(t)
    y = 116
    if t < 1.2:
        draw.text((WIDTH // 2, 195), lines[0][0], font=TITLE, fill=lines[0][1], anchor="mm")
        draw.text((WIDTH // 2, 242), lines[1][0], font=FONT, fill=lines[1][1], anchor="mm")
    else:
        for text, color in lines:
            draw.text((72, y), text, font=FONT, fill=color)
            y += 38
        prompt_y = HEIGHT - 135
        draw.rounded_rectangle((58, prompt_y - 14, WIDTH - 58, prompt_y + 44), radius=8, fill="#1e1e2e")
        draw.text((76, prompt_y), "> ", font=FONT, fill=PURPLE)
        draw.text((104, prompt_y), prompt, font=FONT, fill=TEXT)
        cursor_x = 104 + draw.textlength(prompt, font=FONT)
        if int(t * 2) % 2 == 0:
            draw.rectangle((cursor_x, prompt_y + 1, cursor_x + 3, prompt_y + 25), fill=TEXT)

    draw.text((65, HEIGHT - 61), footer, font=SMALL, fill=DIM)
    draw.text((WIDTH - 65, HEIGHT - 61), "audio stays local", font=SMALL, fill=GREEN, anchor="ra")
    # progress
    draw.rectangle((58, HEIGHT - 39, WIDTH - 58, HEIGHT - 35), fill="#313244")
    draw.rectangle((58, HEIGHT - 39, 58 + (WIDTH - 116) * min(1, t / DURATION), HEIGHT - 35), fill=PURPLE)
    return image


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    ASSETS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ptt-demo-") as temp:
        frames = Path(temp) / "frames"
        frames.mkdir()
        for index in range(round(FPS * DURATION)):
            render_frame(index / FPS).save(frames / f"{index:05d}.png")
        silent = Path(temp) / "silent.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent)])

        voice = Path(temp) / "voice.aiff"
        if shutil.which("say"):
            run(["say", "-v", "Samantha", "-r", "165", "-o", str(voice), "with the new token validation helper"])
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(voice),
                 "-filter_complex", "[1:a]adelay=3500|3500,apad=pad_dur=15[a]", "-map", "0:v", "-map", "[a]",
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(ASSETS / "demo.mp4")])
        else:
            shutil.copy2(silent, ASSETS / "demo.mp4")

        palette = Path(temp) / "palette.png"
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-vf", "fps=10,scale=800:-1:flags=lanczos,palettegen=max_colors=96", str(palette)])
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(palette),
             "-lavfi", "fps=10,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer", str(ASSETS / "demo.gif")])
    print(ASSETS / "demo.mp4")
    print(ASSETS / "demo.gif")


if __name__ == "__main__":
    main()
