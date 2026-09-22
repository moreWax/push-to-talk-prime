#!/usr/bin/env python3
"""Render a real-model Prime-style demo from captured Parakeet events."""
from __future__ import annotations
import json, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent.parent; ASSETS=ROOT/"assets"
EVENTS=json.loads((ASSETS/"demo-events.json").read_text()); AUDIO=ROOT/EVENTS["source_audio"]
WIDTH,HEIGHT,FPS=960,540,20; AUDIO_START=3.0
FINAL_TIME=AUDIO_START+max(event["time"] for event in EVENTS["events"]); DURATION=max(12.0,FINAL_TIME+3.0)
FONT_PATH=Path("/System/Library/Fonts/Menlo.ttc")
if not FONT_PATH.exists():FONT_PATH=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT=ImageFont.truetype(str(FONT_PATH),22); PROMPT=ImageFont.truetype(str(FONT_PATH),16); SMALL=ImageFont.truetype(str(FONT_PATH),15); TITLE=ImageFont.truetype(str(FONT_PATH),30)
BG,PANEL,BORDER="#11111b","#181825","#45475a";TEXT,DIM,GREEN,PURPLE,YELLOW="#cdd6f4","#7f849c","#a6e3a1","#cba6f7","#f9e2af"

def reveal(text,progress):return text[:max(0,min(len(text),int(len(text)*progress)))]
def latest_event(relative):
    found=None
    for event in EVENTS["events"]:
        if event["time"]<=relative:found=event
    return found

def state_at(t):
    lines=[];footer="hold Space to speak"
    if t<1.2:return [("Push to Talk for Prime Agent",PURPLE),("Real computer voice · real local transcription",DIM)],"",footer
    lines.append(("$ prime-agent",GREEN))
    if t<2.1:return lines,"",footer
    if t<AUDIO_START:
        return lines,reveal("Voice note: ",(t-2.1)/(AUDIO_START-2.1)),footer
    relative=t-AUDIO_START;event=latest_event(relative)
    if event is None:
        return lines,"Voice note: "+"▁▂▃▄▅▆▇█"[int(t*10)%8],"real MPS audio stream"
    text="Voice note: "+event["text"]
    footer="exact final correction" if event["type"]=="final" else "stable live model output"
    if t<FINAL_TIME+0.8:return lines,text,footer
    if t<FINAL_TIME+1.8:
        lines.append(("> /voice status",TEXT));lines.append((f"Voice: hold · Preset: balanced · Device: gpu (mps)",GREEN));return lines,"","one shared warm service"
    lines.extend([("Session A ─┐",PURPLE),("Session B ─┼── authenticated singleton",PURPLE),("Session C ─┘              │",PURPLE),("                         └── one warm Parakeet model",GREEN)])
    return lines,"","one model shared by every Prime session"

def render(t):
    image=Image.new("RGB",(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((38,34,WIDTH-38,HEIGHT-34),radius=15,fill=PANEL,outline=BORDER,width=2)
    for x,c in [(59,"#f38ba8"),(82,"#f9e2af"),(105,"#a6e3a1")]:draw.ellipse((x,54,x+14,68),fill=c)
    draw.text((140,51),"prime-agent · push-to-talk",font=SMALL,fill=DIM);draw.text((WIDTH-60,51),"REAL MODEL DEMO",font=SMALL,fill=YELLOW,anchor="ra");draw.line((58,82,WIDTH-58,82),fill=BORDER)
    lines,prompt,footer=state_at(t);y=116
    if t<1.2:
        draw.text((WIDTH//2,195),lines[0][0],font=TITLE,fill=lines[0][1],anchor="mm");draw.text((WIDTH//2,242),lines[1][0],font=FONT,fill=lines[1][1],anchor="mm")
    else:
        for text,color in lines:draw.text((72,y),text,font=FONT,fill=color);y+=38
        py=HEIGHT-135;draw.rounded_rectangle((58,py-14,WIDTH-58,py+44),radius=8,fill="#1e1e2e");draw.text((76,py),"> ",font=FONT,fill=PURPLE);draw.text((104,py+3),prompt,font=PROMPT,fill=TEXT)
        cx=104+draw.textlength(prompt,font=PROMPT)
        if int(t*2)%2==0:draw.rectangle((cx,py+3,cx+3,py+25),fill=TEXT)
    draw.text((65,HEIGHT-61),f"LIVE · {footer}",font=SMALL,fill=DIM);draw.text((WIDTH-65,HEIGHT-61),"audio stays local",font=SMALL,fill=GREEN,anchor="ra")
    draw.rectangle((58,HEIGHT-39,WIDTH-58,HEIGHT-35),fill="#313244");draw.rectangle((58,HEIGHT-39,58+(WIDTH-116)*min(1,t/DURATION),HEIGHT-35),fill=PURPLE)
    return image

def run(args):subprocess.run(args,check=True)
def main():
    if not shutil.which("ffmpeg"):raise SystemExit("ffmpeg is required")
    ASSETS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ptt-demo-") as tmp:
        frames=Path(tmp)/"frames";frames.mkdir()
        for index in range(round(FPS*DURATION)):render(index/FPS).save(frames/f"{index:05d}.png")
        silent=Path(tmp)/"silent.mp4";run(["ffmpeg","-y","-loglevel","error","-framerate",str(FPS),"-i",str(frames/"%05d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(silent)])
        delay=round(AUDIO_START*1000);run(["ffmpeg","-y","-loglevel","error","-i",str(silent),"-i",str(AUDIO),"-filter_complex",f"[1:a]adelay={delay}|{delay},apad=pad_dur={DURATION}[a]","-map","0:v","-map","[a]","-c:v","copy","-c:a","aac","-shortest",str(ASSETS/"demo.mp4")])
        palette=Path(tmp)/"palette.png";run(["ffmpeg","-y","-loglevel","error","-i",str(silent),"-vf","fps=10,scale=800:-1:flags=lanczos,palettegen=max_colors=96",str(palette)]);run(["ffmpeg","-y","-loglevel","error","-i",str(silent),"-i",str(palette),"-lavfi","fps=10,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",str(ASSETS/"demo.gif")])
    print(json.dumps({"mp4":str(ASSETS/"demo.mp4"),"gif":str(ASSETS/"demo.gif"),"events":len(EVENTS["events"]),"final":EVENTS["events"][-1]["text"]},indent=2))
if __name__=="__main__":main()
