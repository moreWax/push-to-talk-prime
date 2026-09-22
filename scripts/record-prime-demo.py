#!/usr/bin/env python3
"""Record the real Prime TUI while the extension transcribes a WAV source."""
from __future__ import annotations
import fcntl, json, os, pty, select, shutil, signal, struct, subprocess, tempfile, termios, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pyte

ROOT=Path(__file__).resolve().parent.parent; ASSETS=ROOT/"assets"; COLS,ROWS=100,30; WIDTH,HEIGHT,FPS=1000,540,15
FONT_PATH=Path("/System/Library/Fonts/Menlo.ttc")
if not FONT_PATH.exists():FONT_PATH=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT=ImageFont.truetype(str(FONT_PATH),15); BG="#11111b"; FG="#cdd6f4"; CURSOR="#f5e0dc"

def render(screen):
    image=Image.new("RGB",(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(image);line_height=17;x0,y0=20,15
    for row,line in enumerate(screen.display):
        draw.text((x0,y0+row*line_height),line,font=FONT,fill=FG)
    if screen.cursor and not screen.cursor.hidden:
        x=x0+screen.cursor.x*draw.textlength("M",font=FONT);y=y0+screen.cursor.y*line_height
        draw.rectangle((x,y,x+2,y+line_height-2),fill=CURSOR)
    return image

def stop_existing_service():
    state=Path.home()/".prime/agent/push-to-talk-server.json"
    try:
        pid=json.loads(state.read_text())["pid"];os.kill(int(pid),signal.SIGTERM);time.sleep(1)
    except Exception:pass

def main():
    if not shutil.which("ffmpeg") or not shutil.which("prime-agent"):raise SystemExit("prime-agent and ffmpeg are required")
    stop_existing_service();ASSETS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ptt-real-tui-") as temp:
        temp=Path(temp);settings=temp/"settings.json";state=temp/"service.json"
        settings.write_text(json.dumps({"enabled":True,"mode":"hold","autoSubmit":False,"preset":"balanced","device":"mps"}))
        debug=Path("/tmp/ptt-real-demo-debug.jsonl");debug.unlink(missing_ok=True)
        env={**os.environ,"PTT_CONFIG":str(settings),"PTT_SERVICE_STATE":str(state),"PTT_DEMO_AUDIO_FILE":str(ASSETS/"demo-input.wav"),"PTT_DEBUG_LOG":str(debug)}
        pid,fd=pty.fork()
        if pid==0:
            os.chdir(ROOT);os.execvpe("prime-agent",["prime-agent","--no-session","--offline","-ne","-e",str(ROOT/"extensions/push-to-talk.ts")],env)
        fcntl.ioctl(fd,termios.TIOCSWINSZ,struct.pack("HHHH",ROWS,COLS,0,0));os.set_blocking(fd,False)
        screen=pyte.Screen(COLS,ROWS);stream=pyte.Stream(screen)
        # Let Prime and the singleton warm before starting the captured timeline.
        warm_deadline=time.monotonic()+12
        while time.monotonic()<warm_deadline:
            ready,_,_=select.select([fd],[],[],0.05)
            if ready:
                try:stream.feed(os.read(fd,65536).decode("utf8","ignore"))
                except OSError:break
            if any("gpt-" in line.lower() or "model" in line.lower() for line in screen.display) and time.monotonic()>warm_deadline-6:break
        video=temp/"silent.mp4";ffmpeg=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),"-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(video)],stdin=subprocess.PIPE)
        started=time.monotonic();duration=12.5;hold_start=1.5;hold_end=hold_start+4.1;next_space=hold_start
        try:
            while time.monotonic()-started<duration:
                now=time.monotonic()-started
                while next_space<=hold_end and now>=next_space:
                    os.write(fd,b" ");next_space+=0.04
                ready,_,_=select.select([fd],[],[],0)
                if ready:
                    try:stream.feed(os.read(fd,65536).decode("utf8","ignore"))
                    except OSError:break
                assert ffmpeg.stdin is not None;ffmpeg.stdin.write(render(screen).tobytes())
                target=started+(round(now*FPS)+1)/FPS;time.sleep(max(0,target-time.monotonic()))
        finally:
            if ffmpeg.stdin:ffmpeg.stdin.close()
            ffmpeg.wait(timeout=30)
            try:os.write(fd,b"\x03\x03")
            except OSError:pass
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:os.waitpid(pid,0)
            except ChildProcessError:pass
            try:
                service_pid=json.loads(state.read_text())["pid"];os.kill(int(service_pid),signal.SIGTERM)
            except Exception:pass
            service_log=Path(str(state)+".log")
            if service_log.exists(): shutil.copy2(service_log,"/tmp/ptt-real-demo-service.log")
        delay=round((hold_start+0.16)*1000)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(video),"-i",str(ASSETS/"demo-input.wav"),"-filter_complex",f"[1:a]adelay={delay}|{delay},apad=pad_dur={duration}[a]","-map","0:v","-map","[a]","-c:v","copy","-c:a","aac","-shortest",str(ASSETS/"demo.mp4")],check=True)
        palette=temp/"palette.png";subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(video),"-vf","fps=10,scale=800:-1:flags=lanczos,palettegen=max_colors=96",str(palette)],check=True);subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(video),"-i",str(palette),"-lavfi","fps=10,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",str(ASSETS/"demo.gif")],check=True)
    print(ASSETS/"demo.mp4");print(ASSETS/"demo.gif")
if __name__=="__main__":main()
