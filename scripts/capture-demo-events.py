#!/usr/bin/env python3
"""Capture real Parakeet streaming events for the product demo."""
from __future__ import annotations
import argparse, asyncio, json, os, time, wave
from pathlib import Path
import numpy as np
from ptt_worker import InterimTranscriptStabilizer, load_model

ROOT=Path(__file__).resolve().parent.parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--audio",default=str(ROOT/"assets/demo-input.wav"))
    parser.add_argument("--output",default=str(ROOT/"assets/demo-events.json"))
    parser.add_argument("--device",default="mps" if os.uname().sysname=="Darwin" else "cpu")
    parser.add_argument("--chunk-ms",type=float,default=160)
    parser.add_argument("--right-ms",type=float,default=480)
    parser.add_argument("--left-ms",type=float,default=4000)
    args=parser.parse_args()
    os.environ.update(PTT_STREAM_CHUNK_MS=str(args.chunk_ms),PTT_STREAM_RIGHT_MS=str(args.right_ms),PTT_STREAM_LEFT_MS=str(args.left_ms))
    with wave.open(args.audio,"rb") as source:
        rate=source.getframerate(); channels=source.getnchannels(); width=source.getsampwidth(); raw=source.readframes(source.getnframes())
    if channels!=1 or width!=2: raise SystemExit("demo audio must be mono 16-bit PCM")
    pcm=np.frombuffer(raw,dtype="<i2").astype(np.float32)/32768.0
    frames=max(1,round(rate*args.chunk_ms/1000))
    timing={"start":None}
    async def audio():
        loop=asyncio.get_running_loop(); timing["start"]=loop.time()
        for index,offset in enumerate(range(0,len(pcm),frames)):
            target=timing["start"]+index*frames/rate
            delay=target-loop.time()
            if delay>0: await asyncio.sleep(delay)
            yield pcm[offset:offset+frames]
    photon,speech=load_model(args.device,announce=False)
    events=[]; stabilizer=InterimTranscriptStabilizer(2)
    started=time.perf_counter()
    try:
        stream=speech.transcribe(audio=audio(),sample_rate=rate,timestamps="none",stream=True)
        for update in stream:
            if update.get("provisional") is False: continue
            stable=stabilizer.push(str(update.get("text","")))
            if stable is not None: events.append({"time":round(time.perf_counter()-started,3),"type":"interim","text":stable})
        result=stream.result(); final=str(result.get("text","")).strip()
        events.append({"time":round(time.perf_counter()-started,3),"type":"final","text":final})
    finally:
        photon.__exit__(None,None,None)
    payload={"source_audio":str(Path(args.audio).relative_to(ROOT)),"device":args.device,"chunk_ms":args.chunk_ms,"right_ms":args.right_ms,"left_ms":args.left_ms,"audio_duration":round(len(pcm)/rate,3),"events":events}
    Path(args.output).write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
