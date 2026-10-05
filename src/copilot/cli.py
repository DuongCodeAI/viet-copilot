"""copilot chat   - gõ lệnh bằng chữ (có thể không dấu), không cần mic/camera
copilot replay --dashcam drive.mp4 --cabin face.mp4 --script demo/script.yaml   - quay demo
copilot live --dashcam drive.mp4 --cabin 0   - mic + webcam thật

Thành phần nào thiếu model thì tự tắt và báo, để vẫn chạy thử được từng phần.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from .bus import EventBus
from .copilot import Copilot
from .vehicle import VehicleSim


def load_cfg(path: str) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


def build(cfg: dict, need_brain: bool = True):
    parts = {}
    try:
        from vnlaw_rag.pipeline import LawRAG

        os.environ.setdefault("VNLAW_MODELS", cfg["law_models"])
        parts["law"] = LawRAG.from_dir(cfg["law_data"], models_dir=cfg["law_models"], threads=cfg["threads"])
    except Exception as e:  # noqa: BLE001
        print("[tắt] tra luật:", e)
    try:
        from vi_diacritics import Restorer

        parts["restorer"] = Restorer.load(cfg["diacritics"], min_conf=0.5).restore
    except Exception as e:  # noqa: BLE001
        print("[tắt] thêm dấu:", e)
    if need_brain:
        from vi_fc.inference import FunctionCaller

        b = cfg["brain"]
        if b["backend"] == "llama_cpp":
            if not os.path.exists(b["gguf"]):
                raise SystemExit(f"chưa có model gọi tool {b['gguf']}: chạy "
                                 "`python scripts/download_models.py --only brain` hoặc đổi brain.backend sang "
                                 "openai (Groq, cần GROQ_API_KEY) trong configs/default.yaml")
            brain = FunctionCaller("llama_cpp", b["gguf"], n_threads=cfg["threads"])
        else:  # Groq hoặc llama.cpp server, dùng khi máy quá yếu / chưa có GGUF
            brain = FunctionCaller("openai", b["model"], base_url=b["base_url"], api_key=os.getenv("GROQ_API_KEY"))
        # lượt đầu của llama.cpp phải xử lý cả system prompt + 17 tool (~5k ký tự); làm trước cho lệnh đầu không chậm
        print(f"warmup bộ não: {brain.warmup():.1f}s")
        parts["brain"] = brain
    return parts


def cmd_chat(args, cfg):
    parts = build(cfg)
    bus = EventBus()
    cp = Copilot(bus, VehicleSim(speed_kmh=args.speed), parts["brain"], parts.get("law"),
                 vehicle_kind=cfg["vehicle_kind"], restorer=parts.get("restorer"))
    print("Gõ lệnh (Enter trống để thoát). Tốc độ xe:", args.speed, "km/h")
    while True:
        text = input("> ").strip()
        if not text:
            break
        turn = asyncio.run(cp.on_speech(text))
        print("   ", {k: round(v) for k, v in turn.ms.items()}, "ms")


async def _replay(args, cfg):
    from .sources import cabin_source, dashcam_source, script_source

    parts = build(cfg)
    stt = tts = None
    if args.script and any("wav" in it for it in yaml.safe_load(open(args.script, encoding="utf-8"))):
        from .speech import STT

        stt = STT(cfg["stt"], threads=cfg["threads"])
    if args.speak:
        from .speech import TTS

        tts = TTS(cfg["tts"])
    bus, stats = EventBus(), {}
    spoken = []

    async def say(text):
        spoken.append((round(time.monotonic() - t0, 2), text))
        print(f"[{spoken[-1][0]:6.1f}s] trợ lý: {text}")
        if tts:
            audio, ms = await asyncio.to_thread(tts.synth, text)
            stats.setdefault("tts_ms", []).append(ms)
            import sounddevice as sd

            await asyncio.to_thread(lambda: (sd.play(audio, tts.sr), sd.wait()))

    cp = Copilot(bus, VehicleSim(speed_kmh=args.speed), parts["brain"], parts.get("law"), say=say,
                 vehicle_kind=cfg["vehicle_kind"], restorer=parts.get("restorer"))
    stop = asyncio.Event()
    t0 = time.monotonic()
    print("đang tra trước luật cho các biển báo...")
    await cp.warm_sign_cache()
    runner = asyncio.create_task(cp.run(stop))
    tasks = []
    if args.dashcam:
        tasks.append(dashcam_source(bus, args.dashcam, cfg["dashcam_models"], stats=stats))
    if args.cabin:
        tasks.append(cabin_source(bus, args.cabin, cfg["face"], stats=stats))
    if args.script:
        tasks.append(script_source(bus, args.script, stt, stats=stats))
    await asyncio.gather(*tasks)
    await asyncio.sleep(3)  # chờ trợ lý nói nốt
    stop.set()
    await runner

    out = Path("logs")
    out.mkdir(exist_ok=True)
    summary = {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n": len(v)}
               for k, v in stats.items() if v}
    for t in cp.turns:
        for k, v in t.ms.items():
            summary.setdefault(f"speech_{k}", {"vals": []})["vals"].append(v)
    for k, v in summary.items():
        if "vals" in v:
            summary[k] = {"p50": float(np.percentile(v["vals"], 50)), "p95": float(np.percentile(v["vals"], 95)),
                          "n": len(v["vals"])}
    json.dump({"spoken": spoken, "turns": [t.__dict__ for t in cp.turns], "latency_ms": summary},
              open(out / "replay.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="copilot")
    ap.add_argument("--config", default="configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--speed", type=float, default=40)
    r = sub.add_parser("replay")
    r.add_argument("--dashcam")
    r.add_argument("--cabin")
    r.add_argument("--script")
    r.add_argument("--speed", type=float, default=40)
    r.add_argument("--speak", action="store_true", help="đọc thành tiếng bằng Piper")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.cmd == "chat":
        cmd_chat(args, cfg)
    else:
        asyncio.run(_replay(args, cfg))


if __name__ == "__main__":
    main()
