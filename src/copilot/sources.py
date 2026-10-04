"""Các nguồn sự kiện chạy song song, đẩy vào EventBus.

- dashcam_source: video (file hoặc camera) -> SignPipeline (vn-dashcam-vision) -> "sign"
- cabin_source:   webcam / video tài xế -> FaceLandmarks + DrowsinessMonitor -> "drowsy"
- mic_source:     micro -> STT -> "speech"
- script_source:  kịch bản câu lệnh có mốc thời gian (để quay demo lặp lại được) -> "speech"

Ở chế độ replay, video được đọc theo đúng tốc độ thật (sleep theo fps) để mô phỏng xe đang chạy;
nếu máy xử lý không kịp thì bỏ frame chứ không chạy chậm lại (giống camera thật).
"""

import asyncio
import time

import yaml

from .bus import EventBus


async def dashcam_source(bus: EventBus, video: str, models_dir: str, every: int = 2, realtime: bool = True,
                         stats: dict | None = None):
    import cv2
    from dashcam.pipeline import SignPipeline

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pipe = SignPipeline.load(models_dir, fps=fps / every)
    t_start, i = time.monotonic(), 0
    while True:
        ok, frame = await asyncio.to_thread(cap.read)
        if not ok:
            break
        i += 1
        if realtime:
            lag = (time.monotonic() - t_start) - i / fps
            if lag > 1 / fps:  # đang chậm hơn thời gian thực -> bỏ frame
                continue
            if lag < 0:
                await asyncio.sleep(-lag)
        if i % every:
            continue
        res = await asyncio.to_thread(pipe.process, frame)
        if stats is not None:
            stats.setdefault("dashcam_ms", []).append(sum(res.ms.values()))
        for ev in res.events:
            bus.publish("sign", ev)
    cap.release()


async def cabin_source(bus: EventBus, source, face_model: str, fps: float = 10, stats: dict | None = None):
    """source: chỉ số webcam (0) hoặc đường dẫn video."""
    import cv2

    from .drowsiness import DrowsinessMonitor, FaceLandmarks

    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / fps)), 1)
    lm, mon = FaceLandmarks(face_model), DrowsinessMonitor(fps=fps)
    i, last = 0, "ok"
    while True:
        ok, frame = await asyncio.to_thread(cap.read)
        if not ok:
            break
        i += 1
        if i % step:
            continue
        t0 = time.perf_counter()
        st = mon.update(await asyncio.to_thread(lm, frame, int(1000 / fps)))
        if stats is not None:
            stats.setdefault("cabin_ms", []).append((time.perf_counter() - t0) * 1e3)
        if st.level in ("warn", "alarm") and st.level != last:
            bus.publish("drowsy", st)
        last = st.level
        if isinstance(source, str):
            await asyncio.sleep(step / src_fps)
    cap.release()


async def mic_source(bus: EventBus, stt, stats: dict | None = None):
    from .speech import energy_vad_record

    while True:
        audio = await asyncio.to_thread(energy_vad_record)
        if len(audio) < 4000:
            continue
        tr = await asyncio.to_thread(stt, audio)
        if stats is not None:
            stats.setdefault("stt_ms", []).append(tr.ms)
        if tr.text:
            bus.publish("speech", tr.text)


async def script_source(bus: EventBus, script_path: str, stt=None, stats: dict | None = None):
    """YAML: - {t: 12.5, text: "bật điều hoà 22 độ"}  hoặc  - {t: 30, wav: data/cmd1.wav} (đi qua STT thật)."""
    items = yaml.safe_load(open(script_path, encoding="utf-8"))
    t0 = time.monotonic()
    for it in sorted(items, key=lambda x: x["t"]):
        await asyncio.sleep(max(0.0, it["t"] - (time.monotonic() - t0)))
        if "wav" in it and stt is not None:
            import soundfile as sf

            audio, sr = sf.read(it["wav"], dtype="float32")
            tr = await asyncio.to_thread(stt, audio)
            text = tr.text
            if stats is not None:
                stats.setdefault("stt_ms", []).append(tr.ms)
        else:
            text = it["text"]
        bus.publish("speech", text)
