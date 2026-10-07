"""Sinh dữ liệu cho web demo (docs/, GitHub Pages) bằng chính pipeline chạy trên laptop.

Web demo chạy thật trong trình duyệt 2 phần: nhận diện biển báo (2 model ONNX int8 của vn-dashcam-vision)
và theo dõi buồn ngủ (MediaPipe + EAR/PERCLOS viết lại bằng JS). Hai phần cần model lớn thì lấy kết quả thật
từ script này:
- câu cảnh báo cho 52 mã biển x 2 loại xe: LawRAG.lookup_sign + sign_warning, đọc bằng Piper
- lệnh giọng nói: wav -> PhoWhisper -> Qwen3-1.7B GGUF fine-tune -> guard -> xe giả lập -> Piper,
  ghi lại đúng output và latency từng bước (model 1 GB không chạy nổi trên trình duyệt)

python scripts/build_web_demo.py          # cần models/ (download_models.py) + dữ liệu luật bên vn-traffic-law-rag
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, "src")
from copilot.cli import build, load_cfg  # noqa: E402
from copilot.copilot import Copilot  # noqa: E402
from copilot.drowsiness import DrowsyState  # noqa: E402
from copilot.fines import parse_fine, sign_warning  # noqa: E402
from copilot.vehicle import VehicleSim  # noqa: E402

VNTS = "thanhhiepvos/vietnam_traffic_sign"  # bản sao VNTS (CC BY-SA) trên HF, cùng file với bản Kaggle
# ảnh trong tập TEST của VNTS (split_dataset/test_files.txt), model chưa thấy lúc train.
# Chọn cho đủ loại biển; giữ cả ảnh model sai để demo không chỉ toàn ca đẹp.
SAMPLES = ["0832", "2611", "0660", "1683", "0112", "0809", "2859", "1965"]

# lệnh gõ thêm để thấy guard và tra luật; (câu, tốc độ km/h, ban đêm)
TYPED = [
    ("mo khoa cua giup minh", 40, False),
    ("mở khoá cửa giúp mình", 0, False),
    ("tắt đèn đi", 60, True),
    ("mở hết cửa sổ ra", 100, False),
    ("xe máy vượt đèn đỏ phạt bao nhiêu", 40, False),
]


def to_mp3(audio: np.ndarray, sr: int, out: Path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    sf.write(tmp, audio, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-b:a", "48k", str(out)], check=True)
    os.unlink(tmp)


class Speaker:
    """Piper -> mp3, câu trùng nhau chỉ tổng hợp một lần (tên file theo hash của câu)."""

    def __init__(self, tts, folder: Path):
        self.tts, self.folder, self.done = tts, folder, {}

    def __call__(self, text: str) -> tuple[str, float]:
        if text not in self.done:
            audio, ms = self.tts.synth(text)
            name = f"say_{hashlib.md5(text.encode()).hexdigest()[:10]}.mp3"
            to_mp3(audio, self.tts.sr, self.folder / name)
            self.done[text] = (f"audio/{name}", round(ms))
        return self.done[text]


def build_signs(law, names: list[str], display: list[str], speak: Speaker) -> list[dict]:
    out = []
    for code, disp in zip(names, display, strict=True):
        row = {"code": code, "name": disp, "warn": {}}
        for kind in ("ô tô", "xe máy"):
            info = law.lookup_sign(code, kind, 3)
            if info is None:  # biển chỉ dẫn / không có trong bảng tra -> trợ lý im lặng
                row["warn"][kind] = None
                continue
            text = sign_warning(info.name, kind, info.hits, speak_fine=getattr(info, "speak_fine", True))
            main = next((h for h in info.hits if parse_fine(h.chunk["text"]) and not h.expanded_from), None)
            audio, ms = speak(text)
            row["warn"][kind] = {"text": text, "audio": audio, "tts_ms": ms,
                                 "cite": main.citation if main and info.speak_fine else None,
                                 "law": main.chunk["text"][:600] if main and info.speak_fine else None}
        out.append(row)
    return out


async def run_command(parts, text: str, speed: float, night: bool, speak: Speaker) -> dict:
    said = []
    v = VehicleSim(speed_kmh=speed, is_night=night)
    cp = Copilot(None, v, parts["brain"], parts.get("law"), say=said.append, restorer=parts.get("restorer"))
    turn = await cp.on_speech(text)
    audio, tts_ms = speak(said[0])
    return {"input": text, "model_input": turn.input, "speed": speed, "night": night, "action": turn.action,
            "reply": said[0],
            "audio_out": audio, "ms": {k: round(x) for k, x in turn.ms.items()} | {"tts": tts_ms}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--no-voice", action="store_true", help="giữ data/voice.json cũ, không chạy STT + LLM")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    out = Path(args.out)
    for d in ("model", "data", "audio", "samples"):
        (out / d).mkdir(parents=True, exist_ok=True)

    # model chạy trong trình duyệt: bản int8 (detector 3.2 MB, SignNet 1.2 MB) + face landmarker
    dash = Path(cfg["dashcam_models"])
    for f in ("detector_1cls.int8.onnx", "signnet.int8.onnx", "names.json"):
        shutil.copy(dash / f, out / "model" / f)
    shutil.copy(cfg["face"], out / "model" / "face_landmarker.task")

    from huggingface_hub import hf_hub_download

    def vnts(path):
        return hf_hub_download(VNTS, f"archive/{path}", repo_type="dataset")

    names = json.loads((dash / "names.json").read_text("utf-8"))
    display = [s.strip() for s in Path(vnts("classes_vie.txt")).read_text("utf-8").splitlines() if s.strip()]
    samples = []
    for s in SAMPLES:
        shutil.copy(vnts(f"images/{s}.jpg"), out / "samples" / f"{s}.jpg")
        rows = [ln.split() for ln in Path(vnts(f"labels/{s}.txt")).read_text().splitlines() if ln.strip()]
        samples.append({"file": f"samples/{s}.jpg", "gt": [{"code": names[int(r[0])],
                                                            "box": [float(x) for x in r[1:5]]} for r in rows]})
    (out / "data" / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), "utf-8")

    from copilot.speech import STT, TTS

    parts = build(cfg, need_brain=not args.no_voice)
    speak = Speaker(TTS(cfg["tts"]), out / "audio")
    t0 = time.perf_counter()
    signs = build_signs(parts["law"], names, display, speak)
    (out / "data" / "signs.json").write_text(json.dumps(signs, ensure_ascii=False), "utf-8")
    print(f"biển báo: {len(signs)} mã, {time.perf_counter() - t0:.0f}s")

    # câu trợ lý nói khi phát hiện buồn ngủ: lấy đúng từ Copilot.on_drowsy
    drowsy = {}
    for level in ("warn", "alarm"):
        said = []
        asyncio.run(Copilot(None, VehicleSim(speed_kmh=60), say=said.append).on_drowsy(DrowsyState(level, "")))
        drowsy[level] = {"text": said[0], "audio": speak(said[0])[0]}
    (out / "data" / "drowsy.json").write_text(json.dumps(drowsy, ensure_ascii=False), "utf-8")
    if args.no_voice:
        return

    stt = STT(cfg["stt"], threads=cfg["threads"])
    stt(np.zeros(16000, np.float32))  # lần đầu nạp model chậm hơn hẳn, không tính
    cmds = []
    for i, it in enumerate(yaml.safe_load(open("demo/script_voice.yaml", encoding="utf-8")), 1):
        audio, sr = sf.read(it["wav"], dtype="float32")
        tr = stt(audio)
        r = asyncio.run(run_command(parts, tr.text, 40, False, speak))
        name = f"audio/cmd_{i:02d}.mp3"
        to_mp3(audio, sr, out / name)
        r |= {"said": it["text"], "audio_in": name, "ms": {"stt": round(tr.ms)} | r["ms"]}
        cmds.append(r)
        print(r["said"], "->", r["input"], "->", r["reply"], r["ms"])
    for text, speed, night in TYPED:
        r = asyncio.run(run_command(parts, text, speed, night, speak))
        cmds.append(r)
        print(text, "->", r["reply"], r["action"]["kind"], r["ms"])
    (out / "data" / "voice.json").write_text(json.dumps(cmds, ensure_ascii=False, indent=1), "utf-8")
    spoken = [c for c in cmds if "audio_in" in c]
    p50 = {k: float(np.median([c["ms"].get(k, 0) for c in spoken])) / 1000 for k in ("stt", "brain")}
    total = float(np.median([sum(c["ms"].values()) for c in spoken])) / 1000
    def sec(x):
        return f"{x:.1f}".replace(".", ",")

    note = (f"Chạy ngày {time.strftime('%d/%m/%Y')} trên laptop CPU ({cfg['threads']} luồng). Trung vị lệnh nói: "
            f"STT {sec(p50['stt'])} s, LLM {sec(p50['brain'])} s, cả lệnh {sec(total)} s, chưa đạt mục tiêu 1,5 s. "
            "Wav lệnh đọc bằng giọng Piper (giọng tổng hợp) nên STT nghe sai câu không dấu và tên bài hát nhiều hơn "
            "giọng người thật.")
    (out / "data" / "voice_meta.json").write_text(json.dumps({"note": note}, ensure_ascii=False), "utf-8")
    print("xong:", out)


if __name__ == "__main__":
    main()
