"""Tải các model chạy offline về models/ (tổng ~1.6GB). Chạy một lần.

| thành phần | nguồn | dung lượng |
|---|---|---|
| bộ não (function calling) | DuongCodeAI/vi-fc-qwen3-1.7b (GGUF Q4_K_M) | ~1.1GB |
| STT | DuongCodeAI/phowhisper-small-noisy-ct2-int8 (hoặc tự convert vinai/PhoWhisper-small) | ~250MB |
| TTS | rhasspy/piper-voices vi_VN-vais1000-medium | ~63MB |
| mặt tài xế | MediaPipe face_landmarker.task | ~4MB |
| biển báo | DuongCodeAI/vn-dashcam-vision (ONNX) | ~10MB |
| thêm dấu | DuongCodeAI/vi-diacritics-tagger (ONNX int8) | ~6MB |

python scripts/download_models.py --only tts face
"""

import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

M = Path("models")
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
            "face_landmarker.task")


def get(name: str):
    if name == "brain":
        snapshot_download("DuongCodeAI/vi-fc-qwen3-1.7b", allow_patterns=["*Q4_K_M.gguf"], local_dir=M)
    elif name == "stt":
        snapshot_download("DuongCodeAI/phowhisper-small-noisy-ct2-int8", local_dir=M / "phowhisper-small-ct2-int8")
    elif name == "tts":
        for ext in ("onnx", "onnx.json"):
            p = hf_hub_download("rhasspy/piper-voices", f"vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.{ext}")
            (M / "piper").mkdir(parents=True, exist_ok=True)
            (M / "piper" / Path(p).name).write_bytes(Path(p).read_bytes())
    elif name == "face":
        M.mkdir(exist_ok=True)
        urllib.request.urlretrieve(FACE_URL, M / "face_landmarker.task")
    elif name == "dashcam":
        snapshot_download("DuongCodeAI/vn-dashcam-vision", allow_patterns=["*.onnx", "*.json"], local_dir=M / "dashcam")
    elif name == "diacritics":
        snapshot_download("DuongCodeAI/vi-diacritics-tagger", allow_patterns=["*.onnx", "*.json"],
                          local_dir=M / "diacritics")
    print("xong", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["brain", "stt", "tts", "face", "dashcam", "diacritics"])
    for n in ap.parse_args().only:
        get(n)
