"""Tải các model chạy offline về models/ (tổng ~1.6GB). Chạy một lần.

| thành phần | nguồn | dung lượng |
|---|---|---|
| bộ não (function calling) | <HF_USER>/vi-fc-qwen3-1.7b-GGUF (Q4_K_M) | ~1.1GB |
| STT | vinai/PhoWhisper-small, tự đổi sang CTranslate2 int8 (cần torch + transformers, tải ~1GB) | ~240MB |
| TTS | rhasspy/piper-voices vi_VN-vais1000-medium | ~63MB |
| mặt tài xế | MediaPipe face_landmarker.task | ~4MB |
| biển báo | <HF_USER>/vn-dashcam-vision (ONNX) | ~10MB |
| thêm dấu | <HF_USER>/vi-diacritics-tagger (ONNX int8) | ~6MB |

python scripts/download_models.py --only tts face
Tài khoản HF chứa model: biến môi trường HF_USER (mặc định hgdkakhs).
"""

import argparse
import os
import subprocess
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

M = Path("models")
HF_USER = os.environ.get("HF_USER", "hgdkakhs")
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
            "face_landmarker.task")


def get(name: str, stt_size: str = "small"):
    if name == "brain":
        snapshot_download(f"{HF_USER}/vi-fc-qwen3-1.7b-GGUF", allow_patterns=["*Q4_K_M.gguf"], local_dir=M)
    elif name == "stt":
        # bản fine-tune có ồn kém hơn bản gốc (README) -> dùng PhoWhisper gốc;
        # small vì tiny/base WER VIVOS cao gấp ~4 lần
        subprocess.run(["ct2-transformers-converter", "--model", f"vinai/PhoWhisper-{stt_size}", "--output_dir",
                        str(M / f"phowhisper-{stt_size}-ct2-int8"), "--quantization", "int8",
                        "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"], check=True)
    elif name == "tts":
        for ext in ("onnx", "onnx.json"):
            p = hf_hub_download("rhasspy/piper-voices", f"vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.{ext}")
            (M / "piper").mkdir(parents=True, exist_ok=True)
            (M / "piper" / Path(p).name).write_bytes(Path(p).read_bytes())
    elif name == "face":
        M.mkdir(exist_ok=True)
        urllib.request.urlretrieve(FACE_URL, M / "face_landmarker.task")
    elif name == "dashcam":
        snapshot_download(f"{HF_USER}/vn-dashcam-vision", allow_patterns=["*.onnx", "*.json"], local_dir=M / "dashcam")
    elif name == "diacritics":
        snapshot_download(f"{HF_USER}/vi-diacritics-tagger", allow_patterns=["*.onnx", "*.json"],
                          local_dir=M / "diacritics")
    print("xong", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["brain", "stt", "tts", "face", "dashcam", "diacritics"])
    ap.add_argument("--stt-size", default="small", choices=["tiny", "base", "small"])
    args = ap.parse_args()
    failed = []
    for n in args.only:
        try:
            get(n, args.stt_size)
        except Exception as e:  # repo HF chưa có (model chưa train xong) -> bỏ qua, các phần khác vẫn tải
            print(f"bỏ qua {n}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            failed.append(n)
    if failed:
        print("chưa tải được:", ", ".join(failed), "-> copilot tự tắt các thành phần này")
