"""Tạo wav lệnh bằng Piper (16 kHz) cho demo/script_voice.yaml: chạy đường giọng nói thật trong replay.

Giọng tổng hợp, không phải người thật. python scripts/make_voice_cmds.py
"""
import sys, yaml, numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.path.insert(0, "src")
from copilot.speech import TTS

tts = TTS("models/piper/vi_VN-vais1000-medium.onnx")
items = yaml.safe_load(open("demo/script.yaml", encoding="utf-8"))
out = []
for i, it in enumerate(items, 1):
    a, _ = tts.synth(it["text"])
    a = resample_poly(a.astype(np.float32) / 32768, 16000, tts.sr).astype(np.float32)
    a = np.concatenate([np.zeros(4000, np.float32), a, np.zeros(4000, np.float32)])
    path = f"demo/voice/cmd_{i:02d}.wav"
    sf.write(path, a, 16000)
    out.append({"t": it["t"], "wav": path, "text": it["text"]})
with open("demo/script_voice.yaml", "w", encoding="utf-8") as f:
    f.write("# như script.yaml nhưng đi qua STT thật; wav tạo bằng Piper (giọng tổng hợp, 16 kHz)\n")
    yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, width=200)
print(open("demo/script_voice.yaml", encoding="utf-8").read())
