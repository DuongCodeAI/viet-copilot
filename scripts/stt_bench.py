"""Đo latency STT trên laptop giống speech.STT (faster-whisper int8, CPU, beam 1, vad_filter).

Không có VIVOS trên máy -> tổng hợp 20 câu lệnh trong xe bằng Piper (giọng vais1000), resample về 16 kHz.
WER trên giọng tổng hợp chỉ để kiểm tra model chạy đúng, không so được với WER trên VIVOS.

python scripts/stt_bench.py models/phowhisper-tiny-ct2-int8 [--threads 4]   (chạy trong thư mục viet-copilot)
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def resample(a, sr_in, sr_out=16000):
    t = np.arange(int(len(a) * sr_out / sr_in)) * sr_in / sr_out
    return np.interp(t, np.arange(len(a)), a).astype(np.float32)

sys.path.insert(0, "src")
from copilot.speech import STT, TTS  # noqa: E402

SENTS = [
    "bật điều hoà lên hai mươi hai độ", "mở cửa sổ bên phía tài xế", "tìm trạm sạc gần nhất",
    "gọi cho mẹ", "xe máy vượt đèn đỏ bị phạt bao nhiêu tiền", "giảm âm lượng xuống một chút",
    "dẫn đường đến sân bay nội bài", "bật chế độ sưởi ghế", "còn bao nhiêu phần trăm pin",
    "tắt nhạc đi", "đỗ xe trên vỉa hè có bị phạt không", "mở bài hát tiếp theo",
    "tăng nhiệt độ lên một độ", "khoá cửa xe", "đi ngược chiều thì bị trừ mấy điểm",
    "bật đèn pha", "quãng đường còn lại bao xa", "đóng cửa sổ trời",
    "chuyển sang chế độ lái tiết kiệm", "nhắc tôi nghỉ sau một tiếng nữa",
]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.lower().split(), hyp.lower().replace(",", "").replace(".", "").replace("?", "").split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--voice", default="models/piper/vi_VN-vais1000-medium.onnx")
    args = ap.parse_args()

    # Piper có nhiễu ngẫu nhiên -> mỗi lần tổng hợp ra audio khác; lưu lại để mọi model đo trên cùng audio
    os.makedirs("logs", exist_ok=True)
    cache = "logs/stt_audio_20.npz"
    try:
        z = np.load(cache)
        audios = [z[f"a{i}"] for i in range(len(SENTS))]
    except FileNotFoundError:
        tts = TTS(args.voice)
        audios = []
        for s in SENTS:
            a, _ = tts.synth(s)
            a = resample(a.astype(np.float32) / 32768, tts.sr)
            audios.append(np.concatenate([np.zeros(4000, np.float32), a, np.zeros(4000, np.float32)]))  # 0.25 s lặng 2 đầu
        np.savez(cache, **{f"a{i}": a for i, a in enumerate(audios)})

    stt = STT(args.model, threads=args.threads)
    stt(audios[0])  # warm-up
    ms, rtf, errs, words, rows = [], [], 0, 0, []
    for s, a in zip(SENTS, audios):
        t0 = time.perf_counter()
        tr = stt(a)
        dt = (time.perf_counter() - t0) * 1e3
        e, n = wer(s, tr.text)
        errs, words = errs + e, words + n
        ms.append(dt)
        rtf.append(dt / 1000 / (len(a) / 16000))
        rows.append({"ref": s, "hyp": tr.text, "ms": round(dt), "sec": round(len(a) / 16000, 2)})
    ms_s = sorted(ms)
    res = {"threads": args.threads, "n": len(SENTS), "audio_sec_mean": round(float(np.mean([r["sec"] for r in rows])), 2),
           "p50_ms": round(ms_s[len(ms_s) // 2]), "p95_ms": round(ms_s[int(len(ms_s) * 0.95) - 1]),
           "rtf_mean": round(float(np.mean(rtf)), 3), "wer_tts": round(errs / words, 4), "rows": rows}
    json.dump(res, open("logs/stt_bench.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: v for k, v in res.items() if k != "rows"})
    for r in rows:
        if r["hyp"].lower().strip(" .?,") != r["ref"]:
            print("  ", r["ref"], "->", r["hyp"])


if __name__ == "__main__":
    main()
