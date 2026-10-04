"""WER của PhoWhisper CT2 int8 trên laptop: 200 câu đầu VIVOS test, sạch + ồn SNR 10/5/0 dB.

Tạo ồn, trộn ồn, chuẩn hoá text giống hệt notebook 01 (cùng seed) để so được với bảng fine-tune.
Khác notebook: engine là faster-whisper int8 + beam 1 + vad_filter (giống speech.STT), không phải transformers fp16.

python scripts/stt_vivos_eval.py models/phowhisper-small-ct2-int8 small [--n 200]   (chạy trong thư mục viet-copilot)
"""
import argparse
import io
import json
import os
import re
import sys
import time

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

sys.path.insert(0, "src")
from copilot.speech import STT, mix_noise  # noqa: E402

SR = 16000
rng = np.random.default_rng(0)


def colored(n, power):
    spec = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= f[1:] ** (power / 2)
    spec[f < 60] = 0
    x = np.fft.irfft(spec, n)
    return (x / np.abs(x).max()).astype(np.float32)


def make_noise_test():
    n = SR * 60
    t = np.arange(n) / SR
    road = colored(n, 1.0)
    wind = colored(n, 0.3) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t))
    engine = colored(n, 2.0) + 0.3 * np.sin(2 * np.pi * 45 * t) + 0.15 * np.sin(2 * np.pi * 90 * t)
    return [x.astype(np.float32)[int(len(x) * 0.8):] for x in (road, wind, engine)]


def norm(s):
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def load_vivos_test(n):
    p = hf_hub_download("ademax/vivos-vie-speech2text", "data/test-00000-of-00001-dc934696ee903158.parquet",
                        repo_type="dataset")
    tb = pq.read_table(p).slice(0, n).to_pylist()
    col_a = [c for c in tb[0] if "audio" in c][0]
    col_t = [c for c in tb[0] if c in ("sentence", "text", "transcription")][0]
    audios, texts = [], []
    for row in tb:
        a, sr = sf.read(io.BytesIO(row[col_a]["bytes"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != SR:
            t = np.arange(int(len(a) * SR / sr)) * sr / SR
            a = np.interp(t, np.arange(len(a)), a).astype(np.float32)
        audios.append(a)
        texts.append(row[col_t])
    return audios, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    noise_test = make_noise_test()
    audios, texts = load_vivos_test(args.n)
    stt = STT(args.model, threads=args.threads)
    refs = [norm(t) for t in texts]
    out = {"model": args.name, "n": args.n}
    for snr in (None, 10, 5, 0):
        r = np.random.default_rng(123)
        hyps, t0 = [], time.perf_counter()
        for a in audios:
            if snr is not None:
                a = mix_noise(a, noise_test[r.integers(len(noise_test))], snr, r)
            hyps.append(norm(stt(a).text))
        key = "sạch" if snr is None else f"snr{snr}"
        out[key] = round(100 * jiwer.wer(refs, hyps), 2)
        out[key + "_s_per_utt"] = round((time.perf_counter() - t0) / len(audios), 2)
        print(key, out[key], "%", out[key + "_s_per_utt"], "s/câu", flush=True)
        json.dump(out, open(f"logs/stt_vivos_{args.name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
