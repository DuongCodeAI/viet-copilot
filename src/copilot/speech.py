"""Nghe (STT) và nói (TTS), đều chạy offline trên CPU.

STT: PhoWhisper-small (VinAI, BSD-3) đổi sang CTranslate2 int8 để chạy bằng faster-whisper.
     Bản gốc 967MB fp32; int8 ~250MB và nhanh hơn ~3-4 lần trên CPU.
TTS: Piper, giọng vi_VN-vais1000-medium (63MB, dữ liệu CC-BY-4.0). Piper là GPL-3.0.
"""

import time
from dataclasses import dataclass

import numpy as np

SR = 16000


@dataclass
class Transcript:
    text: str
    ms: float
    audio_sec: float


class STT:
    def __init__(self, model_dir: str, threads: int = 4, beam_size: int = 1):
        from faster_whisper import WhisperModel

        # beam 1 (greedy): câu lệnh trong xe ngắn, beam 5 chậm gần gấp đôi mà WER gần như không đổi
        self.model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)
        self.beam = beam_size

    def __call__(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        segs, _ = self.model.transcribe(audio.astype(np.float32), language="vi", beam_size=self.beam,
                                        vad_filter=True, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        return Transcript(text, (time.perf_counter() - t0) * 1e3, len(audio) / SR)


class TTS:
    def __init__(self, voice_path: str):
        from piper import PiperVoice

        self.voice = PiperVoice.load(voice_path)
        self.sr = self.voice.config.sample_rate

    def synth(self, text: str) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
        return audio, (time.perf_counter() - t0) * 1e3

    def speak(self, text: str):
        import sounddevice as sd

        audio, _ = self.synth(text)
        sd.play(audio, self.sr)
        sd.wait()


def energy_vad_record(max_sec: float = 8, silence_sec: float = 0.8, thr: float = 0.012) -> np.ndarray:
    """Ghi âm từ micro tới khi im lặng `silence_sec` giây (VAD theo năng lượng, đủ dùng trong phòng/xe yên)."""
    import sounddevice as sd

    block = int(0.05 * SR)
    frames, silent, started = [], 0.0, False
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=block) as st:
        t = 0.0
        while t < max_sec:
            x, _ = st.read(block)
            x = x[:, 0]
            loud = float(np.sqrt((x ** 2).mean())) > thr
            started |= loud
            if started:
                frames.append(x)
                silent = 0.0 if loud else silent + 0.05
                if silent >= silence_sec:
                    break
            t += 0.05
    return np.concatenate(frames) if frames else np.zeros(0, np.float32)


def mix_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Trộn tiếng ồn (gió, máy xe) vào giọng nói theo SNR cho trước. Dùng khi fine-tune STT."""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(clean) + 1))
    n = noise[start:start + len(clean)]
    p_s = float((clean ** 2).mean()) + 1e-10
    p_n = float((n ** 2).mean()) + 1e-10
    scale = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    out = clean + scale * n
    return (out / max(1.0, float(np.abs(out).max()))).astype(np.float32)
