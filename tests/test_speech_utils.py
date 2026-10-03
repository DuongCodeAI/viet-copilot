import numpy as np

from copilot.speech import mix_noise


def test_mix_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 200, 16000)).astype(np.float32) * 0.3
    noise = rng.normal(0, 0.1, 8000).astype(np.float32)  # ngắn hơn clean -> phải lặp lại
    out = mix_noise(clean, noise, snr_db=5, rng=rng)
    n = out - clean
    snr = 10 * np.log10((clean ** 2).mean() / (n ** 2).mean())
    assert abs(snr - 5) < 0.5
    assert np.abs(out).max() <= 1.0
