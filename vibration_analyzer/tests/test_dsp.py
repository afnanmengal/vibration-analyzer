"""
test_dsp.py - Unit and Validation Tests for DSP & Vibration Feature Extraction Engine
"""

import numpy as np
import pytest
from vibration_analyzer.dsp import (
    DSPOptimizer,
    RealtimeBandpassFilter,
    VibrationFeatures,
    WindowType,
)


def test_circular_buffer():
    from vibration_analyzer.audio_stream import CircularAudioBuffer

    buf = CircularAudioBuffer(capacity=100)
    assert len(buf.read_latest(50)) == 50
    # Initially zero-padded
    assert np.all(buf.read_latest(50) == 0)

    # Write 60 samples
    data1 = np.ones(60, dtype=np.float32) * 5.0
    buf.write(data1)
    read = buf.read_latest(60)
    assert np.allclose(read, 5.0)

    # Write 60 more samples (wrap-around)
    data2 = np.ones(60, dtype=np.float32) * 9.0
    buf.write(data2)
    read_latest_30 = buf.read_latest(30)
    assert np.allclose(read_latest_30, 9.0)

    # Read across the boundary
    read_80 = buf.read_latest(80)
    assert np.allclose(read_80[-60:], 9.0)
    assert np.allclose(read_80[:20], 5.0)


def test_window_coherent_gain():
    dsp = DSPOptimizer(sample_rate=44100, block_size=4096)
    w_hann = dsp.get_window(WindowType.HANN, 4096)
    # Theoretical coherent gain of Hann window is 0.5
    assert abs(w_hann.coherent_gain - 0.5) < 0.01

    w_rect = dsp.get_window(WindowType.RECTANGULAR, 4096)
    assert abs(w_rect.coherent_gain - 1.0) < 1e-6


def test_fft_pure_sine():
    sr = 44100
    n = 4096
    dsp = DSPOptimizer(sample_rate=sr, block_size=n)

    # Pure 100 Hz sine wave with peak amplitude 1.0
    freq_target = 100.0
    amp_target = 1.0
    t = np.arange(n) / sr
    sig = amp_target * np.sin(2 * np.pi * freq_target * t)

    freqs, amps, _ = dsp.compute_fft(sig, WindowType.BLACKMAN_HARRIS)
    peak_idx = np.argmax(amps)
    detected_freq = freqs[peak_idx]

    # Frequency bin resolution df = 44100 / 4096 ≈ 10.76 Hz
    assert abs(detected_freq - freq_target) < 11.0
    # Peak amplitude should be close to 1.0 thanks to coherent gain correction
    assert abs(amps[peak_idx] - amp_target) < 0.15


def test_time_features():
    dsp = DSPOptimizer(sample_rate=44100, block_size=2048)

    # Generate Gaussian noise (expected Kurtosis ≈ 3.0, Skewness ≈ 0.0)
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=0.5, size=2048)
    feats = dsp.extract_time_features(noise)

    assert abs(feats["rms"] - 0.5) < 0.05
    assert abs(feats["kurtosis"] - 3.0) < 0.35
    assert abs(feats["skewness"]) < 0.2

    # Impulsive signal (high kurtosis and high crest factor)
    impulsive = noise.copy()
    impulsive[100] = 15.0  # Large sharp spike
    imp_feats = dsp.extract_time_features(impulsive)
    assert imp_feats["kurtosis"] > 10.0
    assert imp_feats["crest_factor"] > 10.0


def test_bandpass_filter_continuity():
    # Verify that sosfilt state maintains continuity across consecutive blocks without glitches
    sr = 44100
    filt = RealtimeBandpassFilter(sample_rate=sr, low_cutoff_hz=20.0, high_cutoff_hz=500.0)

    t = np.arange(4096) / sr
    # Low frequency drift (5 Hz - should be attenuated) + target vibration (100 Hz - passed)
    sig_block1 = 0.8 * np.sin(2 * np.pi * 5.0 * t[:2048]) + 0.5 * np.sin(2 * np.pi * 100.0 * t[:2048])
    sig_block2 = 0.8 * np.sin(2 * np.pi * 5.0 * t[2048:]) + 0.5 * np.sin(2 * np.pi * 100.0 * t[2048:])

    out1 = filt.process(sig_block1)
    out2 = filt.process(sig_block2)

    # Check that boundary transition is smooth
    boundary_jump = abs(out2[0] - out1[-1])
    # The jump between contiguous filtered frames should be small (< 0.1)
    assert boundary_jump < 0.1


def test_harmonic_and_thd_detection():
    sr = 44100
    n = 4096
    dsp = DSPOptimizer(sample_rate=sr, block_size=n)

    t = np.arange(n) / sr
    # Fundamental: 50 Hz (Amp 1.0)
    # 2X harmonic: 100 Hz (Amp 0.3)
    # 3X harmonic: 150 Hz (Amp 0.1)
    sig = 1.0 * np.sin(2 * np.pi * 50.0 * t) + 0.3 * np.sin(2 * np.pi * 100.0 * t) + 0.1 * np.sin(2 * np.pi * 150.0 * t)

    freqs, amps, _ = dsp.compute_fft(sig, WindowType.BLACKMAN_HARRIS)
    freq_feats = dsp.extract_frequency_features(freqs, amps)

    assert abs(freq_feats["dominant_frequency_hz"] - 50.0) < 5.0
    assert abs(freq_feats["calculated_rpm"] - 3000.0) < 300.0
    # Expected THD = sqrt(0.3^2 + 0.1^2) / 1.0 ≈ 31.6%
    assert 20.0 < freq_feats["thd_percent"] < 40.0
    assert "1X" in freq_feats["harmonics"]
    assert "2X" in freq_feats["harmonics"]
