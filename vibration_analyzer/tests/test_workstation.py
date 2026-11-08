"""
test_workstation.py - Unit & Integration Tests for GPU-Accelerated Workstation Pipeline
"""

import os
import tempfile
import numpy as np
import pytest
from vibration_analyzer.audio_engine import AudioConfig, AudioEngine, AtomicAudioRingBuffer
from vibration_analyzer.dsp_worker import (
    _calc_numba_kinematics,
    _find_trigger_index,
    DSPWorkerThread,
    WindowType,
)


def test_atomic_ring_buffer():
    buf = AtomicAudioRingBuffer(capacity=1000)
    out = np.zeros(200, dtype=np.float32)

    # Initial read should be zeros
    res = buf.read_latest(200, out)
    assert np.all(res == 0.0)

    # Write 300 samples of 4.0
    data = np.full(300, 4.0, dtype=np.float32)
    buf.write(data)
    buf.read_latest(200, out)
    assert np.allclose(out, 4.0)

    # Wrap-around write
    data2 = np.full(900, 7.0, dtype=np.float32)
    buf.write(data2)
    buf.read_latest(200, out)
    assert np.allclose(out, 7.0)


def test_numba_kinematics_accuracy():
    np.random.seed(42)
    x = np.random.normal(loc=0.0, scale=1.0, size=4096).astype(np.float32)

    rms, peak, p2p, cf, kurt, skew, sf = _calc_numba_kinematics(x)

    # Compare with standard definitions
    expected_rms = float(np.sqrt(np.mean(x ** 2)))
    expected_peak = float(np.max(np.abs(x)))
    expected_p2p = float(np.max(x) - np.min(x))

    assert abs(rms - expected_rms) < 1e-4
    assert abs(peak - expected_peak) < 1e-4
    assert abs(p2p - expected_p2p) < 1e-4
    # Gaussian kurtosis should be near 3.0
    assert abs(kurt - 3.0) < 0.35
    assert abs(skew) < 0.2


def test_oscilloscope_trigger():
    # Construct a sine wave
    t = np.linspace(0, 1.0, 1000, dtype=np.float32)
    sig = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz

    trig_idx = _find_trigger_index(sig, trigger_level=0.0)
    # At trigger index, sig[trig_idx] <= 0 and sig[trig_idx + 1] > 0
    assert sig[trig_idx] <= 0.0
    assert sig[trig_idx + 1] > 0.0


def test_flight_recorder():
    engine = AudioEngine(AudioConfig(sample_rate=44100, block_size=1024))
    engine.start_flight_recording()

    # Simulate 5 audio chunks
    for _ in range(5):
        chunk = 0.2 * np.sin(np.linspace(0, 10, 1024)).astype(np.float32)
        engine._audio_callback(chunk[:, None], 1024, {}, None)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = os.path.join(tmpdir, "test_flight.wav")
        samples_written = engine.stop_and_save_flight_recording(wav_file)
        assert samples_written == 5 * 1024
        assert os.path.exists(wav_file)
        assert os.path.getsize(wav_file) > 1000
