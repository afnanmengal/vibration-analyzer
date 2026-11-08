"""
audio_stream.py - High-Performance Real-Time Audio & Vibro-Acoustic Stream Acquisition Engine

Architectural Features:
- Non-blocking real-time audio input ingestion using `sounddevice`.
- Thread-safe circular ring buffer (lock-free / mutex-protected) to decouple ingestion from DSP/GUI.
- Comprehensive buffer overrun / underrun tracking and telemetry.
- Hardware device enumeration and safe fallback mechanism.
- Integrated synthetic mechanical vibration generator mode for hardware-independent diagnostics.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

# Configure module logger
logger = logging.getLogger("vibration_analyzer.audio_stream")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [AudioStream] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class AudioStreamConfig:
    """Configuration parameters for audio acquisition."""
    sample_rate: int = 44100
    block_size: int = 4096
    channels: int = 1
    dtype: str = "float32"
    device_index: Optional[int] = None
    ring_buffer_size: int = 65536  # ~1.48 seconds at 44.1 kHz
    gain: float = 1.0


@dataclass
class StreamDiagnostics:
    """Real-time streaming statistics and hardware health metrics."""
    is_active: bool = False
    samples_acquired: int = 0
    buffer_overruns: int = 0
    buffer_underruns: int = 0
    latency_ms: float = 0.0
    actual_sample_rate: int = 44100
    active_device_name: str = "None"
    is_synthetic: bool = False


class CircularAudioBuffer:
    """
    High-performance thread-safe circular ring buffer for real-time audio samples.
    Enables single-writer (audio callback thread) and multi-reader (DSP & GUI threads)
    without data race conditions.
    """

    def __init__(self, capacity: int, dtype: str = "float32"):
        self.capacity = int(capacity)
        self.buffer = np.zeros(self.capacity, dtype=dtype)
        self.write_index = 0
        self.total_samples_written = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """Write incoming audio frame chunk into circular buffer."""
        chunk = data.flatten()
        chunk_len = len(chunk)
        if chunk_len == 0:
            return

        with self._lock:
            if chunk_len >= self.capacity:
                # Chunk larger than buffer; keep only the latest capacity samples
                self.buffer[:] = chunk[-self.capacity:]
                self.write_index = 0
            else:
                end_index = self.write_index + chunk_len
                if end_index <= self.capacity:
                    self.buffer[self.write_index:end_index] = chunk
                else:
                    first_part = self.capacity - self.write_index
                    self.buffer[self.write_index:self.capacity] = chunk[:first_part]
                    second_part = chunk_len - first_part
                    self.buffer[0:second_part] = chunk[first_part:]
                self.write_index = end_index % self.capacity
            self.total_samples_written += chunk_len

    def read_latest(self, num_samples: int) -> np.ndarray:
        """
        Extract the most recent `num_samples` contiguous audio samples.
        If requested count exceeds written samples, zero-pads the oldest slots.
        """
        num_samples = min(int(num_samples), self.capacity)
        out = np.zeros(num_samples, dtype=self.buffer.dtype)

        with self._lock:
            if self.total_samples_written == 0:
                return out

            samples_available = min(self.total_samples_written, self.capacity)
            samples_to_read = min(num_samples, samples_available)

            current_ptr = self.write_index
            start_ptr = (current_ptr - samples_to_read) % self.capacity

            if start_ptr < current_ptr:
                read_data = self.buffer[start_ptr:current_ptr]
            else:
                # Wrapped around the end of the circular buffer
                first_slice = self.buffer[start_ptr:self.capacity]
                second_slice = self.buffer[0:current_ptr]
                read_data = np.concatenate((first_slice, second_slice))

            # Place into end of output buffer if zero-padding was necessary
            out[-samples_to_read:] = read_data
            return out.copy()

    def clear(self) -> None:
        """Reset buffer state."""
        with self._lock:
            self.buffer.fill(0)
            self.write_index = 0
            self.total_samples_written = 0


class AudioStreamEngine:
    """
    Core non-blocking audio stream ingestion engine.
    Supports hardware microphones via PortAudio/sounddevice as well as
    synthetic mechanical vibration generator mode.
    """

    def __init__(self, config: Optional[AudioStreamConfig] = None):
        self.config = config or AudioStreamConfig()
        self.ring_buffer = CircularAudioBuffer(
            capacity=self.config.ring_buffer_size,
            dtype=self.config.dtype,
        )
        self.diagnostics = StreamDiagnostics()
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        self._is_synthetic = False
        self._synthetic_thread: Optional[threading.Thread] = None
        self._synthetic_stop_event = threading.Event()
        self._synthetic_generator: Optional[Callable[[int, int], np.ndarray]] = None
        self._state_lock = threading.Lock()

    @staticmethod
    def get_input_devices() -> List[Dict[str, any]]:
        """
        Enumerate all host audio input devices.
        Returns a list of device metadata dictionaries.
        """
        devices = []
        try:
            device_list = sd.query_devices()
            default_input = sd.default.device[0]
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Device #{idx}"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": int(dev.get("default_samplerate", 44100)),
                        "is_default": (idx == default_input),
                        "hostapi": dev.get("hostapi", 0),
                    })
        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}")
        return devices

    def set_gain(self, gain: float) -> None:
        """Adjust digital pre-amplification gain."""
        self.config.gain = max(0.0, float(gain))

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Real-time audio callback running on high-priority PortAudio thread.
        Applies gain, monitors overflow/underflow, and enqueues to ring buffer.
        """
        if status:
            if status.input_overflow:
                self.diagnostics.buffer_overruns += 1
                logger.warning("Audio buffer overrun detected in PortAudio callback.")
            if status.input_underflow:
                self.diagnostics.buffer_underruns += 1
                logger.warning("Audio buffer underrun detected in PortAudio callback.")

        # Extract mono channel and scale by gain
        audio_chunk = indata[:, 0] * self.config.gain
        self.ring_buffer.write(audio_chunk)
        self.diagnostics.samples_acquired += frames

    def start(self, device_index: Optional[int] = None, use_synthetic: bool = False) -> bool:
        """
        Start audio stream acquisition.
        If `use_synthetic` is True or no valid audio hardware is found,
        runs in synthetic mechanical vibration simulation mode.
        """
        with self._state_lock:
            if self._is_running:
                logger.info("Stream already running; restarting with new parameters...")
                self._stop_internal()

            self._is_synthetic = use_synthetic
            if device_index is not None:
                self.config.device_index = device_index

            if self._is_synthetic:
                return self._start_synthetic()

            # Attempt hardware device initialization
            try:
                selected_device = self.config.device_index
                device_info = (
                    sd.query_devices(selected_device, "input")
                    if selected_device is not None
                    else sd.query_devices(kind="input")
                )
                dev_name = device_info.get("name", "Default Input")
                logger.info(f"Opening audio input stream on '{dev_name}' @ {self.config.sample_rate} Hz")

                self._stream = sd.InputStream(
                    samplerate=self.config.sample_rate,
                    blocksize=self.config.block_size,
                    device=selected_device,
                    channels=self.config.channels,
                    dtype=self.config.dtype,
                    callback=self._audio_callback,
                )
                self._stream.start()

                self._is_running = True
                self.diagnostics.is_active = True
                self.diagnostics.is_synthetic = False
                self.diagnostics.active_device_name = dev_name
                self.diagnostics.actual_sample_rate = self.config.sample_rate
                self.diagnostics.latency_ms = (self.config.block_size / self.config.sample_rate) * 1000.0
                return True

            except Exception as exc:
                logger.error(f"Hardware audio device failed to open: {exc}. Falling back to synthetic simulation mode.")
                return self._start_synthetic()

    def set_synthetic_generator(self, generator_fn: Callable[[int, int], np.ndarray]) -> None:
        """Register a custom synthetic vibration signal generator callback."""
        self._synthetic_generator = generator_fn

    def _start_synthetic(self) -> bool:
        """Launch background synthetic vibration signal simulation thread."""
        self._synthetic_stop_event.clear()
        self._is_synthetic = True
        self._is_running = True

        self.diagnostics.is_active = True
        self.diagnostics.is_synthetic = True
        self.diagnostics.active_device_name = "Synthetic Mechanical Rig Simulator"
        self.diagnostics.actual_sample_rate = self.config.sample_rate
        self.diagnostics.latency_ms = (self.config.block_size / self.config.sample_rate) * 1000.0

        def synthetic_loop():
            sr = self.config.sample_rate
            bs = self.config.block_size
            sleep_duration = bs / sr

            # Fallback internal generator if none registered
            phase = 0.0
            rpm = 1800.0
            f0 = rpm / 60.0  # 30 Hz fundamental shaft speed

            logger.info("Synthetic Mechanical Vibration thread started.")
            while not self._synthetic_stop_event.is_set():
                start_time = time.perf_counter()

                if self._synthetic_generator is not None:
                    try:
                        chunk = self._synthetic_generator(bs, sr)
                    except Exception as err:
                        logger.error(f"Synthetic generator callback error: {err}")
                        chunk = np.zeros(bs, dtype=np.float32)
                else:
                    # Realistic synthetic motor signature
                    t = (np.arange(bs) + phase) / sr
                    phase += bs

                    # 1X RPM (Unbalance component)
                    s_1x = 0.35 * np.sin(2 * np.pi * f0 * t)
                    # 2X RPM (Misalignment component)
                    s_2x = 0.18 * np.sin(2 * np.pi * (2 * f0) * t)
                    # 3X RPM (Looseness / non-linear stiffness)
                    s_3x = 0.08 * np.sin(2 * np.pi * (3 * f0) * t)
                    # 120 Hz Electrical 2x line frequency humming
                    s_line = 0.05 * np.sin(2 * np.pi * 120.0 * t)
                    # High frequency structural resonance & bearing background
                    s_res = 0.04 * np.sin(2 * np.pi * 850.0 * t)
                    # Gaussian acoustic floor
                    noise = 0.03 * np.random.randn(bs)

                    chunk = (s_1x + s_2x + s_3x + s_line + s_res + noise).astype(np.float32)

                chunk = chunk * self.config.gain
                self.ring_buffer.write(chunk)
                self.diagnostics.samples_acquired += bs

                elapsed = time.perf_counter() - start_time
                remaining = sleep_duration - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)

            logger.info("Synthetic Mechanical Vibration thread exited.")

        self._synthetic_thread = threading.Thread(
            target=synthetic_loop,
            daemon=True,
            name="SyntheticVibrationWorker",
        )
        self._synthetic_thread.start()
        return True

    def _stop_internal(self) -> None:
        """Internal teardown without holding redundant state locks."""
        self._is_running = False
        self.diagnostics.is_active = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Stream closing note: {e}")
            self._stream = None

        if self._synthetic_thread is not None and self._synthetic_thread.is_alive():
            self._synthetic_stop_event.set()
            self._synthetic_thread.join(timeout=1.0)
            self._synthetic_thread = None

    def stop(self) -> None:
        """Gracefully stop stream acquisition."""
        with self._state_lock:
            self._stop_internal()
            logger.info("Audio acquisition stream stopped.")

    def get_latest_samples(self, count: Optional[int] = None) -> np.ndarray:
        """Retrieve latest `count` samples (defaults to configured block_size)."""
        samples_to_read = count or self.config.block_size
        return self.ring_buffer.read_latest(samples_to_read)

    def is_active(self) -> bool:
        """Check if stream is actively acquiring samples."""
        return self._is_running
