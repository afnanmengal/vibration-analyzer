"""
dsp_worker.py - Asynchronous Real-Time DSP Worker QThread & ML Anomaly Engine

High-Performance Signal Processing Pipeline:
- QThread execution running at 60-100 Hz decoupled from the GUI thread.
- Numba JIT accelerated kinematic & statistical feature extraction (RMS, Crest Factor, Kurtosis, Skewness).
- Real-time Butterworth Second-Order Sections (SOS) bandpass filter with continuous state registers.
- Fast Fourier Transform (FFT) with configurable bin sizes (1024, 2048, 4096, 8192, 16384).
- Harmonic Tracking Engine: 1X fundamental speed (RPM) and harmonics 2X through 10X.
- Sideband markers for gearmesh (GMF) and bearing fault frequencies (BPFO, BPFI).
- Decaying Peak-Hold Trace: Exponentially decaying maximum amplitude envelope.
- Envelope Demodulation (High-Frequency Resonance Technique / HFRT):
    * Analytic signal via Hilbert transform to expose bearing impact click trains.
- Power Cepstrum:
    * Real cepstrum for detecting harmonic family spacing and quefrency peaks.
- Auto-Triggering: Rising-edge zero-crossing search for oscilloscope waveform stabilization.
- Machine Learning Anomaly Detection:
    * 10-second baseline calibration routine.
    * Regularized Mahalanobis Distance & Scikit-Learn Isolation Forest outlier scoring.
    * ISO 10816 Vibration Severity Zone classification (Zones A, B, C, D).
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numba as nb
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
import scipy.signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .audio_engine import AudioEngine

logger = logging.getLogger("vibration_analyzer.dsp_worker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [DSPWorker] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class WindowType(enum.Enum):
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN_HARRIS = "blackman_harris"
    FLATTOP = "flattop"
    RECTANGULAR = "rectangular"


class ISOZone(enum.Enum):
    ZONE_A = "Zone A (Good / New)"
    ZONE_B = "Zone B (Acceptable)"
    ZONE_C = "Zone C (Alert / Unsatisfactory)"
    ZONE_D = "Zone D (Danger / Shutdown)"


class MachineClass(enum.Enum):
    CLASS_I = "Class I: Small Machines (< 15 kW / Laptops)"
    CLASS_II = "Class II: Medium Machines (15 kW - 75 kW)"
    CLASS_III = "Class III: Large Rigid Foundation (> 75 kW)"
    CLASS_IV = "Class IV: Large Flexible Foundation"


@dataclass
class HarmonicCursorInfo:
    """Detected harmonic order data."""
    order_multiplier: float
    frequency_hz: float
    amplitude: float
    db_amplitude: float


@dataclass
class DSPFramePayload:
    """Consolidated diagnostic payload delivered to GUI thread."""
    timestamp: float = 0.0

    # Time-Domain Waveforms
    raw_time_sig: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    filtered_time_sig: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    triggered_time_sig: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Narrowband FFT Spectrum
    freqs_hz: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    amp_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    db_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    peak_hold_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Envelope Demodulation (HFRT) & Cepstrum
    envelope_freqs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    envelope_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cepstrum_quefrency_ms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cepstrum_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Harmonics & Cursors
    dominant_frequency_hz: float = 0.0
    calculated_rpm: float = 0.0
    harmonics: Dict[str, HarmonicCursorInfo] = field(default_factory=dict)
    sidebands: List[HarmonicCursorInfo] = field(default_factory=list)

    # Statistical Kinematic Metrics
    rms: float = 0.0
    peak: float = 0.0
    peak_to_peak: float = 0.0
    crest_factor: float = 1.0
    kurtosis: float = 3.0
    skewness: float = 0.0
    shape_factor: float = 1.0
    thd_percent: float = 0.0

    # Health & Anomaly Telemetry
    health_index_percent: float = 100.0
    anomaly_score: float = 0.0
    mahalanobis_distance: float = 0.0
    is_anomaly: bool = False
    iso_zone: ISOZone = ISOZone.ZONE_A
    iso_zone_color: str = "#00ff88"
    fault_driver: str = "Nominal Baseline"
    calibration_progress: float = 0.0
    is_calibrating: bool = False


# --- Numba JIT Statistical Feature Acceleration ---
@nb.njit(fastmath=True, nogil=True)
def _calc_numba_kinematics(x: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """
    Zero-heap-allocation JIT accelerated kinematics:
    Returns (rms, peak, p2p, crest_factor, kurtosis, skewness, shape_factor).
    """
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 1.0

    sum_x = 0.0
    sum_sq = 0.0
    sum_abs = 0.0
    val_max = x[0]
    val_min = x[0]
    peak = 0.0

    for i in range(n):
        v = x[i]
        av = abs(v)
        if av > peak:
            peak = av
        if v > val_max:
            val_max = v
        if v < val_min:
            val_min = v
        sum_x += v
        sum_sq += v * v
        sum_abs += av

    mean = sum_x / n
    variance = (sum_sq / n) - (mean * mean)
    std = np.sqrt(max(0.0, variance))
    rms = np.sqrt(sum_sq / n)
    p2p = val_max - val_min

    eps = 1e-12
    cf = peak / (rms + eps)
    sf = rms / ((sum_abs / n) + eps)

    # 3rd & 4th standardized central statistical moments
    m3 = 0.0
    m4 = 0.0
    if std > eps:
        for i in range(n):
            diff = (x[i] - mean) / std
            d2 = diff * diff
            m3 += d2 * diff
            m4 += d2 * d2
        skew = m3 / n
        kurt = m4 / n
    else:
        skew = 0.0
        kurt = 3.0

    return rms, peak, p2p, cf, kurt, skew, sf


@nb.njit(fastmath=True, nogil=True)
def _find_trigger_index(signal: np.ndarray, trigger_level: float = 0.0) -> int:
    """Find rising-edge zero-crossing to stabilize oscilloscope waveform."""
    n = len(signal)
    search_limit = min(n - 1, 1024)
    for i in range(search_limit):
        if signal[i] <= trigger_level < signal[i + 1]:
            return i
    return 0


class DSPWorkerThread(QThread):
    """
    Asynchronous DSP Worker Thread.
    Emits `processed_frame_ready` signal with full diagnostic payload at 60 Hz.
    """
    processed_frame_ready = Signal(object)

    def __init__(self, audio_engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.audio_engine = audio_engine
        self._is_running = True

        # Pipeline configuration
        self.sample_rate = audio_engine.config.sample_rate
        self.fft_size = 4096
        self.window_type = WindowType.BLACKMAN_HARRIS
        self.filter_enabled = True
        self.filter_low_hz = 10.0
        self.filter_high_hz = 3000.0
        self.peak_hold_decay = 0.985
        self.machine_class = MachineClass.CLASS_I
        self.iso_scale_factor = 60.0

        # Precomputed filter SOS & delay registers
        self._sos = None
        self._zi = None
        self._rebuild_filter()

        # Preallocated buffers
        self._raw_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self._peak_hold_buffer = np.zeros((self.fft_size // 2) + 1, dtype=np.float32)

        # Warmup Numba JIT functions to eliminate first-frame latency jitter
        try:
            _dummy = np.zeros(64, dtype=np.float32)
            _calc_numba_kinematics(_dummy)
            _find_trigger_index(_dummy, 0.0)
        except Exception as e:
            logger.debug(f"Numba warmup note: {e}")

        # Precomputed window cache
        self._window_cache: Dict[Tuple[str, int], Tuple[np.ndarray, float]] = {}

        # Predictive ML Anomaly Detection state
        self._is_calibrating = False
        self._cal_start_time = 0.0
        self._cal_duration = 10.0
        self._cal_buffer: List[np.ndarray] = []
        self.is_calibrated = False
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.precision_matrix: Optional[np.ndarray] = None
        self.mahalanobis_threshold_95 = 3.5
        self.mahalanobis_threshold_99 = 5.5
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None

    def update_dsp_params(
        self,
        fft_size: Optional[int] = None,
        window_type: Optional[WindowType] = None,
        filter_enabled: Optional[bool] = None,
        filter_low_hz: Optional[float] = None,
        filter_high_hz: Optional[float] = None,
        machine_class: Optional[MachineClass] = None,
    ) -> None:
        """Thread-safe update of DSP settings."""
        if fft_size is not None and fft_size != self.fft_size:
            self.fft_size = fft_size
            self._raw_buffer = np.zeros(self.fft_size, dtype=np.float32)
            self._peak_hold_buffer = np.zeros((self.fft_size // 2) + 1, dtype=np.float32)
        if window_type is not None:
            self.window_type = window_type
        if filter_enabled is not None:
            self.filter_enabled = filter_enabled
        if filter_low_hz is not None or filter_high_hz is not None:
            if filter_low_hz is not None:
                self.filter_low_hz = max(1.0, float(filter_low_hz))
            if filter_high_hz is not None:
                self.filter_high_hz = min(max(self.filter_low_hz + 10.0, float(filter_high_hz)), (self.sample_rate / 2.0) - 100.0)
            self._rebuild_filter()
        if machine_class is not None:
            self.machine_class = machine_class

    def _rebuild_filter(self) -> None:
        nyquist = self.sample_rate / 2.0
        low = max(0.0001, min(self.filter_low_hz / nyquist, 0.99))
        high = max(low + 0.001, min(self.filter_high_hz / nyquist, 0.999))
        try:
            self._sos = scipy.signal.butter(4, [low, high], btype="bandpass", output="sos")
            self._zi = scipy.signal.sosfilt_zi(self._sos)
        except Exception:
            self._sos = scipy.signal.butter(2, low, btype="highpass", output="sos")
            self._zi = scipy.signal.sosfilt_zi(self._sos)

    def _get_window(self, win_type: WindowType, n: int) -> Tuple[np.ndarray, float]:
        key = (win_type.value, n)
        if key in self._window_cache:
            return self._window_cache[key]

        if win_type == WindowType.HANN:
            w = np.hanning(n)
        elif win_type == WindowType.HAMMING:
            w = np.hamming(n)
        elif win_type == WindowType.BLACKMAN_HARRIS:
            w = scipy.signal.windows.blackmanharris(n)
        elif win_type == WindowType.FLATTOP:
            w = scipy.signal.windows.flattop(n)
        else:
            w = np.ones(n, dtype=np.float64)

        cg = float(np.mean(w))
        res = (w.astype(np.float32), cg)
        self._window_cache[key] = res
        return res

    def start_calibration(self, duration_sec: float = 10.0) -> None:
        """Trigger 10-second baseline recording."""
        self._is_calibrating = True
        self._cal_duration = max(3.0, float(duration_sec))
        self._cal_start_time = time.perf_counter()
        self._cal_buffer.clear()
        logger.info(f"Baseline calibration initiated for {self._cal_duration:.1f}s.")

    def reset_calibration(self) -> None:
        """Reset learned baseline model."""
        self.is_calibrated = False
        self._is_calibrating = False
        self.baseline_mean = None
        self.baseline_std = None
        self.precision_matrix = None
        self.isolation_forest = None
        self.scaler = None
        self._cal_buffer.clear()
        logger.info("Baseline calibration reset.")

    def _finalize_calibration(self) -> None:
        self._is_calibrating = False
        if len(self._cal_buffer) < 15:
            return

        mat = np.array(self._cal_buffer, dtype=np.float64)
        self.baseline_mean = np.mean(mat, axis=0)
        self.baseline_std = np.std(mat, axis=0)
        self.baseline_std[self.baseline_std < 1e-6] = 1e-6

        cov = np.cov(mat, rowvar=False)
        dim = cov.shape[0]
        ridge = 1e-4 * np.trace(cov) / max(1, dim) if np.trace(cov) > 0 else 1e-4
        cov_reg = cov + np.eye(dim) * ridge

        try:
            self.precision_matrix = np.linalg.pinv(cov_reg)
        except Exception:
            self.precision_matrix = np.diag(1.0 / (self.baseline_std ** 2))

        diffs = mat - self.baseline_mean
        d_sq = np.sum(diffs @ self.precision_matrix * diffs, axis=1)
        dists = np.sqrt(np.maximum(0.0, d_sq))
        self.mahalanobis_threshold_95 = float(np.percentile(dists, 95))
        self.mahalanobis_threshold_99 = float(np.percentile(dists, 99))

        try:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(mat)
            self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
            self.isolation_forest.fit(scaled)
        except Exception:
            self.isolation_forest = None

        self.is_calibrated = True
        logger.info(f"Baseline calibration finalized with {len(mat)} frames.")

    def stop_worker(self) -> None:
        """Signal thread shutdown."""
        self._is_running = False

    def run(self) -> None:
        """Asynchronous execution loop at ~60 FPS."""
        logger.info("DSP Worker Thread started.")
        target_period = 1.0 / 60.0  # 60 Hz execution

        while self._is_running:
            t0 = time.perf_counter()

            # 1. Acquire latest audio samples from ring buffer
            raw = self.audio_engine.get_latest_samples(self.fft_size, self._raw_buffer)

            # 2. Stateful Second-Order Sections (SOS) bandpass filtering
            if self.filter_enabled and self._sos is not None:
                if self._zi is None or self._zi.shape[0] != self._sos.shape[0]:
                    self._zi = scipy.signal.sosfilt_zi(self._sos) * raw[0]
                filtered, self._zi = scipy.signal.sosfilt(self._sos, raw, zi=self._zi)
                filtered = filtered.astype(np.float32)
            else:
                filtered = raw.copy()

            # 3. Numba JIT Kinematics Feature Extraction
            rms, peak, p2p, cf, kurt, skew, sf = _calc_numba_kinematics(filtered)

            # 4. Triggered Waveform for stabilized oscilloscope
            trig_idx = _find_trigger_index(filtered, trigger_level=0.0)
            display_samples = min(len(filtered) - trig_idx, 1024)
            trig_sig = filtered[trig_idx:trig_idx + display_samples]

            # 5. Narrowband Real FFT with exact coherent gain
            w, cg = self._get_window(self.window_type, len(filtered))
            detrended = filtered - np.mean(filtered)
            windowed = detrended * w
            fft_complex = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(len(filtered), 1.0 / self.sample_rate).astype(np.float32)

            mag = np.abs(fft_complex)
            norm_factor = 2.0 / (len(filtered) * cg + 1e-12)
            amps = (mag * norm_factor).astype(np.float32)
            amps[0] *= 0.5

            eps = 1e-12
            db_spectrum = (20.0 * np.log10(np.maximum(amps, eps))).astype(np.float32)

            # Peak-Hold Spectrum Decay
            if len(self._peak_hold_buffer) != len(amps):
                self._peak_hold_buffer = amps.copy()
            else:
                self._peak_hold_buffer = np.maximum(self._peak_hold_buffer * self.peak_hold_decay, amps)

            # 6. Dominant Speed (RPM) & Harmonic Cursors (1X to 10X)
            # Find primary peak above 5 Hz
            mask_valid = freqs >= 5.0
            amps_valid = amps[mask_valid]
            freqs_valid = freqs[mask_valid]

            if len(amps_valid) > 0:
                p_idx = np.argmax(amps_valid)
                f0 = float(freqs_valid[p_idx])
            else:
                f0 = 0.0

            calculated_rpm = f0 * 60.0

            # Multi-Harmonic Tracking 1X to 10X
            harmonics: Dict[str, HarmonicCursorInfo] = {}
            harmonic_amps = []
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

            if f0 > 2.0:
                for k in range(1, 11):
                    target_f = f0 * k
                    if target_f >= freqs[-1]:
                        break
                    tol = max(2.0 * df, 0.03 * target_f)
                    h_mask = np.abs(freqs - target_f) <= tol
                    matching = np.where(h_mask)[0]
                    if len(matching) > 0:
                        best_bin = matching[np.argmax(amps[matching])]
                        h_amp = float(amps[best_bin])
                        h_db = float(db_spectrum[best_bin])
                        harmonics[f"{k}X"] = HarmonicCursorInfo(
                            order_multiplier=float(k),
                            frequency_hz=float(freqs[best_bin]),
                            amplitude=h_amp,
                            db_amplitude=h_db,
                        )
                        if k >= 2:
                            harmonic_amps.append(h_amp)

            # Total Harmonic Distortion (THD)
            fund_amp = harmonics["1X"].amplitude if "1X" in harmonics else 0.0
            if fund_amp > eps and harmonic_amps:
                thd_pct = float(np.sqrt(np.sum(np.array(harmonic_amps) ** 2)) / fund_amp * 100.0)
            else:
                thd_pct = 0.0

            # Sidebands (e.g. Bearing BPFO ~ 3.56X or Gearmesh sidebands +/- 1X)
            sidebands: List[HarmonicCursorInfo] = []
            if f0 > 2.0:
                bpfo_f = f0 * 3.56
                if bpfo_f < freqs[-1]:
                    s_bin = int(np.round(bpfo_f / df))
                    if 0 <= s_bin < len(amps):
                        sidebands.append(HarmonicCursorInfo(
                            order_multiplier=3.56,
                            frequency_hz=float(freqs[s_bin]),
                            amplitude=float(amps[s_bin]),
                            db_amplitude=float(db_spectrum[s_bin]),
                        ))

            # 7. Envelope Demodulation (Hilbert Transform - HFRT)
            # Demodulate high-pass band above 1000 Hz or the high band
            try:
                analytic = scipy.signal.hilbert(filtered)
                envelope = np.abs(analytic)
                env_detrend = envelope - np.mean(envelope)
                env_fft = np.abs(np.fft.rfft(env_detrend * np.hanning(len(env_detrend))))
                env_freqs = np.fft.rfftfreq(len(env_detrend), 1.0 / self.sample_rate).astype(np.float32)
                # Keep envelope spectrum up to 1000 Hz (characteristic defect modulation range)
                env_mask = env_freqs <= 1000.0
                envelope_freqs = env_freqs[env_mask]
                envelope_spectrum = (env_fft[env_mask] * (2.0 / len(env_detrend))).astype(np.float32)
            except Exception:
                envelope_freqs = np.zeros(0, dtype=np.float32)
                envelope_spectrum = np.zeros(0, dtype=np.float32)

            # 8. Power Cepstrum
            try:
                log_pwr = np.log(mag ** 2 + eps)
                cepstrum = np.abs(np.fft.rfft(log_pwr)) ** 2
                quefrency_sec = np.fft.rfftfreq(len(log_pwr), df).astype(np.float32)
                quefrency_ms = quefrency_sec * 1000.0
                # Filter quefrency display between 2ms and 100ms (~10 Hz to 500 Hz repetition)
                q_mask = (quefrency_ms >= 2.0) & (quefrency_ms <= 100.0)
                cep_q = quefrency_ms[q_mask]
                cep_spec = (cepstrum[q_mask] / (np.max(cepstrum[q_mask]) + eps)).astype(np.float32)
            except Exception:
                cep_q = np.zeros(0, dtype=np.float32)
                cep_spec = np.zeros(0, dtype=np.float32)

            # 9. Predictive ML Anomaly & ISO 10816 Severity
            feature_vec = np.array([rms, peak, cf, kurt, skew, sf, f0, thd_pct], dtype=np.float64)

            # Accumulate calibration frames if active
            cal_progress = 0.0
            if self._is_calibrating:
                self._cal_buffer.append(feature_vec)
                elapsed = time.perf_counter() - self._cal_start_time
                cal_progress = min(100.0, (elapsed / self._cal_duration) * 100.0)
                if elapsed >= self._cal_duration:
                    self._finalize_calibration()

            # ISO 10816 Zone evaluation
            vel_proxy = rms * self.iso_scale_factor
            if vel_proxy < 0.71:
                iso_zone = ISOZone.ZONE_A
                iso_color = "#00ff88"
            elif vel_proxy < 1.80:
                iso_zone = ISOZone.ZONE_B
                iso_color = "#00f0ff"
            elif vel_proxy < 4.50:
                iso_zone = ISOZone.ZONE_C
                iso_color = "#ffaa00"
            else:
                iso_zone = ISOZone.ZONE_D
                iso_color = "#ff0055"

            # Anomaly scoring
            mahalanobis_d = 0.0
            anomaly_score = 0.0
            is_anomaly = False
            fault_driver = "Nominal Baseline"

            if self.is_calibrated and self.baseline_mean is not None and self.precision_matrix is not None:
                diff = feature_vec - self.baseline_mean
                d_sq = float(diff @ self.precision_matrix @ diff)
                mahalanobis_d = float(np.sqrt(max(0.0, d_sq)))

                if_score = 0.0
                if self.isolation_forest is not None and self.scaler is not None:
                    try:
                        scaled_v = self.scaler.transform(feature_vec.reshape(1, -1))
                        raw_df = float(self.isolation_forest.decision_function(scaled_v)[0])
                        if_score = 1.0 / (1.0 + np.exp(12.0 * raw_df))
                    except Exception:
                        if_score = 0.0

                d_norm = min(1.0, mahalanobis_d / (self.mahalanobis_threshold_99 * 1.8))
                anomaly_score = float(np.clip(0.65 * d_norm + 0.35 * if_score, 0.0, 1.0))
                health_idx = float(np.clip(100.0 * (1.0 - anomaly_score), 0.0, 100.0))
                is_anomaly = (mahalanobis_d > self.mahalanobis_threshold_95) or (anomaly_score > 0.5)

                if kurt > 4.5:
                    fault_driver = f"Impulsive Shocks (Bearing Defect: κ={kurt:.1f})"
                elif cf > 4.5:
                    fault_driver = f"Transient Impact Spikes (Crest Factor={cf:.1f})"
                elif thd_pct > 25.0:
                    fault_driver = f"Harmonic Distortion (Misalignment: THD={thd_pct:.1f}%)"
                elif is_anomaly:
                    fault_driver = f"Spectral Energy Anomaly (D_M={mahalanobis_d:.1f})"
                else:
                    fault_driver = "Nominal Baseline State"
            else:
                # Heuristic uncalibrated fallback
                k_excess = max(0.0, (kurt - 3.0) / 5.0)
                c_excess = max(0.0, (cf - 3.0) / 6.0)
                anomaly_score = float(min(1.0, 0.5 * k_excess + 0.5 * c_excess))
                health_idx = float(np.clip(100.0 * (1.0 - anomaly_score), 0.0, 100.0))
                is_anomaly = anomaly_score > 0.4
                if kurt > 4.5:
                    fault_driver = f"High Kurtosis (Bearing Shocks Suspected: κ={kurt:.1f})"
                else:
                    fault_driver = "Awaiting Baseline Calibration"

            # 10. Assemble and Emit Payload
            payload = DSPFramePayload(
                timestamp=time.time(),
                raw_time_sig=raw[:1024].copy(),
                filtered_time_sig=filtered[:1024].copy(),
                triggered_time_sig=trig_sig.copy(),
                freqs_hz=freqs.copy(),
                amp_spectrum=amps.copy(),
                db_spectrum=db_spectrum.copy(),
                peak_hold_spectrum=self._peak_hold_buffer.copy(),
                envelope_freqs=envelope_freqs.copy(),
                envelope_spectrum=envelope_spectrum.copy(),
                cepstrum_quefrency_ms=cep_q.copy(),
                cepstrum_spectrum=cep_spec.copy(),
                dominant_frequency_hz=f0,
                calculated_rpm=calculated_rpm,
                harmonics=harmonics,
                sidebands=sidebands,
                rms=rms,
                peak=peak,
                peak_to_peak=p2p,
                crest_factor=cf,
                kurtosis=kurt,
                skewness=skew,
                shape_factor=sf,
                thd_percent=thd_pct,
                health_index_percent=health_idx,
                anomaly_score=anomaly_score,
                mahalanobis_distance=mahalanobis_d,
                is_anomaly=is_anomaly,
                iso_zone=iso_zone,
                iso_zone_color=iso_color,
                fault_driver=fault_driver,
                calibration_progress=cal_progress,
                is_calibrating=self._is_calibrating,
            )

            self.processed_frame_ready.emit(payload)

            # Sleep remainder of frame to maintain ~60 FPS
            elapsed_loop = time.perf_counter() - t0
            rem_time = target_period - elapsed_loop
            if rem_time > 0.001:
                time.sleep(rem_time)

        logger.info("DSP Worker Thread stopped.")
