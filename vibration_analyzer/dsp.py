"""
dsp.py - Real-Time Digital Signal Processing (DSP) & Vibration Metric Extraction Engine

Features:
- Windowing functions with mathematically exact coherent & power gain normalization.
- Fast Fourier Transform (FFT) computing Single-Sided Amplitude & dB Spectra.
- Comprehensive Time-Domain extraction (RMS, Peak-to-Peak, Crest Factor, Kurtosis, Skewness, Shape Factor).
- High-resolution Frequency-Domain extraction (Sub-bin parabolic peak interpolation, top-5 peaks,
  harmonic orders 1X-5X, THD, Spectral Centroid, Spread, Flatness, Rolloff).
- Real-time stateful Butterworth Bandpass Filter using Second-Order Sections (SOS) for glitch-free frame boundaries.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal


class WindowType(enum.Enum):
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN_HARRIS = "blackman_harris"
    FLATTOP = "flattop"
    RECTANGULAR = "rectangular"


@dataclass
class WindowCache:
    """Precomputed window coefficients and gain normalizers."""
    window: np.ndarray
    coherent_gain: float
    power_gain: float


@dataclass
class PeakInfo:
    """Detected spectral peak metadata."""
    frequency_hz: float
    amplitude: float
    bin_index: int
    harmonic_order: Optional[float] = None  # e.g., 1.0 for 1X, 2.0 for 2X


@dataclass
class VibrationFeatures:
    """Consolidated vibro-acoustic feature payload for a single frame."""
    # Time-Domain
    rms: float = 0.0
    peak_to_peak: float = 0.0
    peak: float = 0.0
    crest_factor: float = 1.0
    kurtosis: float = 3.0
    skewness: float = 0.0
    shape_factor: float = 1.0
    impulse_factor: float = 1.0
    margin_factor: float = 1.0

    # Frequency-Domain
    dominant_frequency_hz: float = 0.0
    calculated_rpm: float = 0.0
    spectral_centroid_hz: float = 0.0
    spectral_rolloff_hz: float = 0.0
    spectral_spread_hz: float = 0.0
    spectral_flatness: float = 0.0
    thd_percent: float = 0.0

    # Peak & Harmonic Spectrum
    top_peaks: List[PeakInfo] = field(default_factory=list)
    harmonics: Dict[str, PeakInfo] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert primary diagnostic scalar metrics into a 1D vector
        for anomaly detection models (Isolation Forest, Mahalanobis).
        """
        return np.array([
            self.rms,
            self.peak,
            self.crest_factor,
            self.kurtosis,
            self.skewness,
            self.shape_factor,
            self.dominant_frequency_hz,
            self.spectral_centroid_hz,
            self.thd_percent,
            self.spectral_flatness,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "RMS",
            "Peak",
            "Crest Factor",
            "Kurtosis",
            "Skewness",
            "Shape Factor",
            "Dominant Freq (Hz)",
            "Spectral Centroid (Hz)",
            "THD (%)",
            "Spectral Flatness",
        ]


class RealtimeBandpassFilter:
    """
    Real-time Butterworth filter using Second-Order Sections (SOS).
    Maintains internal delay registers (`zi`) to ensure continuous, glitch-free filtering
    across audio frame boundaries.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        low_cutoff_hz: float = 10.0,
        high_cutoff_hz: float = 2000.0,
        order: int = 4,
        enabled: bool = True,
    ):
        self.sample_rate = float(sample_rate)
        self.low_cutoff = max(1.0, float(low_cutoff_hz))
        self.high_cutoff = min(float(high_cutoff_hz), (self.sample_rate / 2.0) - 100.0)
        self.order = int(order)
        self.enabled = enabled
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None
        self._rebuild_filter()

    def update_params(
        self,
        low_cutoff_hz: Optional[float] = None,
        high_cutoff_hz: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Dynamically update filter cutoffs or bypass state."""
        changed = False
        if low_cutoff_hz is not None and abs(low_cutoff_hz - self.low_cutoff) > 0.1:
            self.low_cutoff = max(1.0, float(low_cutoff_hz))
            changed = True
        if high_cutoff_hz is not None and abs(high_cutoff_hz - self.high_cutoff) > 0.1:
            max_nyq = (self.sample_rate / 2.0) - 50.0
            self.high_cutoff = min(max(self.low_cutoff + 10.0, float(high_cutoff_hz)), max_nyq)
            changed = True
        if enabled is not None:
            self.enabled = enabled

        if changed:
            self._rebuild_filter()

    def _rebuild_filter(self) -> None:
        """Compute SOS matrix and initial conditions."""
        nyquist = self.sample_rate / 2.0
        low_norm = self.low_cutoff / nyquist
        high_norm = self.high_cutoff / nyquist

        # Guard against invalid normalized frequency bounds
        low_norm = max(0.0001, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.001, min(high_norm, 0.999))

        try:
            self.sos = scipy.signal.butter(
                self.order,
                [low_norm, high_norm],
                btype="bandpass",
                output="sos",
            )
            self.zi = scipy.signal.sosfilt_zi(self.sos)
        except Exception as e:
            # Fallback to highpass if bandpass fails
            self.sos = scipy.signal.butter(
                2,
                low_norm,
                btype="highpass",
                output="sos",
            )
            self.zi = scipy.signal.sosfilt_zi(self.sos)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter contiguous input block maintaining continuity."""
        if not self.enabled or self.sos is None:
            return x

        # Ensure correct state shape matching input
        if self.zi is None or self.zi.shape[0] != self.sos.shape[0]:
            self.zi = scipy.signal.sosfilt_zi(self.sos) * x[0]

        filtered, self.zi = scipy.signal.sosfilt(self.sos, x, zi=self.zi)
        return filtered.astype(x.dtype)

    def reset(self) -> None:
        """Reset internal filter delay state."""
        if self.sos is not None:
            self.zi = scipy.signal.sosfilt_zi(self.sos)


class DSPOptimizer:
    """
    High-performance Digital Signal Processing engine.
    Computes windowed FFT, harmonic breakdown, and time/frequency vibration features.
    """

    def __init__(self, sample_rate: int = 44100, block_size: int = 4096):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self._window_cache: Dict[Tuple[str, int], WindowCache] = {}
        self.filter = RealtimeBandpassFilter(
            sample_rate=self.sample_rate,
            low_cutoff_hz=10.0,
            high_cutoff_hz=3000.0,
            enabled=True,
        )

    def set_sample_rate(self, sample_rate: int) -> None:
        """Update sample rate and rebuild filter."""
        self.sample_rate = int(sample_rate)
        self.filter = RealtimeBandpassFilter(
            sample_rate=self.sample_rate,
            low_cutoff_hz=self.filter.low_cutoff,
            high_cutoff_hz=self.filter.high_cutoff,
            enabled=self.filter.enabled,
        )
        self._window_cache.clear()

    def get_window(self, window_type: WindowType, n: int) -> WindowCache:
        """Retrieve precomputed window vector with coherent and power gain factors."""
        cache_key = (window_type.value, n)
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]

        if window_type == WindowType.HANN:
            w = np.hanning(n)
        elif window_type == WindowType.HAMMING:
            w = np.hamming(n)
        elif window_type == WindowType.BLACKMAN_HARRIS:
            w = scipy.signal.windows.blackmanharris(n)
        elif window_type == WindowType.FLATTOP:
            w = scipy.signal.windows.flattop(n)
        elif window_type == WindowType.RECTANGULAR:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.hanning(n)

        coherent_gain = float(np.mean(w))
        power_gain = float(np.sqrt(np.mean(w ** 2)))

        wc = WindowCache(window=w.astype(np.float64), coherent_gain=coherent_gain, power_gain=power_gain)
        self._window_cache[cache_key] = wc
        return wc

    def compute_fft(
        self,
        signal: np.ndarray,
        window_type: WindowType = WindowType.BLACKMAN_HARRIS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Single-Sided Amplitude Spectrum and dB Spectrum.
        Returns:
            frequencies_hz: 1D array of frequency bins [0, fs/2]
            amplitude_spectrum: 1D array of linear amplitudes
            db_spectrum: 1D array of amplitudes in decibels (dB ref 1.0)
        """
        n = len(signal)
        # Detrend signal (remove DC bias before windowing)
        detrended = signal - np.mean(signal)

        # Apply window
        w_cache = self.get_window(window_type, n)
        windowed = detrended * w_cache.window

        # Single-sided real FFT
        fft_complex = np.fft.rfft(windowed)
        frequencies_hz = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        # Magnitude with exact coherent gain compensation
        # Single-sided scaling: factor of 2 for all positive frequencies, factor of 1 for DC/Nyquist
        cg = max(1e-12, w_cache.coherent_gain * n)
        magnitude = np.abs(fft_complex)
        amplitude_spectrum = (2.0 * magnitude) / cg
        amplitude_spectrum[0] = magnitude[0] / cg
        if n % 2 == 0 and len(amplitude_spectrum) > 1:
            amplitude_spectrum[-1] = magnitude[-1] / cg

        # dB Spectrum relative to 1.0 (arbitrary engineering reference)
        eps = 1e-12
        db_spectrum = 20.0 * np.log10(np.maximum(amplitude_spectrum, eps))

        return frequencies_hz, amplitude_spectrum, db_spectrum

    @staticmethod
    def extract_time_features(signal: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive time-domain statistical and kinematic metrics:
        RMS, Peak, Peak-to-Peak, Crest Factor, Kurtosis, Skewness, Shape Factor,
        Impulse Factor, Margin Factor.
        """
        n = len(signal)
        if n == 0:
            return {
                "rms": 0.0, "peak": 0.0, "peak_to_peak": 0.0, "crest_factor": 1.0,
                "kurtosis": 3.0, "skewness": 0.0, "shape_factor": 1.0,
                "impulse_factor": 1.0, "margin_factor": 1.0,
            }

        eps = 1e-12
        abs_sig = np.abs(signal)
        peak = float(np.max(abs_sig))
        peak_to_peak = float(np.max(signal) - np.min(signal))

        # RMS (Root Mean Square)
        rms = float(np.sqrt(np.mean(signal ** 2)))
        crest_factor = float(peak / (rms + eps))

        # Central statistical moments
        mean_val = float(np.mean(signal))
        std_val = float(np.std(signal))

        if std_val > eps:
            normalized = (signal - mean_val) / std_val
            skewness = float(np.mean(normalized ** 3))
            kurtosis = float(np.mean(normalized ** 4))
        else:
            skewness = 0.0
            kurtosis = 3.0  # Gaussian nominal baseline

        # Shape Factor & Impulse Factor
        mean_abs = float(np.mean(abs_sig))
        shape_factor = float(rms / (mean_abs + eps))
        impulse_factor = float(peak / (mean_abs + eps))

        # Margin Factor
        mean_sqrt_abs = float(np.mean(np.sqrt(abs_sig)))
        margin_factor = float(peak / ((mean_sqrt_abs ** 2) + eps))

        return {
            "rms": rms,
            "peak": peak,
            "peak_to_peak": peak_to_peak,
            "crest_factor": crest_factor,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "shape_factor": shape_factor,
            "impulse_factor": impulse_factor,
            "margin_factor": margin_factor,
        }

    def extract_frequency_features(
        self,
        frequencies_hz: np.ndarray,
        amplitude_spectrum: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, any]:
        """
        Calculate frequency-domain metrics:
        - Top-K Peaks with parabolic sub-bin frequency interpolation
        - Dominant Shaft Speed (RPM)
        - Harmonic identification (1X, 2X, 3X, 4X, 5X)
        - Total Harmonic Distortion (THD)
        - Spectral Centroid, Rolloff, Spread, and Flatness
        """
        eps = 1e-12
        total_amp = np.sum(amplitude_spectrum) + eps

        # 1. Spectral Centroid
        centroid = float(np.sum(frequencies_hz * amplitude_spectrum) / total_amp)

        # 2. Spectral Spread (Standard deviation around centroid)
        spread = float(np.sqrt(np.sum(((frequencies_hz - centroid) ** 2) * amplitude_spectrum) / total_amp))

        # 3. Spectral Rolloff (85% energy frequency)
        cum_amp = np.cumsum(amplitude_spectrum)
        rolloff_idx = np.searchsorted(cum_amp, 0.85 * cum_amp[-1])
        rolloff = float(frequencies_hz[min(rolloff_idx, len(frequencies_hz) - 1)])

        # 4. Spectral Flatness (Wiener Entropy: geometric mean / arithmetic mean)
        power_spectrum = amplitude_spectrum ** 2 + eps
        log_mean = np.mean(np.log(power_spectrum))
        geom_mean = np.exp(log_mean)
        arith_mean = np.mean(power_spectrum)
        flatness = float(geom_mean / arith_mean)

        # 5. Peak Finding with prominence threshold
        min_height = max(1e-5, 0.05 * np.max(amplitude_spectrum))
        peak_indices, props = scipy.signal.find_peaks(
            amplitude_spectrum,
            height=min_height,
            distance=max(3, int(0.005 * len(amplitude_spectrum))),
            prominence=min_height * 0.5,
        )

        peaks: List[PeakInfo] = []
        df = frequencies_hz[1] - frequencies_hz[0] if len(frequencies_hz) > 1 else 1.0

        if len(peak_indices) > 0:
            # Sort peaks by amplitude descending
            sorted_by_amp = peak_indices[np.argsort(amplitude_spectrum[peak_indices])[::-1]]
            top_indices = sorted_by_amp[:top_k]

            for idx in top_indices:
                # Sub-bin parabolic interpolation for precision
                if 0 < idx < len(amplitude_spectrum) - 1:
                    alpha = amplitude_spectrum[idx - 1]
                    beta = amplitude_spectrum[idx]
                    gamma = amplitude_spectrum[idx + 1]
                    denom = alpha - 2 * beta + gamma
                    delta_p = 0.5 * (alpha - gamma) / (denom - eps) if abs(denom) > 1e-9 else 0.0
                    sub_bin = idx + delta_p
                    peak_freq = float(sub_bin * df)
                    peak_amp = float(beta - 0.25 * (alpha - gamma) * delta_p)
                else:
                    peak_freq = float(frequencies_hz[idx])
                    peak_amp = float(amplitude_spectrum[idx])

                peaks.append(PeakInfo(
                    frequency_hz=peak_freq,
                    amplitude=peak_amp,
                    bin_index=idx,
                ))

        # Dominant fundamental peak
        dominant_freq = peaks[0].frequency_hz if peaks else 0.0
        calculated_rpm = dominant_freq * 60.0

        # 6. Harmonic Tracking (1X, 2X, 3X, 4X, 5X)
        harmonics: Dict[str, PeakInfo] = {}
        thd_percent = 0.0

        if dominant_freq > 2.0:  # Only evaluate harmonics above 2 Hz
            harmonics["1X"] = peaks[0]
            peaks[0].harmonic_order = 1.0
            fundamental_amp = peaks[0].amplitude

            harmonic_amps = []
            for order in [2, 3, 4, 5]:
                target_f = dominant_freq * order
                if target_f >= frequencies_hz[-1]:
                    break

                # Tolerance window (+/- 4% or 2 frequency bins)
                tol = max(2.0 * df, 0.04 * target_f)
                mask = np.abs(frequencies_hz - target_f) <= tol
                matching_bins = np.where(mask)[0]

                if len(matching_bins) > 0:
                    best_bin = matching_bins[np.argmax(amplitude_spectrum[matching_bins])]
                    h_amp = float(amplitude_spectrum[best_bin])
                    h_freq = float(frequencies_hz[best_bin])
                    p_info = PeakInfo(
                        frequency_hz=h_freq,
                        amplitude=h_amp,
                        bin_index=best_bin,
                        harmonic_order=float(order),
                    )
                    harmonics[f"{order}X"] = p_info
                    harmonic_amps.append(h_amp)

            # Total Harmonic Distortion (THD)
            if fundamental_amp > eps and harmonic_amps:
                sum_sq_harmonics = np.sum(np.array(harmonic_amps) ** 2)
                thd_percent = float(np.sqrt(sum_sq_harmonics) / fundamental_amp * 100.0)

        return {
            "dominant_frequency_hz": dominant_freq,
            "calculated_rpm": calculated_rpm,
            "spectral_centroid_hz": centroid,
            "spectral_rolloff_hz": rolloff,
            "spectral_spread_hz": spread,
            "spectral_flatness": flatness,
            "thd_percent": thd_percent,
            "top_peaks": peaks,
            "harmonics": harmonics,
        }

    def process_frame(
        self,
        raw_signal: np.ndarray,
        window_type: WindowType = WindowType.BLACKMAN_HARRIS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, VibrationFeatures]:
        """
        Complete end-to-end DSP pipeline for a single audio frame:
        1. Stateful Butterworth Bandpass Filtering
        2. Time-Domain Vibration Metrics Extraction
        3. Windowed Fast Fourier Transform
        4. Frequency-Domain & Harmonic Extraction
        Returns:
            (filtered_signal, frequencies_hz, amplitude_spectrum, features)
        """
        # Step 1: Filter raw signal
        filtered = self.filter.process(raw_signal)

        # Step 2: Time-domain extraction
        time_feats = self.extract_time_features(filtered)

        # Step 3: Fast Fourier Transform
        frequencies_hz, amp_spectrum, _ = self.compute_fft(filtered, window_type=window_type)

        # Step 4: Frequency-domain extraction
        freq_feats = self.extract_frequency_features(frequencies_hz, amp_spectrum)

        # Assemble unified feature dataclass
        features = VibrationFeatures(
            rms=time_feats["rms"],
            peak_to_peak=time_feats["peak_to_peak"],
            peak=time_feats["peak"],
            crest_factor=time_feats["crest_factor"],
            kurtosis=time_feats["kurtosis"],
            skewness=time_feats["skewness"],
            shape_factor=time_feats["shape_factor"],
            impulse_factor=time_feats["impulse_factor"],
            margin_factor=time_feats["margin_factor"],
            dominant_frequency_hz=freq_feats["dominant_frequency_hz"],
            calculated_rpm=freq_feats["calculated_rpm"],
            spectral_centroid_hz=freq_feats["spectral_centroid_hz"],
            spectral_rolloff_hz=freq_feats["spectral_rolloff_hz"],
            spectral_spread_hz=freq_feats["spectral_spread_hz"],
            spectral_flatness=freq_feats["spectral_flatness"],
            thd_percent=freq_feats["thd_percent"],
            top_peaks=freq_feats["top_peaks"],
            harmonics=freq_feats["harmonics"],
        )

        return filtered, frequencies_hz, amp_spectrum, features
