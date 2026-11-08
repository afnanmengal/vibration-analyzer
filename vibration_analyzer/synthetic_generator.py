"""
synthetic_generator.py - Industrial Mechanical Vibro-Acoustic Signal Generator

Simulates real-world mechanical vibration phenomena for validation, benchmarking,
and live demonstrations:
- Rotor Unbalance (1X RPM fundamental surge)
- Shaft Misalignment (2X RPM harmonic dominance)
- Mechanical Looseness (Truncated harmonics & half-order components: 0.5X, 1.5X, 2X, 3X)
- Rolling Element Bearing Faults:
    * BPFO (Ball Pass Frequency Outer race: ~3.56X)
    * BPFI (Ball Pass Frequency Inner race: ~5.43X)
    * Impact trains exciting structural high-frequency resonant ring-downs
- Variable Speed Ramp (Run-up and coast-down spectrogram demonstration)
- Ambient Acoustic Noise floor & 50/60 Hz line hum
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
import numpy as np


@dataclass
class SyntheticMechanicalConfig:
    """Parameters governing mechanical vibration simulation."""
    base_rpm: float = 1800.0  # 30 Hz fundamental
    unbalance_severity: float = 0.35  # 1X amplitude
    misalignment_severity: float = 0.15  # 2X amplitude
    looseness_severity: float = 0.05  # Harmonics 3X, 4X, 5X + 0.5X
    bearing_fault_severity: float = 0.0  # Impulsive BPFO/BPFI shocks (0.0 to 1.0)
    bearing_resonance_hz: float = 2800.0  # Structural resonant frequency
    noise_floor: float = 0.02
    enable_speed_sweep: bool = False
    sweep_rpm_min: float = 1200.0
    sweep_rpm_max: float = 3600.0
    sweep_period_sec: float = 12.0


class SyntheticVibrationGenerator:
    """
    Continuous phase-coherent mechanical vibration synthesizer.
    Maintains internal phase accumulator across audio blocks to eliminate clicks.
    """

    def __init__(self, config: SyntheticMechanicalConfig | None = None):
        self.config = config or SyntheticMechanicalConfig()
        self._phase_1x = 0.0
        self._bearing_phase = 0.0
        self._sweep_time = 0.0
        self._lock = threading.Lock()

    def update_config(self, **kwargs) -> None:
        """Dynamically adjust parameters in real time."""
        with self._lock:
            for key, val in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, val)

    def generate_block(self, block_size: int, sample_rate: int) -> np.ndarray:
        """Generate a contiguous block of vibro-acoustic samples."""
        with self._lock:
            sr = float(sample_rate)
            bs = int(block_size)
            dt = 1.0 / sr

            # Dynamic RPM calculation (static or smooth sinusoidal sweep)
            if self.config.enable_speed_sweep:
                # Oscillating speed sweep for run-up / coast-down visualization
                sweep_freq = 1.0 / max(1.0, self.config.sweep_period_sec)
                t_arr = self._sweep_time + np.arange(bs) * dt
                self._sweep_time = (self._sweep_time + bs * dt) % self.config.sweep_period_sec
                rpm_center = 0.5 * (self.config.sweep_rpm_max + self.config.sweep_rpm_min)
                rpm_span = 0.5 * (self.config.sweep_rpm_max - self.config.sweep_rpm_min)
                current_rpm = rpm_center + rpm_span * np.sin(2 * np.pi * sweep_freq * t_arr)
            else:
                current_rpm = np.full(bs, self.config.base_rpm)

            # Instantaneous fundamental frequency in Hz (f0 = RPM / 60)
            f0_arr = current_rpm / 60.0

            # Discrete phase integration
            delta_phase = 2 * np.pi * f0_arr * dt
            phases = self._phase_1x + np.cumsum(delta_phase)
            self._phase_1x = phases[-1] % (2 * np.pi)

            # 1. Fundamental 1X (Unbalance)
            signal = self.config.unbalance_severity * np.sin(phases)

            # 2. Harmonic 2X (Misalignment)
            if self.config.misalignment_severity > 0:
                signal += self.config.misalignment_severity * np.sin(2.0 * phases + 0.4)

            # 3. Harmonics 3X, 4X, 5X & sub-harmonic 0.5X (Mechanical Looseness)
            if self.config.looseness_severity > 0:
                ls = self.config.looseness_severity
                signal += 0.7 * ls * np.sin(3.0 * phases + 1.1)
                signal += 0.4 * ls * np.sin(4.0 * phases + 2.3)
                signal += 0.3 * ls * np.sin(5.0 * phases + 0.8)
                signal += 0.5 * ls * np.sin(0.5 * phases)  # Half-order subharmonic

            # 4. Electrical Grid Hum (60 Hz or 120 Hz)
            signal += 0.04 * np.sin(2 * np.pi * 120.0 * np.arange(bs) * dt)

            # 5. Rolling Element Bearing Defect (BPFO impact shock wave train)
            if self.config.bearing_fault_severity > 0:
                bpfo_multiplier = 3.56  # Typical ball-pass outer race coefficient
                bpfo_freq = f0_arr * bpfo_multiplier
                delta_bearing = 2 * np.pi * bpfo_freq * dt
                bearing_phases = self._bearing_phase + np.cumsum(delta_bearing)
                self._bearing_phase = bearing_phases[-1] % (2 * np.pi)

                # Generate sharp impulse triggers when phase wraps around 2*pi
                wrap_triggers = np.diff(bearing_phases // (2 * np.pi), prepend=bearing_phases[0] // (2 * np.pi)) > 0
                impact_signal = np.zeros(bs, dtype=np.float32)

                # Excite damped exponential ringing at structural resonance frequency
                f_res = self.config.bearing_resonance_hz
                decay_rate = 350.0  # 1/sec
                impulse_indices = np.where(wrap_triggers)[0]

                for idx in impulse_indices:
                    rem_samples = bs - idx
                    ring_len = min(rem_samples, int(0.015 * sr))  # ~15ms ringdown
                    if ring_len > 0:
                        t_ring = np.arange(ring_len) * dt
                        decay = np.exp(-decay_rate * t_ring)
                        ring = decay * np.sin(2 * np.pi * f_res * t_ring)
                        impact_signal[idx:idx + ring_len] += ring

                signal += self.config.bearing_fault_severity * impact_signal

            # 6. Ambient acoustic noise floor
            if self.config.noise_floor > 0:
                signal += self.config.noise_floor * np.random.randn(bs)

            return signal.astype(np.float32)
