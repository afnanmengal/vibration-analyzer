"""
detector.py - Predictive Analytics, Machine Calibration & Mechanical Anomaly Detection Engine

Features:
- Automated 10-second Baseline Calibration routine.
- Dual-Mode Anomaly Scoring:
    1. Regularized Mahalanobis Distance over multi-dimensional vibro-acoustic feature space.
    2. Scikit-Learn Isolation Forest / One-Class outlier classification.
- Machine Health Index (0% to 100%) computation.
- Primary Anomaly Driver Attribution (identifies which specific metric e.g., Kurtosis, Crest Factor, RMS, THD triggered the anomaly).
- ISO 10816 / ISO 20816 Vibration Severity Zone mapping (Class A, B, C, D).
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .dsp import VibrationFeatures

logger = logging.getLogger("vibration_analyzer.detector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Detector] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ISOZone(enum.Enum):
    ZONE_A = "Zone A (Good)"
    ZONE_B = "Zone B (Acceptable)"
    ZONE_C = "Zone C (Alert / Unsatisfactory)"
    ZONE_D = "Zone D (Danger / Shutdown)"


class MachineClass(enum.Enum):
    CLASS_I = "Class I: Small Machines (< 15 kW / Laptops / Auxiliary Fans)"
    CLASS_II = "Class II: Medium Machines (15 kW - 75 kW)"
    CLASS_III = "Class III: Large Rigid Foundation (> 75 kW)"
    CLASS_IV = "Class IV: Large Flexible Foundation"


@dataclass
class AnomalyReport:
    """Consolidated anomaly and health assessment for the current time frame."""
    mahalanobis_distance: float = 0.0
    isolation_forest_score: float = 0.0
    anomaly_score: float = 0.0  # 0.0 (pristine) to 1.0 (severe anomaly)
    health_index_percent: float = 100.0  # 0% to 100%
    is_anomaly: bool = False
    iso_zone: ISOZone = ISOZone.ZONE_A
    iso_zone_color: str = "#00ff88"  # Green
    primary_fault_driver: str = "Nominal"
    fault_confidence: float = 0.0
    calibrated: bool = False


class AnomalyDetector:
    """
    Real-time mechanical health and anomaly detection engine.
    Supports continuous calibration and dual statistical/machine learning scoring.
    """

    # ISO 10816-1 RMS thresholds (in velocity mm/s or calibrated relative units)
    # [A/B boundary, B/C boundary, C/D boundary]
    ISO_THRESHOLDS = {
        MachineClass.CLASS_I: (0.71, 1.80, 4.50),
        MachineClass.CLASS_II: (1.12, 2.80, 7.10),
        MachineClass.CLASS_III: (1.80, 4.50, 11.20),
        MachineClass.CLASS_IV: (2.80, 7.10, 18.00),
    }

    def __init__(
        self,
        machine_class: MachineClass = MachineClass.CLASS_I,
        iso_scale_factor: float = 50.0,  # Multiplier converting raw mic RMS to acoustic mm/s equiv
    ):
        self.machine_class = machine_class
        self.iso_scale_factor = iso_scale_factor

        # Calibration state
        self._is_calibrating = False
        self._calibration_start_time = 0.0
        self._calibration_duration = 10.0  # seconds
        self._calibration_buffer: List[np.ndarray] = []
        self.is_calibrated = False

        # Statistical baseline parameters
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.precision_matrix: Optional[np.ndarray] = None  # Inverse covariance
        self.mahalanobis_threshold_95: float = 3.0
        self.mahalanobis_threshold_99: float = 5.0

        # Scikit-learn model
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None

        self._lock = threading.Lock()

    def start_calibration(self, duration_sec: float = 10.0) -> None:
        """Trigger baseline calibration routine for normal machinery operating state."""
        with self._lock:
            self._is_calibrating = True
            self._calibration_duration = max(3.0, float(duration_sec))
            self._calibration_start_time = time.perf_counter()
            self._calibration_buffer.clear()
            logger.info(f"Calibration initiated for {self._calibration_duration:.1f} seconds...")

    def is_calibrating(self) -> Tuple[bool, float]:
        """
        Check calibration status.
        Returns (is_active, progress_percentage_0_to_100).
        """
        with self._lock:
            if not self._is_calibrating:
                return False, 100.0 if self.is_calibrated else 0.0

            elapsed = time.perf_counter() - self._calibration_start_time
            progress = min(100.0, (elapsed / self._calibration_duration) * 100.0)
            return True, progress

    def add_calibration_frame(self, features: VibrationFeatures) -> None:
        """Accumulate feature vector during active calibration phase."""
        with self._lock:
            if not self._is_calibrating:
                return

            vec = features.to_feature_vector()
            self._calibration_buffer.append(vec)

            elapsed = time.perf_counter() - self._calibration_start_time
            if elapsed >= self._calibration_duration:
                self._finalize_calibration()

    def _finalize_calibration(self) -> None:
        """Compute baseline statistical matrices and fit Isolation Forest."""
        self._is_calibrating = False
        num_samples = len(self._calibration_buffer)

        if num_samples < 15:
            logger.warning(f"Insufficient calibration samples ({num_samples}). Calibration aborted.")
            return

        data_matrix = np.array(self._calibration_buffer, dtype=np.float64)  # Shape (N, D)

        # Baseline mean & std
        self.baseline_mean = np.mean(data_matrix, axis=0)
        self.baseline_std = np.std(data_matrix, axis=0)
        self.baseline_std[self.baseline_std < 1e-6] = 1e-6

        # Regularized Covariance Matrix & Precision Matrix (Mahalanobis)
        cov_matrix = np.cov(data_matrix, rowvar=False)
        dim = cov_matrix.shape[0]

        # Tikhonov Ridge regularization to guarantee non-singular positive-definite matrix
        ridge = 1e-4 * np.trace(cov_matrix) / max(1, dim) if np.trace(cov_matrix) > 0 else 1e-4
        cov_regularized = cov_matrix + np.eye(dim) * ridge

        try:
            self.precision_matrix = np.linalg.pinv(cov_regularized)
        except Exception as e:
            logger.error(f"Covariance inversion failed: {e}. Using diagonal inverse.")
            self.precision_matrix = np.diag(1.0 / (self.baseline_std ** 2))

        # Calculate baseline Mahalanobis distances to set empirical thresholds
        diffs = data_matrix - self.baseline_mean
        d_sq = np.sum(diffs @ self.precision_matrix * diffs, axis=1)
        distances = np.sqrt(np.maximum(0.0, d_sq))

        self.mahalanobis_threshold_95 = float(np.percentile(distances, 95))
        self.mahalanobis_threshold_99 = float(np.percentile(distances, 99))
        if self.mahalanobis_threshold_99 <= self.mahalanobis_threshold_95:
            self.mahalanobis_threshold_99 = self.mahalanobis_threshold_95 * 1.5

        # Fit Isolation Forest model
        try:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(data_matrix)
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.03,
                random_state=42,
            )
            self.isolation_forest.fit(scaled_data)
        except Exception as err:
            logger.error(f"Failed to fit Isolation Forest: {err}")
            self.isolation_forest = None

        self.is_calibrated = True
        logger.info(
            f"Baseline calibration complete with {num_samples} frames. "
            f"D_M(95%)={self.mahalanobis_threshold_95:.2f}, D_M(99%)={self.mahalanobis_threshold_99:.2f}"
        )

    def reset_calibration(self) -> None:
        """Clear baseline calibration state."""
        with self._lock:
            self.is_calibrated = False
            self._is_calibrating = False
            self.baseline_mean = None
            self.baseline_std = None
            self.precision_matrix = None
            self.isolation_forest = None
            self.scaler = None
            self._calibration_buffer.clear()
            logger.info("Baseline calibration reset.")

    def evaluate(self, features: VibrationFeatures) -> AnomalyReport:
        """
        Evaluate real-time vibration features against calibrated baseline
        and ISO 10816 severity thresholds.
        """
        # If in active calibration phase, feed frame
        cal_active, _ = self.is_calibrating()
        if cal_active:
            self.add_calibration_frame(features)

        # 1. ISO 10816 Zone Classification
        # Convert audio RMS to calibrated mechanical vibration velocity proxy (mm/s)
        calibrated_rms_velocity = features.rms * self.iso_scale_factor
        thresh_ab, thresh_bc, thresh_cd = self.ISO_THRESHOLDS.get(
            self.machine_class,
            (0.71, 1.80, 4.50),
        )

        if calibrated_rms_velocity < thresh_ab:
            iso_zone = ISOZone.ZONE_A
            zone_color = "#00ff88"  # Glowing green (Good)
        elif calibrated_rms_velocity < thresh_bc:
            iso_zone = ISOZone.ZONE_B
            zone_color = "#a3e635"  # Lime yellow (Acceptable)
        elif calibrated_rms_velocity < thresh_cd:
            iso_zone = ISOZone.ZONE_C
            zone_color = "#f59e0b"  # Amber warning (Alert / Unsatisfactory)
        else:
            iso_zone = ISOZone.ZONE_D
            zone_color = "#ef4444"  # Red alert (Danger / Shutdown)

        # 2. Statistical Anomaly & Health Index (if calibrated)
        report = AnomalyReport(
            iso_zone=iso_zone,
            iso_zone_color=zone_color,
            calibrated=self.is_calibrated,
        )

        vec = features.to_feature_vector()
        feature_names = VibrationFeatures.feature_names()

        if self.is_calibrated and self.baseline_mean is not None and self.precision_matrix is not None:
            diff = vec - self.baseline_mean

            # Mahalanobis Distance: sqrt( (x - mu)^T * Sigma^-1 * (x - mu) )
            d_sq = float(diff @ self.precision_matrix @ diff)
            d_m = float(np.sqrt(max(0.0, d_sq)))
            report.mahalanobis_distance = d_m

            # Isolation Forest scoring
            if_score = 0.0
            if self.isolation_forest is not None and self.scaler is not None:
                try:
                    scaled_vec = self.scaler.transform(vec.reshape(1, -1))
                    # decision_function: positive is inlier, negative is outlier
                    raw_df = float(self.isolation_forest.decision_function(scaled_vec)[0])
                    # Sigmoidal mapping to [0, 1] outlier confidence
                    if_score = 1.0 / (1.0 + np.exp(12.0 * raw_df))
                except Exception:
                    if_score = 0.0
            report.isolation_forest_score = if_score

            # Normalized combined anomaly score (0.0 = completely healthy, 1.0 = extreme anomaly)
            d_norm = min(1.0, d_m / (self.mahalanobis_threshold_99 * 1.8))
            combined_score = float(np.clip(0.65 * d_norm + 0.35 * if_score, 0.0, 1.0))
            report.anomaly_score = combined_score
            report.health_index_percent = float(np.clip(100.0 * (1.0 - combined_score), 0.0, 100.0))
            report.is_anomaly = (d_m > self.mahalanobis_threshold_95) or (combined_score > 0.55)

            # 3. Attribution of primary fault driver
            # Prioritize definitive mechanical diagnostic indicators first
            if features.kurtosis > 4.5:
                report.primary_fault_driver = f"Impulsive Shocks (Bearing / Impact Spall: Kurtosis={features.kurtosis:.1f})"
                report.fault_confidence = min(1.0, (features.kurtosis - 3.0) / 4.0)
            elif features.crest_factor > 4.5:
                report.primary_fault_driver = f"Transient Impacting (Crest Factor={features.crest_factor:.1f})"
                report.fault_confidence = min(1.0, (features.crest_factor - 2.5) / 3.0)
            elif features.thd_percent > 25.0:
                report.primary_fault_driver = f"Harmonic Distortion (Misalignment/Looseness: THD={features.thd_percent:.1f}%)"
                report.fault_confidence = min(1.0, features.thd_percent / 50.0)
            else:
                # Standardized z-score deviation across feature space
                z_scores = np.abs(diff / self.baseline_std)
                top_feature_idx = int(np.argmax(z_scores))
                top_z = float(z_scores[top_feature_idx])

                if top_z > 2.0:
                    top_name = feature_names[top_feature_idx]
                    report.primary_fault_driver = f"Elevated {top_name} (Z={top_z:.1f}σ)"
                    report.fault_confidence = float(min(1.0, top_z / 6.0))
                else:
                    report.primary_fault_driver = "Nominal Baseline State"
                    report.fault_confidence = 0.0

        else:
            # Fallback heuristic when uncalibrated: use standard engineering rules of thumb
            # Kurtosis > 4.5 indicates bearing spall/impacts (Gaussian is 3.0)
            # Crest factor > 5.0 indicates peak shock transients
            kurt_excess = max(0.0, (features.kurtosis - 3.0) / 5.0)
            cf_excess = max(0.0, (features.crest_factor - 3.0) / 6.0)
            heuristic_anomaly = min(1.0, 0.5 * kurt_excess + 0.5 * cf_excess)

            report.anomaly_score = float(heuristic_anomaly)
            report.health_index_percent = float(np.clip(100.0 * (1.0 - heuristic_anomaly), 0.0, 100.0))
            report.is_anomaly = heuristic_anomaly > 0.4

            if features.kurtosis > 5.0:
                report.primary_fault_driver = "High Kurtosis (Bearing / Impact Suspected)"
            elif features.crest_factor > 6.0:
                report.primary_fault_driver = "High Crest Factor (Transient Spikes)"
            else:
                report.primary_fault_driver = "Awaiting Calibration"

        return report
