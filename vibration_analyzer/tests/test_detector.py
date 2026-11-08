"""
test_detector.py - Unit & Integration Tests for Anomaly Detection, Calibration, and ISO Severity
"""

import numpy as np
import pytest
from vibration_analyzer.detector import AnomalyDetector, ISOZone, MachineClass
from vibration_analyzer.dsp import VibrationFeatures


def test_iso_severity_mapping():
    detector = AnomalyDetector(machine_class=MachineClass.CLASS_I, iso_scale_factor=50.0)

    # Class I thresholds:
    # A/B = 0.71 mm/s, B/C = 1.80 mm/s, C/D = 4.50 mm/s

    # Small vibration: RMS = 0.01 -> 0.01 * 50 = 0.5 mm/s -> Zone A (Good)
    feat_a = VibrationFeatures(rms=0.01)
    rep_a = detector.evaluate(feat_a)
    assert rep_a.iso_zone == ISOZone.ZONE_A

    # Moderate vibration: RMS = 0.025 -> 0.025 * 50 = 1.25 mm/s -> Zone B (Acceptable)
    feat_b = VibrationFeatures(rms=0.025)
    rep_b = detector.evaluate(feat_b)
    assert rep_b.iso_zone == ISOZone.ZONE_B

    # Elevated vibration: RMS = 0.05 -> 0.05 * 50 = 2.5 mm/s -> Zone C (Alert)
    feat_c = VibrationFeatures(rms=0.05)
    rep_c = detector.evaluate(feat_c)
    assert rep_c.iso_zone == ISOZone.ZONE_C

    # Severe vibration: RMS = 0.15 -> 0.15 * 50 = 7.5 mm/s -> Zone D (Danger)
    feat_d = VibrationFeatures(rms=0.15)
    rep_d = detector.evaluate(feat_d)
    assert rep_d.iso_zone == ISOZone.ZONE_D


def test_calibration_and_anomaly_detection():
    detector = AnomalyDetector(machine_class=MachineClass.CLASS_I)
    detector.start_calibration(duration_sec=0.2)  # Short calibration duration for testing

    # Feed 25 baseline frames with small variance around nominal healthy state
    np.random.seed(42)
    for _ in range(25):
        feat = VibrationFeatures(
            rms=0.03 + np.random.normal(0, 0.001),
            peak=0.05 + np.random.normal(0, 0.002),
            crest_factor=2.5 + np.random.normal(0, 0.05),
            kurtosis=3.0 + np.random.normal(0, 0.05),
            skewness=0.01 + np.random.normal(0, 0.01),
            shape_factor=1.2 + np.random.normal(0, 0.02),
            dominant_frequency_hz=30.0 + np.random.normal(0, 0.1),
            spectral_centroid_hz=150.0 + np.random.normal(0, 2.0),
            thd_percent=5.0 + np.random.normal(0, 0.2),
            spectral_flatness=0.05 + np.random.normal(0, 0.005),
        )
        detector.evaluate(feat)

    # Force finalize if time hasn't elapsed
    detector._finalize_calibration()
    assert detector.is_calibrated is True
    assert detector.precision_matrix is not None
    assert detector.isolation_forest is not None

    # Test 1: Normal frame should have low Mahalanobis distance and high health index
    normal_feat = VibrationFeatures(
        rms=0.03, peak=0.05, crest_factor=2.5, kurtosis=3.0, skewness=0.01,
        shape_factor=1.2, dominant_frequency_hz=30.0, spectral_centroid_hz=150.0,
        thd_percent=5.0, spectral_flatness=0.05,
    )
    rep_normal = detector.evaluate(normal_feat)
    assert rep_normal.health_index_percent > 70.0
    assert rep_normal.is_anomaly is False

    # Test 2: Extreme bearing anomaly (Kurtosis spiked from 3.0 to 12.0)
    anom_feat = VibrationFeatures(
        rms=0.035, peak=0.25, crest_factor=7.5, kurtosis=12.5, skewness=0.5,
        shape_factor=1.4, dominant_frequency_hz=30.0, spectral_centroid_hz=450.0,
        thd_percent=8.0, spectral_flatness=0.05,
    )
    rep_anom = detector.evaluate(anom_feat)
    assert rep_anom.mahalanobis_distance > detector.mahalanobis_threshold_95
    assert rep_anom.is_anomaly is True
    assert "Bearing" in rep_anom.primary_fault_driver or "Kurtosis" in rep_anom.primary_fault_driver
