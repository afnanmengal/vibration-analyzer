"""
web_app.py - Companion Streamlit Reactive Web Dashboard for Mechanical Vibration Analysis

Provides a browser-accessible dashboard alternative using Plotly and Streamlit.
Shares the exact same DSP (`dsp.py`), Detector (`detector.py`), and Synthetic Rig (`synthetic_generator.py`)
engines with the PyQt6 desktop app.
"""

from __future__ import annotations

import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from .dsp import DSPOptimizer, WindowType
    from .detector import AnomalyDetector, ISOZone, MachineClass
    from .synthetic_generator import SyntheticMechanicalConfig, SyntheticVibrationGenerator
    from .audio_stream import AudioStreamConfig, AudioStreamEngine
except ImportError:
    from dsp import DSPOptimizer, WindowType
    from detector import AnomalyDetector, ISOZone, MachineClass
    from synthetic_generator import SyntheticMechanicalConfig, SyntheticVibrationGenerator
    from audio_stream import AudioStreamConfig, AudioStreamEngine


def init_session_state():
    if "dsp" not in st.session_state:
        st.session_state.dsp = DSPOptimizer(sample_rate=44100, block_size=4096)
    if "detector" not in st.session_state:
        st.session_state.detector = AnomalyDetector(machine_class=MachineClass.CLASS_I, iso_scale_factor=60.0)
    if "synth" not in st.session_state:
        st.session_state.synth = SyntheticVibrationGenerator(SyntheticMechanicalConfig())
    if "spectrogram_history" not in st.session_state:
        st.session_state.spectrogram_history = []


def run_web_app():
    st.set_page_config(
        page_title="Mechanical Vibration & Anomaly Analyzer",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.title("⚙️ Mechanical Vibro-Acoustic Vibration & Anomaly Analyzer")
    st.caption("Real-Time Mechanical Spectral Diagnostics, Harmonic Analysis (1X-5X), and ISO 10816 Severity Classification")

    # --- Sidebar Controls ---
    st.sidebar.header("🔧 Vibration Signal Acquisition")
    source_mode = st.sidebar.selectbox(
        "Signal Source",
        ["Synthetic Industrial Motor Rig", "Hardware Microphone (Default)"],
    )

    st.sidebar.subheader("DSP Filtering & Windowing")
    win_choice = st.sidebar.selectbox(
        "Window Function",
        ["Blackman-Harris (High Dynamic Range)", "Hann", "Hamming", "Flat-Top", "Rectangular"],
    )
    win_map = {
        "Blackman-Harris (High Dynamic Range)": WindowType.BLACKMAN_HARRIS,
        "Hann": WindowType.HANN,
        "Hamming": WindowType.HAMMING,
        "Flat-Top": WindowType.FLATTOP,
        "Rectangular": WindowType.RECTANGULAR,
    }
    selected_window = win_map[win_choice]

    filter_enabled = st.sidebar.checkbox("Enable Butterworth Bandpass Filter", value=True)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        low_cut = st.number_input("High-Pass (Hz)", min_value=1.0, max_value=500.0, value=10.0, step=5.0)
    with c2:
        high_cut = st.number_input("Low-Pass (Hz)", min_value=50.0, max_value=10000.0, value=3000.0, step=50.0)
    st.session_state.dsp.filter.update_params(low_cutoff_hz=low_cut, high_cutoff_hz=high_cut, enabled=filter_enabled)

    st.sidebar.subheader("ISO 10816 Classification")
    mc_choice = st.sidebar.selectbox(
        "Machine Class",
        [m.value for m in MachineClass],
    )
    chosen_class = next(m for m in MachineClass if m.value == mc_choice)
    st.session_state.detector.machine_class = chosen_class

    st.sidebar.subheader("⚙️ Mechanical Fault Injection (Sim Rig)")
    sim_rpm = st.sidebar.slider("Motor Shaft Speed (RPM)", 600, 3600, 1800, step=60)
    sim_unbal = st.sidebar.slider("1X Rotor Unbalance", 0.0, 1.0, 0.35, step=0.05)
    sim_misalign = st.sidebar.slider("2X Shaft Misalignment", 0.0, 1.0, 0.15, step=0.05)
    sim_loose = st.sidebar.slider("Mechanical Looseness", 0.0, 1.0, 0.05, step=0.05)
    sim_bearing = st.sidebar.slider("Bearing Defect (BPFO Impulses)", 0.0, 1.0, 0.0, step=0.05)
    sim_sweep = st.sidebar.checkbox("Dynamic Speed Ramp (Run-up / Coast-down)", value=False)

    st.session_state.synth.update_config(
        base_rpm=sim_rpm,
        unbalance_severity=sim_unbal,
        misalignment_severity=sim_misalign,
        looseness_severity=sim_loose,
        bearing_fault_severity=sim_bearing,
        enable_speed_sweep=sim_sweep,
    )

    # --- Signal Acquisition ---
    sr = 44100
    block_size = 4096

    if source_mode == "Synthetic Industrial Motor Rig":
        audio_chunk = st.session_state.synth.generate_block(block_size, sr)
    else:
        # One-shot hardware capture via sounddevice
        try:
            import sounddevice as sd
            audio_chunk = sd.rec(block_size, samplerate=sr, channels=1, dtype="float32", blocking=True).flatten()
        except Exception as e:
            st.warning(f"Hardware microphone unavailable: {e}. Falling back to synthetic motor rig.")
            audio_chunk = st.session_state.synth.generate_block(block_size, sr)

    # --- DSP Execution ---
    filtered_sig, freqs, amps, features = st.session_state.dsp.process_frame(audio_chunk, selected_window)
    report = st.session_state.detector.evaluate(features)

    # Rolling spectrogram buffer
    db_slice = 20.0 * np.log10(np.maximum(amps, 1e-5))
    st.session_state.spectrogram_history.append(db_slice)
    if len(st.session_state.spectrogram_history) > 40:
        st.session_state.spectrogram_history.pop(0)

    # --- Top Metrics Strip ---
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        st.metric("RMS Amplitude", f"{features.rms:.4f} g", delta=f"Peak {features.peak:.3f}")
    with m2:
        st.metric("Dominant Freq", f"{features.dominant_frequency_hz:.1f} Hz", delta="1X Shaft")
    with m3:
        st.metric("Calculated RPM", f"{int(features.calculated_rpm):,} RPM")
    with m4:
        st.metric("Crest Factor", f"{features.crest_factor:.2f}")
    with m5:
        st.metric("Kurtosis", f"{features.kurtosis:.2f}", delta=f"{'Impulsive' if features.kurtosis > 4.5 else 'Nominal'}")
    with m6:
        st.metric("THD", f"{features.thd_percent:.1f}%")
    with m7:
        st.metric("Machine Health", f"{report.health_index_percent:.0f}%", delta=report.iso_zone.name)

    # Diagnostic alert banner
    if report.is_anomaly or report.iso_zone in (ISOZone.ZONE_C, ISOZone.ZONE_D):
        st.error(f"⚠️ **ANOMALY DETECTED:** {report.primary_fault_driver} | ISO Zone: {report.iso_zone.value}")
    else:
        st.success(f"✅ **NOMINAL OPERATION:** {report.primary_fault_driver} | ISO Zone: {report.iso_zone.value}")

    # Calibration button
    cal_col1, cal_col2 = st.columns([1, 4])
    with cal_col1:
        if st.button("🎯 Calibrate Baseline (10s)"):
            with st.spinner("Recording baseline acoustic fingerprint..."):
                time.sleep(1.0)
                st.session_state.detector.start_calibration(duration_sec=3.0)
                for _ in range(30):
                    sig = st.session_state.synth.generate_block(block_size, sr)
                    _, _, _, feats = st.session_state.dsp.process_frame(sig, selected_window)
                    st.session_state.detector.evaluate(feats)
                st.success("Calibration complete!")
                st.rerun()

    # --- Visual Plots ---
    plot_col1, plot_col2 = st.columns([1, 1])

    with plot_col1:
        # Time-domain Oscilloscope
        time_ms = np.arange(len(filtered_sig)) * (1000.0 / sr)
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=time_ms[::4], y=filtered_sig[::4],
            mode="lines", line=dict(color="#00d2ff", width=1.5),
            name="Raw Waveform",
        ))
        fig_time.update_layout(
            title="Time-Domain Oscilloscope (Filtered Waveform)",
            xaxis_title="Time (ms)",
            yaxis_title="Amplitude (g)",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with plot_col2:
        # FFT Spectrum Analyzer
        max_plot_freq = 2500.0
        mask = freqs <= max_plot_freq
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=freqs[mask], y=amps[mask],
            mode="lines", fill="tozeroy",
            line=dict(color="#38bdf8", width=1.5),
            name="Spectrum",
        ))

        # Harmonic lines
        f0 = features.dominant_frequency_hz
        if f0 > 2.0:
            for k in range(1, 6):
                hk = f0 * k
                if hk <= max_plot_freq:
                    fig_fft.add_vline(x=hk, line_dash="dot", line_color="#eab308", annotation_text=f"{k}X")

        fig_fft.update_layout(
            title="FFT Spectrum Analyzer (Harmonic 1X-5X Tracker)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude (g)",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    # Spectrogram Waterfall Heatmap
    if len(st.session_state.spectrogram_history) > 1:
        spec_mat = np.array(st.session_state.spectrogram_history)  # (time, freqs)
        mask_spec = freqs <= 2500.0
        fig_spec = go.Figure(data=go.Heatmap(
            z=spec_mat[:, mask_spec],
            x=freqs[mask_spec],
            y=list(range(len(spec_mat))),
            colorscale="Turbo",
            colorbar=dict(title="dB"),
        ))
        fig_spec.update_layout(
            title="Rolling 2D Waterfall Spectrogram (Time vs Frequency)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Rolling Time Frames",
            template="plotly_dark",
            height=280,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_spec, use_container_width=True)

    # Telemetry snapshot
    with st.expander("📊 Detailed Telemetry & Feature Vector Table"):
        feat_dict = {
            "RMS (g)": f"{features.rms:.6f}",
            "Peak (g)": f"{features.peak:.6f}",
            "Peak-to-Peak (g)": f"{features.peak_to_peak:.6f}",
            "Crest Factor": f"{features.crest_factor:.4f}",
            "Kurtosis": f"{features.kurtosis:.4f}",
            "Skewness": f"{features.skewness:.4f}",
            "Shape Factor": f"{features.shape_factor:.4f}",
            "Dominant Freq (Hz)": f"{features.dominant_frequency_hz:.2f}",
            "Calculated RPM": f"{features.calculated_rpm:.1f}",
            "Spectral Centroid (Hz)": f"{features.spectral_centroid_hz:.2f}",
            "Spectral Rolloff (Hz)": f"{features.spectral_rolloff_hz:.2f}",
            "THD (%)": f"{features.thd_percent:.2f}%",
            "Spectral Flatness": f"{features.spectral_flatness:.6f}",
            "Mahalanobis Distance": f"{report.mahalanobis_distance:.2f}",
            "Isolation Forest Score": f"{report.isolation_forest_score:.4f}",
            "Machine Health Index": f"{report.health_index_percent:.1f}%",
            "ISO Severity Zone": report.iso_zone.value,
        }
        st.json(feat_dict)

    time.sleep(0.05)
    st.button("🔄 Refresh Telemetry Frame", on_click=lambda: None)


if __name__ == "__main__":
    run_web_app()
