"""
views.py - High-Performance PySide6 & PyQtGraph 60 FPS Diagnostic Viewports

Modular QDockWidget Architecture:
- Panel A: Real-Time Oscilloscope (Dual-Trace with Rising-Edge Trigger & Phosphor Decay Persistence)
- Panel B: Live Narrowband FFT Spectrum (1X-10X Harmonics, Sidebands, Peak-Hold, Baseline Snapshot, Interactive Crosshairs)
- Panel C: GPU-Accelerated Rolling 2D Spectrogram (ImageItem with 30-60s history & custom colormaps)
- Panel D: 3D Waterfall / Spectral Cascade Plot (GLViewWidget OpenGL surface/mesh with interactive 3D orbit)
- Panel E: Envelope Demodulation & Power Cepstrum (Hilbert Transform HFRT & Quefrency Spectrum)
- HUD Telemetry Ribbon: Real-time RPM, RMS, Crest Factor, Kurtosis, ISO 10816 Severity Gauge, Anomaly Strip
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from .dsp_worker import DSPFramePayload, HarmonicCursorInfo, ISOZone

logger = logging.getLogger("vibration_analyzer.views")

# Global High-Contrast Cyber Dark Theme
GRAPHITE_BG = "#0f1117"
PANEL_BG = "#161922"
BORDER_COLOR = "#232838"
CYBER_CYAN = "#00f0ff"
NEON_AMBER = "#ffaa00"
ALARM_CRIMSON = "#ff0055"
BRIGHT_GREEN = "#00ff88"
PURPLE_GHOST = "#bf5af2"


class ModernDockWidget(QtWidgets.QDockWidget):
    """Sleek Industrial QDockWidget wrapper."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setObjectName(title.replace(" ", "_"))
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setStyleSheet(f"""
            QDockWidget {{
                color: #38bdf8;
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 1px;
            }}
            QDockWidget::title {{
                background-color: #12151e;
                border: 1px solid {BORDER_COLOR};
                border-bottom: 2px solid #0284c7;
                padding: 6px 10px;
                text-align: left;
            }}
        """)


class HUDCard(QtWidgets.QFrame):
    """Digital HUD telemetry card with cybernetic neon readouts."""

    def __init__(self, title: str, unit: str = "", default_val: str = "--", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(1)

        self.lbl_title = QtWidgets.QLabel(title.upper())
        self.lbl_title.setStyleSheet("color: #64748b; font-size: 9px; font-weight: 800; letter-spacing: 1px;")

        h_val = QtWidgets.QHBoxLayout()
        h_val.setSpacing(4)
        self.lbl_value = QtWidgets.QLabel(default_val)
        self.lbl_value.setStyleSheet(f"color: {CYBER_CYAN}; font-size: 18px; font-weight: 900; font-family: monospace;")
        self.lbl_unit = QtWidgets.QLabel(unit)
        self.lbl_unit.setStyleSheet("color: #94a3b8; font-size: 10px; padding-top: 3px;")
        h_val.addWidget(self.lbl_value)
        h_val.addWidget(self.lbl_unit)
        h_val.addStretch()

        self.lbl_subtext = QtWidgets.QLabel("Nominal")
        self.lbl_subtext.setStyleSheet("color: #475569; font-size: 9px;")

        layout.addWidget(self.lbl_title)
        layout.addLayout(h_val)
        layout.addWidget(self.lbl_subtext)

    def update_val(self, val_str: str, color: Optional[str] = None, subtext: Optional[str] = None):
        self.lbl_value.setText(val_str)
        if color:
            self.lbl_value.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: 900; font-family: monospace;")
        if subtext:
            self.lbl_subtext.setText(subtext)


class HUDTelemetryRibbon(QtWidgets.QWidget):
    """Top HUD Telemetry Ribbon displaying critical machine diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.card_rpm = HUDCard("Shaft Speed", "RPM", "0")
        self.card_rms = HUDCard("RMS Velocity", "mm/s", "0.00")
        self.card_acc = HUDCard("RMS Acceleration", "g", "0.000")
        self.card_crest = HUDCard("Crest Factor", "", "1.00")
        self.card_kurt = HUDCard("Kurtosis (Spall)", "κ", "3.00")
        self.card_thd = HUDCard("THD (Harmonics)", "%", "0.0%")
        self.card_health = HUDCard("Machine Health", "Index", "100%")
        self.card_iso = HUDCard("ISO 10816 Zone", "Severity", "Zone A")

        for card in [
            self.card_rpm, self.card_rms, self.card_acc,
            self.card_crest, self.card_kurt, self.card_thd,
            self.card_health, self.card_iso,
        ]:
            layout.addWidget(card)

    def update_telemetry(self, p: DSPFramePayload):
        self.card_rpm.update_val(
            f"{int(p.calculated_rpm):,}",
            CYBER_CYAN,
            f"Fundamental: {p.dominant_frequency_hz:.1f} Hz (1X)",
        )
        # Proxy velocity in mm/s and acceleration in g
        vel_proxy = p.rms * 60.0
        self.card_rms.update_val(
            f"{vel_proxy:.2f}",
            ALARM_CRIMSON if vel_proxy > 4.5 else CYBER_CYAN,
            f"Peak: {p.peak * 60.0:.2f} mm/s",
        )
        self.card_acc.update_val(f"{p.rms:.4f}", CYBER_CYAN, f"P-P: {p.peak_to_peak:.3f} g")
        self.card_crest.update_val(
            f"{p.crest_factor:.2f}",
            NEON_AMBER if p.crest_factor > 4.5 else CYBER_CYAN,
            "Impact ratio",
        )
        self.card_kurt.update_val(
            f"{p.kurtosis:.2f}",
            ALARM_CRIMSON if p.kurtosis > 4.5 else (CYBER_CYAN if p.kurtosis < 3.8 else NEON_AMBER),
            "Gaussian = 3.0",
        )
        self.card_thd.update_val(
            f"{p.thd_percent:.1f}%",
            NEON_AMBER if p.thd_percent > 25.0 else CYBER_CYAN,
            "Orders 2X-10X",
        )
        self.card_health.update_val(
            f"{p.health_index_percent:.0f}%",
            p.iso_zone_color,
            f"Anomaly: {p.anomaly_score:.2f}",
        )
        self.card_iso.update_val(
            p.iso_zone.name.replace("_", " "),
            p.iso_zone_color,
            p.iso_zone.value.split("(")[-1].replace(")", ""),
        )


# ==============================================================================
# Panel A: Real-Time Dual-Trace Oscilloscope
# ==============================================================================
class OscilloscopeDock(ModernDockWidget):
    """
    Panel A: Real-Time Dual-Trace Oscilloscope.
    Dual traces: Raw audio waveform vs. bandpass-filtered vibration proxy.
    Includes zero-crossing rising-edge trigger stabilization & persistence phosphor ghost trail.
    """

    def __init__(self, parent=None):
        super().__init__("Panel A: Real-Time Oscilloscope (Time-Domain)", parent)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Amplitude (V / g)", color="#94a3b8")
        self.plot_widget.setLabel("bottom", "Time Frame Samples", color="#94a3b8")
        self.plot_widget.setYRange(-1.0, 1.0)

        # Persistence phosphor ghost trace (previous cycle in transparent cyan)
        self.curve_persist = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 240, 255, 60), width=1.0),
            name="Phosphor Decay",
        )
        # Raw waveform (dim slate cyan)
        self.curve_raw = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 119, 153, 140), width=1.0),
            name="Raw Acoustic",
        )
        # Filtered vibration proxy (cyber cyan neon)
        self.curve_filtered = self.plot_widget.plot(
            pen=pg.mkPen(color=CYBER_CYAN, width=1.8),
            name="Filtered Proxy",
        )

        # Trigger level indicator line
        self.line_trigger = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen(color=NEON_AMBER, width=1, style=QtCore.Qt.PenStyle.DashLine),
            label="Trig Level",
            labelOpts={"color": NEON_AMBER, "position": 0.95},
        )
        self.plot_widget.addItem(self.line_trigger)

        # Phosphor history memory
        self._last_trig_sig = np.zeros(0, dtype=np.float32)
        self.setWidget(self.plot_widget)

    def update_data(self, p: DSPFramePayload):
        # Update persistence trace from previous cycle
        if len(self._last_trig_sig) > 0:
            self.curve_persist.setData(self._last_trig_sig)

        # Update raw and triggered filtered traces
        self.curve_raw.setData(p.raw_time_sig)
        self.curve_filtered.setData(p.triggered_time_sig)
        self._last_trig_sig = p.triggered_time_sig.copy()


# ==============================================================================
# Panel B: Live Narrowband FFT Spectrum & Harmonic Cursor
# ==============================================================================
class NarrowbandSpectrumDock(ModernDockWidget):
    """
    Panel B: Live Narrowband FFT Spectrum & Harmonic Cursor.
    Single-sided amplitude spectrum, decaying peak-hold trace, baseline snapshot overlay,
    harmonic cursors (1X-10X RPM), dynamic sidebands, draggable alarm line, and crosshair delta markers.
    """

    def __init__(self, parent=None):
        super().__init__("Panel B: Narrowband FFT Spectrum & Harmonic Tracker", parent)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Amplitude (g Peak)", color="#94a3b8")
        self.plot_widget.setLabel("bottom", "Frequency (Hz)", color="#94a3b8")
        self.plot_widget.setXRange(0, 3000, padding=0.01)

        # 1. Baseline Snapshot Ghost Overlay (Frozen purple trace)
        self.curve_baseline = self.plot_widget.plot(
            pen=pg.mkPen(color=(191, 90, 242, 160), width=1.5, style=QtCore.Qt.PenStyle.DashDotLine),
            name="Baseline Snapshot",
        )
        self.curve_baseline.setVisible(False)

        # 2. Decaying Peak-Hold Envelope (Neon Amber dashed)
        self.curve_peak_hold = self.plot_widget.plot(
            pen=pg.mkPen(color=NEON_AMBER, width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            name="Peak Hold",
        )

        # 3. Live Real-Time Spectrum (Cyber Cyan filled)
        self.curve_live = self.plot_widget.plot(
            pen=pg.mkPen(color=CYBER_CYAN, width=1.8),
            fillLevel=0.0,
            brush=(0, 240, 255, 35),
            name="Live Spectrum",
        )

        # 4. Draggable Alarm Threshold Line
        self.alarm_threshold = 0.40
        self.line_alarm = pg.InfiniteLine(
            pos=self.alarm_threshold,
            angle=0,
            pen=pg.mkPen(color=ALARM_CRIMSON, width=2, style=QtCore.Qt.PenStyle.DashLine),
            movable=True,
            label="Alarm Threshold ({value:.2f})",
            labelOpts={"color": ALARM_CRIMSON, "position": 0.85, "fill": (18, 21, 30, 220)},
        )
        self.line_alarm.sigPositionChanged.connect(self._on_alarm_moved)
        self.plot_widget.addItem(self.line_alarm)

        # 5. Harmonic Cursors (1X through 10X)
        self.harmonic_cursors: List[pg.InfiniteLine] = []
        harm_colors = [
            "#00ff88", "#00f0ff", "#ffaa00", "#ff0055", "#bf5af2",
            "#38bdf8", "#4ade80", "#facc15", "#fb7185", "#c084fc",
        ]
        for k in range(1, 11):
            line = pg.InfiniteLine(
                pos=-100,
                angle=90,
                pen=pg.mkPen(color=harm_colors[k - 1], width=1, style=QtCore.Qt.PenStyle.DotLine),
                label=f"{k}X",
                labelOpts={"color": harm_colors[k - 1], "position": 0.92, "fill": (15, 17, 23, 180)},
            )
            self.harmonic_cursors.append(line)
            self.plot_widget.addItem(line)

        # 6. Bearing Defect / Sideband Cursor (BPFO)
        self.line_bpfo = pg.InfiniteLine(
            pos=-100,
            angle=90,
            pen=pg.mkPen(color="#f43f5e", width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            label="BPFO",
            labelOpts={"color": "#f43f5e", "position": 0.75, "fill": (15, 17, 23, 180)},
        )
        self.plot_widget.addItem(self.line_bpfo)

        # 7. Interactive Crosshair & Delta Cursor
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#64748b", width=1))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#64748b", width=1))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)

        self.lbl_cursor_info = pg.TextItem(text="", color="#f8fafc", fill=(18, 21, 30, 220), border="#38bdf8")
        self.lbl_cursor_info.setZValue(100)
        self.plot_widget.addItem(self.lbl_cursor_info)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Snapshot cache
        self._baseline_spectrum: Optional[np.ndarray] = None
        self._baseline_freqs: Optional[np.ndarray] = None
        self._use_db = False

        self.setWidget(self.plot_widget)

    def _on_alarm_moved(self):
        self.alarm_threshold = float(self.line_alarm.value())

    def _on_mouse_moved(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_pt = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            fx = mouse_pt.x()
            fy = mouse_pt.y()
            self.v_line.setPos(fx)
            self.h_line.setPos(fy)
            self.lbl_cursor_info.setPos(fx, fy)
            unit = "dBV" if self._use_db else "g"
            self.lbl_cursor_info.setText(f"Freq: {fx:.1f} Hz\nAmp: {fy:.3f} {unit}")

    def capture_baseline_snapshot(self, freqs: np.ndarray, amps: np.ndarray):
        """Freeze current spectrum as ghost overlay."""
        self._baseline_freqs = freqs.copy()
        self._baseline_spectrum = amps.copy()
        self.curve_baseline.setData(self._baseline_freqs, self._baseline_spectrum)
        self.curve_baseline.setVisible(True)
        logger.info("Captured baseline spectrum snapshot overlay.")

    def clear_baseline_snapshot(self):
        self._baseline_spectrum = None
        self.curve_baseline.setVisible(False)

    def set_db_scale(self, enable_db: bool):
        self._use_db = enable_db
        if self._use_db:
            self.plot_widget.setLabel("left", "Amplitude (dBV)", color="#94a3b8")
            self.plot_widget.setYRange(-80.0, 10.0)
        else:
            self.plot_widget.setLabel("left", "Amplitude (g Peak)", color="#94a3b8")

    def update_data(self, p: DSPFramePayload):
        y_data = p.db_spectrum if self._use_db else p.amp_spectrum
        y_peak = 20.0 * np.log10(np.maximum(p.peak_hold_spectrum, 1e-12)) if self._use_db else p.peak_hold_spectrum

        self.curve_live.setData(p.freqs_hz, y_data)
        self.curve_peak_hold.setData(p.freqs_hz, y_peak)

        if not self._use_db:
            max_y = max(self.alarm_threshold * 1.3, float(np.max(y_data)) * 1.25, 0.2)
            self.plot_widget.setYRange(0.0, max_y)

        # Position 1X-10X harmonic cursors
        f0 = p.dominant_frequency_hz
        if f0 > 2.0:
            for k in range(1, 11):
                hk = f0 * k
                self.harmonic_cursors[k - 1].setValue(hk)
            # Position BPFO sideband cursor (~3.56X)
            self.line_bpfo.setValue(f0 * 3.56)
        else:
            for cur in self.harmonic_cursors:
                cur.setValue(-100)
            self.line_bpfo.setValue(-100)


# ==============================================================================
# Panel C: GPU-Accelerated Rolling 2D Spectrogram
# ==============================================================================
class Spectrogram2DDock(ModernDockWidget):
    """
    Panel C: GPU-Accelerated Rolling 2D Spectrogram.
    Continuous scrolling buffer representing the last 30 to 60 seconds of vibration history.
    Renders with PyQtGraph ImageItem and high dynamic range colormaps.
    """

    def __init__(self, parent=None):
        super().__init__("Panel C: Rolling 2D Spectrogram (Time vs Frequency)", parent)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "History Slices (Rolling Time)", color="#94a3b8")
        self.plot_widget.setLabel("bottom", "Frequency (Hz)", color="#94a3b8")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)

        self.img_item = pg.ImageItem()
        # Default colormap: Turbo (high dynamic range)
        turbo_lut = pg.colormap.get("turbo").getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(turbo_lut)
        self.plot_widget.addItem(self.img_item)

        self.history_len = 160
        self.max_freq_hz = 3000.0
        self._spectrogram_matrix = np.full((1, 1), -80.0, dtype=np.float32)

        self.setWidget(self.plot_widget)

    def set_colormap(self, cmap_name: str):
        lut = pg.colormap.get(cmap_name).getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(lut)

    def update_data(self, p: DSPFramePayload):
        try:
            if len(p.freqs_hz) == 0 or len(p.db_spectrum) == 0:
                return

            max_idx = np.searchsorted(p.freqs_hz, self.max_freq_hz)
            max_idx = min(len(p.freqs_hz), max(10, max_idx))
            freq_slice = p.freqs_hz[:max_idx]
            db_slice = p.db_spectrum[:max_idx]

            if self._spectrogram_matrix.shape[0] != max_idx or self._spectrogram_matrix.shape[1] != self.history_len:
                self._spectrogram_matrix = np.full((max_idx, self.history_len), -80.0, dtype=np.float32)

            # Roll along time axis (axis 1) and insert latest slice
            self._spectrogram_matrix = np.roll(self._spectrogram_matrix, -1, axis=1)
            self._spectrogram_matrix[:len(db_slice), -1] = db_slice

            self.img_item.setImage(self._spectrogram_matrix, autoLevels=False, levels=[-65.0, 5.0])
            if len(freq_slice) > 0:
                self.img_item.setRect(QtCore.QRectF(0, 0, float(freq_slice[-1]), float(self.history_len)))
                self.plot_widget.setXRange(0, float(freq_slice[-1]), padding=0.0)
                self.plot_widget.setYRange(0, self.history_len, padding=0.0)
        except Exception as e:
            logger.debug(f"2D Spectrogram update exception: {e}")


# ==============================================================================
# Panel D: 3D Waterfall / Spectral Cascade Plot (OpenGL)
# ==============================================================================
class Cascade3DDock(ModernDockWidget):
    """
    Panel D: 3D Waterfall / Spectral Cascade Plot.
    PyQtGraph GLViewWidget showing frequency vs. time vs. amplitude in 3D perspective
    with interactive orbit, pan, and zoom.
    """

    def __init__(self, parent=None):
        super().__init__("Panel D: 3D Spectral Cascade Plot (OpenGL)", parent)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor("#0c0f17")
        self.gl_view.setCameraPosition(distance=55, elevation=35, azimuth=45)

        # Add 3D reference grid
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        grid.translate(0, 0, -5)
        self.gl_view.addItem(grid)

        # 3D Spectral Surface mesh
        self.cascade_time_depth = 35
        self.freq_bins_3d = 48
        self._z_mesh = np.zeros((self.cascade_time_depth, self.freq_bins_3d), dtype=np.float32)

        x_coords = np.linspace(-25, 25, self.cascade_time_depth)
        y_coords = np.linspace(-25, 25, self.freq_bins_3d)

        self.surface_item = gl.GLSurfacePlotItem(
            x=x_coords,
            y=y_coords,
            z=self._z_mesh,
            shader="shaded",
            color=(0.0, 0.8, 1.0, 0.9),
        )
        self.gl_view.addItem(self.surface_item)

        self.setWidget(self.gl_view)

    def update_data(self, p: DSPFramePayload):
        try:
            # Downsample frequency bins for smooth 60 FPS 3D mesh rendering
            amps = p.amp_spectrum
            if len(amps) == 0:
                return

            step = max(1, len(amps) // self.freq_bins_3d)
            slice_vals = amps[::step][:self.freq_bins_3d]
            downsampled = np.zeros(self.freq_bins_3d, dtype=np.float32)
            downsampled[:len(slice_vals)] = slice_vals * 12.0  # Scale amplitude height

            # Roll along time axis (axis 0)
            self._z_mesh = np.roll(self._z_mesh, -1, axis=0)
            self._z_mesh[-1, :] = downsampled

            self.surface_item.setData(z=self._z_mesh)
        except Exception as e:
            logger.debug(f"3D Cascade render update exception: {e}")


# ==============================================================================
# Panel E: Envelope Demodulation (HFRT) & Power Cepstrum
# ==============================================================================
class EnvelopeCepstrumDock(ModernDockWidget):
    """
    Panel E: Envelope Demodulation (High-Frequency Resonance Technique / HFRT) & Power Cepstrum.
    Analytic signal via Hilbert transform exposes low-frequency bearing click trains;
    Power Cepstrum exposes harmonic family periodicity and quefrency peaks.
    """

    def __init__(self, parent=None):
        super().__init__("Panel E: Envelope Demodulation (HFRT) & Power Cepstrum", parent)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Subplot 1: Hilbert Envelope Spectrum
        self.plot_env = pg.PlotWidget(title="<b style='color:#00f0ff;'>HILBERT ENVELOPE SPECTRUM (BEARING DEFECT BPFO/BPFI)</b>")
        self.plot_env.showGrid(x=True, y=True, alpha=0.2)
        self.plot_env.setLabel("left", "Env Amp", color="#94a3b8")
        self.plot_env.setLabel("bottom", "Modulation Freq (Hz)", color="#94a3b8")
        self.curve_env = self.plot_env.plot(pen=pg.mkPen(color=CYBER_CYAN, width=1.5), fillLevel=0.0, brush=(0, 240, 255, 30))
        layout.addWidget(self.plot_env)

        # Subplot 2: Power Cepstrum
        self.plot_cep = pg.PlotWidget(title="<b style='color:#ffaa00;'>POWER CEPSTRUM (QUEFRENCY HARMONIC ECHO SPECTRUM)</b>")
        self.plot_cep.showGrid(x=True, y=True, alpha=0.2)
        self.plot_cep.setLabel("left", "Norm Power", color="#94a3b8")
        self.plot_cep.setLabel("bottom", "Quefrency (ms)", color="#94a3b8")
        self.curve_cep = self.plot_cep.plot(pen=pg.mkPen(color=NEON_AMBER, width=1.5))
        layout.addWidget(self.plot_cep)

        self.setWidget(container)

    def update_data(self, p: DSPFramePayload):
        if len(p.envelope_freqs) > 0 and len(p.envelope_spectrum) > 0:
            self.curve_env.setData(p.envelope_freqs, p.envelope_spectrum)
        if len(p.cepstrum_quefrency_ms) > 0 and len(p.cepstrum_spectrum) > 0:
            self.curve_cep.setData(p.cepstrum_quefrency_ms, p.cepstrum_spectrum)
