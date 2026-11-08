"""
app.py - Master GPU-Accelerated Real-Time Mechanical Vibration Workstation Application

Architecture:
- PySide6 + PyQtGraph with OpenGL hardware acceleration (useOpenGL=True, antialias=True).
- Asynchronous multi-threaded pipeline:
    * Audio Thread: Non-blocking C-level PortAudio callback writing to atomic ring buffer.
    * DSP Worker QThread: 60-100 Hz Numba JIT statistics, Hilbert demodulation, FFT, harmonic tracking.
    * GUI Thread: 60 FPS slot rendering via PyQtGraph without event loop stalls.
- Modular QDockWidget workspace enabling full docking, undocking, floating, and tab grouping.
- Comprehensive 5-panel multi-modal diagnostic viewports:
    * Panel A: Real-Time Oscilloscope (Dual-trace + rising-edge trigger + persistence)
    * Panel B: Narrowband FFT Spectrum (1X-10X harmonic cursors, sidebands, peak-hold, baseline overlay, crosshairs)
    * Panel C: GPU-Accelerated Rolling 2D Spectrogram (30-60s history)
    * Panel D: 3D Spectral Cascade Plot (OpenGL surface / cascade mesh with interactive orbit)
    * Panel E: Envelope Demodulation (HFRT) & Power Cepstrum
- Session Flight Recorder: Synchronized raw audio WAV + CSV vibration logs.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Slot
import pyqtgraph as pg

# Configure PyQtGraph Performance Options
try:
    pg.setConfigOptions(useOpenGL=False, antialias=False, background="#0f1117", foreground="#94a3b8")
except Exception as e:
    pass

# Support both relative package import and standalone execution
try:
    from .audio_engine import AudioConfig, AudioEngine
    from .dsp_worker import (
        DSPFramePayload,
        DSPWorkerThread,
        ISOZone,
        MachineClass,
        WindowType,
    )
    from .synthetic_generator import SyntheticMechanicalConfig, SyntheticVibrationGenerator
    from .views import (
        Cascade3DDock,
        EnvelopeCepstrumDock,
        HUDTelemetryRibbon,
        ModernDockWidget,
        NarrowbandSpectrumDock,
        OscilloscopeDock,
        Spectrogram2DDock,
    )
except ImportError:
    from audio_engine import AudioConfig, AudioEngine
    from dsp_worker import (
        DSPFramePayload,
        DSPWorkerThread,
        ISOZone,
        MachineClass,
        WindowType,
    )
    from synthetic_generator import SyntheticMechanicalConfig, SyntheticVibrationGenerator
    from views import (
        Cascade3DDock,
        EnvelopeCepstrumDock,
        HUDTelemetryRibbon,
        ModernDockWidget,
        NarrowbandSpectrumDock,
        OscilloscopeDock,
        Spectrogram2DDock,
    )

logger = logging.getLogger("vibration_analyzer.app")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Workstation] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# High-DPI UI scaling
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

WORKSTATION_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0c0f17;
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 12px;
}

QMenuBar {
    background-color: #12151e;
    color: #cbd5e1;
    border-bottom: 1px solid #1e293b;
}

QMenuBar::item:selected {
    background-color: #0284c7;
    color: #ffffff;
}

QMenu {
    background-color: #161922;
    color: #f8fafc;
    border: 1px solid #334155;
}

QMenu::item:selected {
    background-color: #0284c7;
}

QStatusBar {
    background-color: #11141c;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
    font-size: 11px;
}

QGroupBox {
    border: 1px solid #232838;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
    font-size: 11px;
    color: #38bdf8;
    background-color: #141722;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: #141722;
}

QComboBox, QSpinBox, QDoubleSpinBox, QSlider {
    background-color: #1e2330;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #f8fafc;
    padding: 3px 6px;
    min-height: 20px;
}

QPushButton {
    background-color: #1e2330;
    border: 1px solid #334155;
    border-radius: 5px;
    color: #f8fafc;
    font-weight: 600;
    padding: 5px 12px;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #0ea5e9;
}

QPushButton:pressed {
    background-color: #0284c7;
}

QPushButton#btnRecordActive {
    background-color: #881337;
    border: 1px solid #f43f5e;
    color: #ffe4e6;
}

QPushButton#btnCalibrate {
    background-color: #0369a1;
    border: 1px solid #0284c7;
}

QProgressBar {
    border: 1px solid #334155;
    border-radius: 4px;
    background-color: #1e2330;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    height: 16px;
}

QProgressBar::chunk {
    background-color: #0ea5e9;
    border-radius: 3px;
}

QTabWidget::pane {
    border: 1px solid #232838;
    background-color: #11141c;
}

QTabBar::tab {
    background-color: #1e2330;
    border: 1px solid #334155;
    padding: 5px 12px;
    margin-right: 2px;
    color: #94a3b8;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #11141c;
    color: #00f0ff;
    font-weight: bold;
    border-color: #00f0ff;
}
"""


class VibrationWorkstationWindow(QtWidgets.QMainWindow):
    """
    Principal Workstation Application Window.
    Coordinates audio stream acquisition, DSP worker QThread, modular QDockWidgets,
    session flight recorder, and baseline snapshot overlays.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Mechanical Vibration Analyzer & Acoustic Anomaly Workstation")
        self.resize(1620, 1000)
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(WORKSTATION_STYLESHEET)
        self.setDockNestingEnabled(True)

        # Core Engines
        self.sample_rate = 44100
        self.audio_engine = AudioEngine(AudioConfig(sample_rate=self.sample_rate, block_size=4096, gain=1.0))
        self.synthetic_gen = SyntheticVibrationGenerator(SyntheticMechanicalConfig())
        self.audio_engine.set_synthetic_generator(self.synthetic_gen.generate_block)

        # DSP Worker QThread (Asynchronous processing at 60 Hz)
        self.dsp_worker = DSPWorkerThread(self.audio_engine)

        # High-Responsiveness GUI Pacing Engine (prevents Qt event queue backlog)
        self._last_rendered_timestamp = 0.0
        self._is_rendering_frame = False
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(28)  # ~35 FPS paced rendering for 100% responsive UI
        self._render_timer.timeout.connect(self._on_render_tick)

        # Session Flight Recorder State
        self._is_recording_flight = False
        self._flight_log_rows: List[dict] = []

        # Latest frame cache
        self._latest_payload: Optional[DSPFramePayload] = None

        # Build Workstation UI
        self._init_ui()
        self._init_menu_bar()

        # Start acquisition & DSP worker thread
        self._render_timer.start()
        self.audio_engine.start(use_synthetic=False)
        self.dsp_worker.start()

    def _init_ui(self):
        """Construct modular docked viewports."""
        # Top HUD Ribbon in Central Widget (or top bar)
        central_container = QtWidgets.QWidget(self)
        self.setCentralWidget(central_container)
        central_layout = QtWidgets.QVBoxLayout(central_container)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(6)

        # 1. Top HUD Ribbon
        self.hud_ribbon = HUDTelemetryRibbon(self)
        central_layout.addWidget(self.hud_ribbon)

        # Status Bar
        self.statusBar().showMessage("SYSTEM READY | PortAudio & OpenGL Diagnostics Active")

        # 2. Instantiate All 5 Diagnostic Viewports
        self.dock_osc = OscilloscopeDock(self)
        self.dock_fft = NarrowbandSpectrumDock(self)
        self.dock_spec2d = Spectrogram2DDock(self)
        self.dock_casc3d = Cascade3DDock(self)
        self.dock_env_cep = EnvelopeCepstrumDock(self)

        # 3. Instantiate Engineering Controls Dock
        self.dock_controls = self._build_controls_dock()

        # 4. Dock Arrangement
        # Left Area: FFT Spectrum (primary) & Oscilloscope
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_fft)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_osc)

        # Split horizontally: Right Area has 2D Spectrogram, 3D Cascade, and Envelope/Cepstrum tabbed
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_spec2d)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_casc3d)
        self.tabifyDockWidget(self.dock_casc3d, self.dock_env_cep)

        # Far Right: Control Panel
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_controls)

        # Optimal Default Split Proportions
        self.setCorner(QtCore.Qt.Corner.TopLeftCorner, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.BottomLeftCorner, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.TopRightCorner, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.BottomRightCorner, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)

    def _init_menu_bar(self):
        """Application Menu Bar."""
        menu = self.menuBar()

        # View Menu: Viewport toggle visibility
        view_menu = menu.addMenu("Viewports")
        view_menu.addAction(self.dock_osc.toggleViewAction())
        view_menu.addAction(self.dock_fft.toggleViewAction())
        view_menu.addAction(self.dock_spec2d.toggleViewAction())
        view_menu.addAction(self.dock_casc3d.toggleViewAction())
        view_menu.addAction(self.dock_env_cep.toggleViewAction())
        view_menu.addAction(self.dock_controls.toggleViewAction())

        # Reset Layout action
        act_reset = QtGui.QAction("Reset Default Dock Layout", self)
        act_reset.triggered.connect(self._reset_dock_layout)
        view_menu.addSeparator()
        view_menu.addAction(act_reset)

        # Flight Recorder Menu
        rec_menu = menu.addMenu("Flight Recorder")
        act_start_rec = QtGui.QAction("Start Session Recording (WAV + CSV)", self)
        act_start_rec.triggered.connect(self._on_toggle_flight_recorder)
        rec_menu.addAction(act_start_rec)

    def _reset_dock_layout(self):
        """Restore pristine docked arrangement."""
        for d in [self.dock_osc, self.dock_fft, self.dock_spec2d, self.dock_casc3d, self.dock_env_cep, self.dock_controls]:
            d.setVisible(True)
            d.setFloating(False)

    def _build_controls_dock(self) -> ModernDockWidget:
        """Create tabbed engineering controls dock."""
        dock = ModernDockWidget("Engineering Controls & Diagnostics", self)
        tabs = QtWidgets.QTabWidget()

        # --- Tab 1: DSP & Acquisition ---
        t_dsp = QtWidgets.QWidget()
        l_dsp = QtWidgets.QVBoxLayout(t_dsp)
        l_dsp.setSpacing(8)

        # Audio Hardware
        grp_hw = QtWidgets.QGroupBox("Acoustic / Vibration Input")
        lh = QtWidgets.QVBoxLayout(grp_hw)

        lh.addWidget(QtWidgets.QLabel("Input Device:"))
        self.combo_devices = QtWidgets.QComboBox()
        self._populate_devices()
        self.combo_devices.currentIndexChanged.connect(self._on_device_changed)
        lh.addWidget(self.combo_devices)

        # Preamp Gain
        g_box = QtWidgets.QHBoxLayout()
        g_box.addWidget(QtWidgets.QLabel("Preamp Gain:"))
        self.lbl_gain = QtWidgets.QLabel("1.0x")
        g_box.addWidget(self.lbl_gain)
        lh.addLayout(g_box)

        self.slider_gain = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_gain.setRange(1, 150)
        self.slider_gain.setValue(10)
        self.slider_gain.valueChanged.connect(self._on_gain_changed)
        lh.addWidget(self.slider_gain)

        l_dsp.addWidget(grp_hw)

        # DSP Filtering & Windowing
        grp_filt = QtWidgets.QGroupBox("DSP Filtering & Windowing")
        lf = QtWidgets.QVBoxLayout(grp_filt)

        lf.addWidget(QtWidgets.QLabel("FFT Window:"))
        self.combo_window = QtWidgets.QComboBox()
        for w in WindowType:
            self.combo_window.addItem(w.value.replace("_", " ").title(), w)
        self.combo_window.setCurrentIndex(2)  # Blackman-Harris
        self.combo_window.currentIndexChanged.connect(self._on_window_changed)
        lf.addWidget(self.combo_window)

        # FFT Size
        lf.addWidget(QtWidgets.QLabel("FFT Resolution (Bins):"))
        self.combo_fft_size = QtWidgets.QComboBox()
        for sz in [1024, 2048, 4096, 8192, 16384]:
            self.combo_fft_size.addItem(f"{sz} bins (Δf ≈ {self.sample_rate / sz:.1f} Hz)", sz)
        self.combo_fft_size.setCurrentIndex(2)  # 4096 default
        self.combo_fft_size.currentIndexChanged.connect(self._on_fft_size_changed)
        lf.addWidget(self.combo_fft_size)

        # Bandpass Filter
        self.chk_bp_filter = QtWidgets.QCheckBox("Enable Butterworth SOS Bandpass Filter")
        self.chk_bp_filter.setChecked(True)
        self.chk_bp_filter.stateChanged.connect(self._on_filter_changed)
        lf.addWidget(self.chk_bp_filter)

        f_grid = QtWidgets.QGridLayout()
        f_grid.addWidget(QtWidgets.QLabel("High-Pass (Hz):"), 0, 0)
        self.spin_low_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_low_cutoff.setRange(1.0, 500.0)
        self.spin_low_cutoff.setValue(10.0)
        self.spin_low_cutoff.valueChanged.connect(self._on_filter_changed)
        f_grid.addWidget(self.spin_low_cutoff, 0, 1)

        f_grid.addWidget(QtWidgets.QLabel("Low-Pass (Hz):"), 1, 0)
        self.spin_high_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_high_cutoff.setRange(50.0, 15000.0)
        self.spin_high_cutoff.setValue(3000.0)
        self.spin_high_cutoff.valueChanged.connect(self._on_filter_changed)
        f_grid.addWidget(self.spin_high_cutoff, 1, 1)
        lf.addLayout(f_grid)

        # Spectrum Display options
        self.chk_db = QtWidgets.QCheckBox("Display Spectrum in Logarithmic dBV Scale")
        self.chk_db.setChecked(False)
        self.chk_db.stateChanged.connect(lambda s: self.dock_fft.set_db_scale(s == 2 or s == QtCore.Qt.CheckState.Checked.value))
        lf.addWidget(self.chk_db)

        # Colormap for 2D Spectrogram
        cmap_box = QtWidgets.QHBoxLayout()
        cmap_box.addWidget(QtWidgets.QLabel("Spectrogram Colormap:"))
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(["turbo", "inferno", "viridis", "plasma"])
        self.combo_cmap.currentTextChanged.connect(self.dock_spec2d.set_colormap)
        cmap_box.addWidget(self.combo_cmap)
        lf.addLayout(cmap_box)

        l_dsp.addWidget(grp_filt)
        l_dsp.addStretch()
        tabs.addTab(t_dsp, "DSP & Filter")

        # --- Tab 2: Baseline & Flight Recorder ---
        t_rec = QtWidgets.QWidget()
        l_rec = QtWidgets.QVBoxLayout(t_rec)
        l_rec.setSpacing(8)

        # Baseline Calibration Routine
        grp_cal = QtWidgets.QGroupBox("10s Baseline Calibration Routine")
        lc = QtWidgets.QVBoxLayout(grp_cal)
        lc.addWidget(QtWidgets.QLabel("Record 10 seconds of normal machinery operation to train covariance & Isolation Forest:"))
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate Baseline (10s)")
        self.btn_calibrate.setObjectName("btnCalibrate")
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        lc.addWidget(self.btn_calibrate)

        self.progress_cal = QtWidgets.QProgressBar()
        self.progress_cal.setValue(0)
        lc.addWidget(self.progress_cal)

        self.btn_reset_cal = QtWidgets.QPushButton("Reset Learned Baseline")
        self.btn_reset_cal.clicked.connect(self._on_reset_cal_clicked)
        lc.addWidget(self.btn_reset_cal)
        l_rec.addWidget(grp_cal)

        # Baseline Snapshot Ghost Overlay
        grp_snap = QtWidgets.QGroupBox("Baseline Snapshot Ghost Overlay")
        ls = QtWidgets.QVBoxLayout(grp_snap)
        ls.addWidget(QtWidgets.QLabel("Freeze nominal spectrum as purple ghost trace to visually track drift:"))
        snap_box = QtWidgets.QHBoxLayout()
        self.btn_snapshot = QtWidgets.QPushButton("Freeze Spectrum")
        self.btn_snapshot.clicked.connect(self._on_capture_snapshot_clicked)
        snap_box.addWidget(self.btn_snapshot)

        self.btn_clear_snapshot = QtWidgets.QPushButton("Clear Ghost")
        self.btn_clear_snapshot.clicked.connect(self.dock_fft.clear_baseline_snapshot)
        snap_box.addWidget(self.btn_clear_snapshot)
        ls.addLayout(snap_box)
        l_rec.addWidget(grp_snap)

        # Flight Recorder
        grp_flight = QtWidgets.QGroupBox("Session Flight Recorder (WAV + CSV)")
        lfl = QtWidgets.QVBoxLayout(grp_flight)
        lfl.addWidget(QtWidgets.QLabel("Record synchronized raw uncompressed WAV audio + time-stamped CSV vibration logs:"))
        self.btn_flight_rec = QtWidgets.QPushButton("Start Flight Recording")
        self.btn_flight_rec.clicked.connect(self._on_toggle_flight_recorder)
        lfl.addWidget(self.btn_flight_rec)

        self.lbl_rec_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_rec_status.setStyleSheet("color: #64748b; font-size: 10px;")
        lfl.addWidget(self.lbl_rec_status)
        l_rec.addWidget(grp_flight)

        l_rec.addStretch()
        tabs.addTab(t_rec, "Calibration & Recorder")

        # --- Tab 3: Synthetic Rig Simulator ---
        t_sim = QtWidgets.QWidget()
        l_sim = QtWidgets.QVBoxLayout(t_sim)
        l_sim.setSpacing(8)

        grp_sim = QtWidgets.QGroupBox("Industrial Mechanical Rig Simulator")
        lms = QtWidgets.QVBoxLayout(grp_sim)
        lms.addWidget(QtWidgets.QLabel("Inject real physical fault signatures to benchmark and demonstrate:"))

        # RPM
        rpm_box = QtWidgets.QHBoxLayout()
        rpm_box.addWidget(QtWidgets.QLabel("Shaft RPM:"))
        self.lbl_sim_rpm = QtWidgets.QLabel("1800 RPM (30.0 Hz)")
        rpm_box.addWidget(self.lbl_sim_rpm)
        lms.addLayout(rpm_box)

        self.slider_sim_rpm = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_sim_rpm.setRange(600, 3600)
        self.slider_sim_rpm.setValue(1800)
        self.slider_sim_rpm.valueChanged.connect(self._on_sim_params_changed)
        lms.addWidget(self.slider_sim_rpm)

        # Unbalance (1X)
        unbal_box = QtWidgets.QHBoxLayout()
        unbal_box.addWidget(QtWidgets.QLabel("1X Rotor Unbalance:"))
        self.lbl_sim_unbal = QtWidgets.QLabel("35%")
        unbal_box.addWidget(self.lbl_sim_unbal)
        lms.addLayout(unbal_box)

        self.slider_sim_unbal = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_sim_unbal.setRange(0, 100)
        self.slider_sim_unbal.setValue(35)
        self.slider_sim_unbal.valueChanged.connect(self._on_sim_params_changed)
        lms.addWidget(self.slider_sim_unbal)

        # Misalignment (2X)
        mis_box = QtWidgets.QHBoxLayout()
        mis_box.addWidget(QtWidgets.QLabel("2X Misalignment:"))
        self.lbl_sim_mis = QtWidgets.QLabel("15%")
        mis_box.addWidget(self.lbl_sim_mis)
        lms.addLayout(mis_box)

        self.slider_sim_mis = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_sim_mis.setRange(0, 100)
        self.slider_sim_mis.setValue(15)
        self.slider_sim_mis.valueChanged.connect(self._on_sim_params_changed)
        lms.addWidget(self.slider_sim_mis)

        # Looseness (0.5X, 3X, 4X)
        loose_box = QtWidgets.QHBoxLayout()
        loose_box.addWidget(QtWidgets.QLabel("Mechanical Looseness:"))
        self.lbl_sim_loose = QtWidgets.QLabel("5%")
        loose_box.addWidget(self.lbl_sim_loose)
        lms.addLayout(loose_box)

        self.slider_sim_loose = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_sim_loose.setRange(0, 100)
        self.slider_sim_loose.setValue(5)
        self.slider_sim_loose.valueChanged.connect(self._on_sim_params_changed)
        lms.addWidget(self.slider_sim_loose)

        # Bearing BPFO Impulses
        b_box = QtWidgets.QHBoxLayout()
        b_box.addWidget(QtWidgets.QLabel("Bearing Race Spall (BPFO):"))
        self.lbl_sim_bearing = QtWidgets.QLabel("0%")
        b_box.addWidget(self.lbl_sim_bearing)
        lms.addLayout(b_box)

        self.slider_sim_bearing = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_sim_bearing.setRange(0, 100)
        self.slider_sim_bearing.setValue(0)
        self.slider_sim_bearing.valueChanged.connect(self._on_sim_params_changed)
        lms.addWidget(self.slider_sim_bearing)

        # Speed Sweep Toggle
        self.chk_speed_sweep = QtWidgets.QCheckBox("Run-Up / Coast-Down Speed Sweep")
        self.chk_speed_sweep.stateChanged.connect(self._on_sim_params_changed)
        lms.addWidget(self.chk_speed_sweep)

        l_sim.addWidget(grp_sim)
        l_sim.addStretch()
        tabs.addTab(t_sim, "Rig Simulator")

        dock.setWidget(tabs)
        return dock

    def _populate_devices(self):
        self.combo_devices.blockSignals(True)
        self.combo_devices.clear()
        self.combo_devices.addItem("⚙ Synthetic Mechanical Rig Simulator", -1)

        devs = AudioEngine.query_available_devices()
        def_idx = 0
        for dev in devs:
            label = f"🎤 {dev['name']}"
            if dev["is_default"]:
                label += " [Default]"
                def_idx = self.combo_devices.count()
            self.combo_devices.addItem(label, dev["index"])

        if self.combo_devices.count() > 1:
            self.combo_devices.setCurrentIndex(def_idx if def_idx > 0 else 1)
        else:
            self.combo_devices.setCurrentIndex(0)
        self.combo_devices.blockSignals(False)

    def _on_device_changed(self):
        idx = self.combo_devices.currentData()
        if idx == -1:
            self.audio_engine.start(use_synthetic=True)
        else:
            self.audio_engine.start(device_index=idx, use_synthetic=False)

    def _on_gain_changed(self, val: int):
        g = val / 10.0
        self.lbl_gain.setText(f"{g:.1f}x")
        self.audio_engine.set_gain(g)

    def _on_window_changed(self, idx: int):
        w = self.combo_window.currentData()
        self.dsp_worker.update_dsp_params(window_type=w)

    def _on_fft_size_changed(self, idx: int):
        sz = self.combo_fft_size.currentData()
        self.dsp_worker.update_dsp_params(fft_size=sz)

    def _on_filter_changed(self):
        en = self.chk_bp_filter.isChecked()
        low = self.spin_low_cutoff.value()
        high = self.spin_high_cutoff.value()
        self.dsp_worker.update_dsp_params(filter_enabled=en, filter_low_hz=low, filter_high_hz=high)

    def _on_calibrate_clicked(self):
        self.dsp_worker.start_calibration(duration_sec=10.0)
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setText("Recording Baseline (10s)...")

    def _on_reset_cal_clicked(self):
        self.dsp_worker.reset_calibration()
        self.progress_cal.setValue(0)
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("Calibrate Baseline (10s)")

    def _on_capture_snapshot_clicked(self):
        if self._latest_payload is not None:
            self.dock_fft.capture_baseline_snapshot(
                self._latest_payload.freqs_hz,
                self._latest_payload.amp_spectrum,
            )

    def _on_sim_params_changed(self):
        rpm = float(self.slider_sim_rpm.value())
        unbal = self.slider_sim_unbal.value() / 100.0
        mis = self.slider_sim_mis.value() / 100.0
        loose = self.slider_sim_loose.value() / 100.0
        bearing = self.slider_sim_bearing.value() / 100.0
        sweep = self.chk_speed_sweep.isChecked()

        f0 = rpm / 60.0
        self.lbl_sim_rpm.setText(f"{int(rpm)} RPM ({f0:.1f} Hz)")
        self.lbl_sim_unbal.setText(f"{int(unbal * 100)}%")
        self.lbl_sim_mis.setText(f"{int(mis * 100)}%")
        self.lbl_sim_loose.setText(f"{int(loose * 100)}%")
        self.lbl_sim_bearing.setText(f"{int(bearing * 100)}%")

        self.synthetic_gen.update_config(
            base_rpm=rpm,
            unbalance_severity=unbal,
            misalignment_severity=mis,
            looseness_severity=loose,
            bearing_fault_severity=bearing,
            enable_speed_sweep=sweep,
        )

    def _on_toggle_flight_recorder(self):
        """Toggle session flight recording on/off and dump files."""
        if not self._is_recording_flight:
            # Start flight recorder
            self._is_recording_flight = True
            self._flight_log_rows.clear()
            self.audio_engine.start_flight_recording()
            self.btn_flight_rec.setText("⏹ Stop & Save Flight Session")
            self.btn_flight_rec.setObjectName("btnRecordActive")
            self.btn_flight_rec.setStyleSheet("background-color: #881337; border: 1px solid #f43f5e; color: #ffe4e6;")
            self.lbl_rec_status.setText("Recording session in progress...")
            self.lbl_rec_status.setStyleSheet("color: #f43f5e; font-weight: bold;")
        else:
            # Stop and save flight recording
            self._is_recording_flight = False
            self.btn_flight_rec.setText("Start Flight Recording")
            self.btn_flight_rec.setObjectName("")
            self.btn_flight_rec.setStyleSheet("")

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            records_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "flight_recordings"))
            os.makedirs(records_dir, exist_ok=True)

            wav_path = os.path.join(records_dir, f"vibration_flight_{timestamp_str}.wav")
            csv_path = os.path.join(records_dir, f"vibration_flight_{timestamp_str}.csv")

            # Save WAV
            samples_written = self.audio_engine.stop_and_save_flight_recording(wav_path)

            # Save synchronized CSV logs
            if self._flight_log_rows:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(self._flight_log_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(self._flight_log_rows)

            save_msg = f"Saved: {os.path.basename(wav_path)}"
            self.lbl_rec_status.setText(save_msg)
            self.lbl_rec_status.setStyleSheet("color: #00ff88; font-weight: bold;")
            self.statusBar().showMessage(f"FLIGHT RECORDING SAVED: {wav_path} | {csv_path}", 10000)
            logger.info(f"Flight recording saved: WAV={wav_path}, CSV={csv_path}")

    def _on_render_tick(self):
        """Paced rendering tick triggered on GUI thread."""
        if self._is_rendering_frame:
            return
        payload = self.dsp_worker.get_latest_payload()
        if payload is not None and payload.timestamp > self._last_rendered_timestamp:
            self._last_rendered_timestamp = payload.timestamp
            self._is_rendering_frame = True
            try:
                self.dsp_worker._gui_busy = True
                self._on_processed_frame_ready(payload)
            finally:
                self.dsp_worker._gui_busy = False
                self._is_rendering_frame = False

    @Slot(object)
    def _on_processed_frame_ready(self, payload: DSPFramePayload):
        """
        Asynchronous slot invoked by DSPWorkerThread at 60 FPS.
        Updates all PyQtGraph viewports and telemetry cards.
        """
        self._latest_payload = payload

        # 1. Update HUD Ribbon
        self.hud_ribbon.update_telemetry(payload)

        # 2. Update Calibration Progress
        if payload.is_calibrating:
            self.progress_cal.setValue(int(payload.calibration_progress))
        elif not self.btn_calibrate.isEnabled() and self.dsp_worker.is_calibrated:
            self.progress_cal.setValue(100)
            self.btn_calibrate.setEnabled(True)
            self.btn_calibrate.setText("Calibrate Baseline (10s)")

        # 3. Update All 5 Viewports
        if self.dock_osc.isVisible():
            self.dock_osc.update_data(payload)

        if self.dock_fft.isVisible():
            self.dock_fft.update_data(payload)

        if self.dock_spec2d.isVisible():
            self.dock_spec2d.update_data(payload)

        if self.dock_casc3d.isVisible():
            self.dock_casc3d.update_data(payload)

        if self.dock_env_cep.isVisible():
            self.dock_env_cep.update_data(payload)

        # 4. Flight Recorder Frame Accumulator
        if self._is_recording_flight:
            self._flight_log_rows.append({
                "timestamp": datetime.now().isoformat(),
                "rpm": payload.calculated_rpm,
                "f0_hz": payload.dominant_frequency_hz,
                "rms_velocity": payload.rms * 60.0,
                "rms_acc": payload.rms,
                "peak": payload.peak,
                "crest_factor": payload.crest_factor,
                "kurtosis": payload.kurtosis,
                "thd_percent": payload.thd_percent,
                "health_index": payload.health_index_percent,
                "anomaly_score": payload.anomaly_score,
                "iso_zone": payload.iso_zone.name,
                "fault_driver": payload.fault_driver,
            })

        # 5. Status Bar Diagnostics
        diag = self.audio_engine.diagnostics
        self.statusBar().showMessage(
            f"DIAGNOSTIC STATUS: {payload.fault_driver.upper()} | "
            f"Overruns: {diag.buffer_overruns} | Underruns: {diag.buffer_underruns} | "
            f"Latency: {diag.latency_ms:.1f}ms | Device: {diag.active_device_name}"
        )

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Clean shutdown."""
        logger.info("Initiating graceful shutdown...")
        if hasattr(self, "_render_timer"):
            self._render_timer.stop()
        self.dsp_worker.stop_worker()
        if not self.dsp_worker.wait(1500):
            self.dsp_worker.terminate()
            self.dsp_worker.wait(500)
        self.audio_engine.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Industrial Mechanical Vibration Analyzer Workstation")
    window = VibrationWorkstationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
