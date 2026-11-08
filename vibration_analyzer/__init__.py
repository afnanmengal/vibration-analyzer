"""
vibration_analyzer - High-Performance Real-Time Mechanical Vibration Workstation
"""

from .audio_engine import AudioConfig, AudioDiagnostics, AudioEngine, AtomicAudioRingBuffer
from .dsp_worker import (
    DSPFramePayload,
    DSPWorkerThread,
    HarmonicCursorInfo,
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

__all__ = [
    "AudioConfig",
    "AudioDiagnostics",
    "AudioEngine",
    "AtomicAudioRingBuffer",
    "DSPFramePayload",
    "DSPWorkerThread",
    "HarmonicCursorInfo",
    "ISOZone",
    "MachineClass",
    "WindowType",
    "SyntheticMechanicalConfig",
    "SyntheticVibrationGenerator",
    "Cascade3DDock",
    "EnvelopeCepstrumDock",
    "HUDTelemetryRibbon",
    "ModernDockWidget",
    "NarrowbandSpectrumDock",
    "OscilloscopeDock",
    "Spectrogram2DDock",
]
