"""
SSVEP BCI - Brain-Computer Interface using Steady-State Visual Evoked Potentials
A modular implementation for real-time EEG-based control systems.
"""

__version__ = "1.0.0"
__author__ = "BCIH Team"
__description__ = "SSVEP-based Brain-Computer Interface for real-time EEG control"

# Import main components for easy access
from .hardware.brainbit_interface import BrainBitInterface
from .signal_processing.eeg_processor import EEGProcessor
from .signal_processing.ssvep_detector import SSVEPDetector
from .ml.svm_classifier import SSVEPClassifier
from .gui.control_panel import ControlPanel
from .gui.stimulus_window import StimulusWindow

__all__ = [
    'BrainBitInterface',
    'EEGProcessor', 
    'SSVEPDetector',
    'SSVEPClassifier',
    'ControlPanel',
    'StimulusWindow'
] 