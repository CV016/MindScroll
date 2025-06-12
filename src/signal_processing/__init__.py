"""
Signal processing package for EEG analysis and SSVEP detection
"""

from .eeg_processor import EEGProcessor
from .ssvep_detector import SSVEPDetector

__all__ = ['EEGProcessor', 'SSVEPDetector'] 