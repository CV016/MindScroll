"""
SSVEP Detection Module
Implements CCA, peak detection, and feature extraction for SSVEP classification
"""

import numpy as np
from scipy import signal
import pywt
import yaml
import os

class SSVEPDetector:
    """Detect SSVEP responses using multiple methods"""
    
    def __init__(self, config_path="../../config/settings.yaml"):
        """Initialize SSVEP detector
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Frequency settings
        self.freq1 = self.config['frequency']['flicker_freq_1']
        self.freq2 = self.config['frequency']['flicker_freq_2']
        self.freq_tolerance = self.config['frequency']['freq_tolerance']
        
        # Signal processing
        self.fs = self.config['signal_processing']['fs']
        self.window_length = self.config['signal_processing']['window_length']
        self.buffer_size = int(self.window_length * self.fs)
        
        # CCA parameters
        self.n_harmonics = self.config['ml']['n_harmonics']
        
        # Wavelet parameters
        self.wavelet = self.config['ml']['wavelet']
        self.wavelet_levels = self.config['ml']['wavelet_levels']
        
        # Pre-compute reference signals for CCA
        self.ref_signals = self._generate_reference_signals()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            # Try relative to this file first
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, config_path)
            
            if not os.path.exists(full_path):
                # Try relative to current working directory
                full_path = config_path
            
            with open(full_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            # Return default configuration
            return {
                'frequency': {'flicker_freq_1': 8.0, 'flicker_freq_2': 13.0, 'freq_tolerance': 0.5},
                'signal_processing': {'fs': 250, 'window_length': 2},
                'ml': {'n_harmonics': 3, 'wavelet': 'db4', 'wavelet_levels': 5}
            }
    
    def _generate_reference_signals(self):
        """Generate reference signals for CCA
        
        Returns:
            list: Reference signals for each target frequency
        """
        freqs = [self.freq1, self.freq2]
        t = np.arange(self.buffer_size) / self.fs
        references = []
        
        for f in freqs:
            # Create reference signals with harmonics
            reference = []
            for h in range(1, self.n_harmonics + 1):
                reference.append(np.sin(2 * np.pi * h * f * t))
                reference.append(np.cos(2 * np.pi * h * f * t))
            references.append(np.array(reference))
        
        return references
    
    def compute_cca(self, X, Y):
        """Compute canonical correlation between X and Y
        
        Args:
            X: First dataset (channels x samples)
            Y: Second dataset (references x samples)
            
        Returns:
            float: Maximum canonical correlation
        """
        # Mean-center the data
        X = X - np.mean(X, axis=1, keepdims=True)
        Y = Y - np.mean(Y, axis=1, keepdims=True)
        
        # Calculate covariance matrices
        SigmaXX = np.cov(X)
        SigmaYY = np.cov(Y)
        SigmaXY = np.cov(X, Y)[:X.shape[0], X.shape[0]:]
        
        # Handle potential numerical instability
        if np.linalg.cond(SigmaXX) > 1e10 or np.linalg.cond(SigmaYY) > 1e10:
            return 0  # Return zero correlation if covariance matrices are ill-conditioned
        
        # Add small regularization to avoid singularity
        SigmaXX += np.eye(SigmaXX.shape[0]) * 1e-8
        SigmaYY += np.eye(SigmaYY.shape[0]) * 1e-8
        
        # Calculate canonical correlation
        try:
            Ax = np.linalg.solve(SigmaXX, SigmaXY)
            Ay = np.linalg.solve(SigmaYY, SigmaXY.T)
            
            eig_vals = np.linalg.eigvals(np.dot(Ax, Ay))
            # Return the square root of the largest eigenvalue
            max_corr = np.sqrt(np.max(np.real(eig_vals)))
            return max_corr
        except np.linalg.LinAlgError:
            return 0  # Return zero correlation in case of numerical issues
    
    def perform_cca(self, eeg_data):
        """Perform CCA between EEG data and reference signals
        
        Args:
            eeg_data: EEG data array (channels x samples)
            
        Returns:
            list: CCA correlations for each target frequency
        """
        X = eeg_data  # Already in correct shape
        
        # Calculate correlation for each reference signal
        cca_results = []
        for Y in self.ref_signals:
            # Compute canonical correlation
            corr = self.compute_cca(X, Y)
            cca_results.append(corr)
        
        return cca_results
    
    def compute_wavelet_features(self, eeg_data):
        """Extract wavelet features from EEG data
        
        Args:
            eeg_data: EEG data array (channels x samples)
            
        Returns:
            numpy.ndarray: Wavelet features
        """
        # Store wavelet energies for each channel and frequency band
        wavelet_features = []
        
        # Process each channel
        for ch in range(eeg_data.shape[0]):
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(eeg_data[ch], self.wavelet, level=self.wavelet_levels)
            
            # Calculate energy in each frequency band
            energies = [np.sum(c**2) for c in coeffs]
            
            # Normalize energies
            total_energy = sum(energies)
            if total_energy > 0:
                norm_energies = [e/total_energy for e in energies]
            else:
                norm_energies = [0] * len(energies)
            
            wavelet_features.extend(norm_energies)
        
        return np.array(wavelet_features)
    
    def detect_peak_frequency(self, freq_bins, power_spectrum, signal_quality=None):
        """Detect the dominant frequency using enhanced peak finding
        
        Args:
            freq_bins: Frequency bins from FFT
            power_spectrum: Power spectrum for each channel
            signal_quality: Signal quality for each channel (optional)
            
        Returns:
            tuple: (freq1_peak, freq2_peak, peak_dict) or None
        """
        # Focus on the frequency range of interest (4-30 Hz for SSVEP)
        mask = (freq_bins >= 4) & (freq_bins <= 30)
        freqs = freq_bins[mask]
        
        # Weight channels based on signal quality if provided
        if signal_quality is not None:
            channel_weights = [sq/100 for sq in signal_quality]
            # Ensure minimum weights
            channel_weights = [max(0.25, w) for w in channel_weights]
        else:
            channel_weights = [0.25, 0.25, 0.25, 0.25]
        
        # Normalize weights
        weight_sum = sum(channel_weights)
        if weight_sum > 0:
            channel_weights = [w/weight_sum for w in channel_weights]
        else:
            channel_weights = [0.25, 0.25, 0.25, 0.25]
        
        # Combine weighted power from all channels for more robust peak detection
        weighted_power = np.zeros_like(power_spectrum[0, mask])
        for i in range(power_spectrum.shape[0]):
            weighted_power += channel_weights[i] * power_spectrum[i, mask]
        
        # Smooth the spectrum for cleaner peak detection
        window_size = 3
        smoothed_power = np.convolve(weighted_power, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks in the power spectrum with enhanced criteria
        peaks, _ = signal.find_peaks(smoothed_power, 
                                   height=0.2*np.max(smoothed_power),
                                   distance=3,  # Min distance between peaks (in bins)
                                   prominence=0.1*np.max(smoothed_power))  # Require significant prominence
        
        if len(peaks) == 0:
            return None
        
        # Get the frequencies of the peaks
        peak_freqs = freqs[peaks]
        peak_powers = smoothed_power[peaks]
        
        # Create a dictionary of peak frequencies and their powers
        peak_dict = dict(zip(peak_freqs, peak_powers))
        
        # Check if the peaks match our target frequencies
        freq1_peak = 0
        freq2_peak = 0
        
        # Check for target frequencies with adaptive tolerance
        adaptive_tolerance = self.freq_tolerance
        if signal_quality is not None:
            avg_quality = np.mean(signal_quality) / 100
            adaptive_tolerance = self.freq_tolerance * (1 + (1 - avg_quality))
        
        for freq, power in peak_dict.items():
            if abs(freq - self.freq1) <= adaptive_tolerance:
                freq1_peak = power
            if abs(freq - self.freq2) <= adaptive_tolerance:
                freq2_peak = power
        
        # Check for harmonics with increased weight
        for freq, power in peak_dict.items():
            if abs(freq - 2*self.freq1) <= adaptive_tolerance:
                freq1_peak += 0.7 * power
            if abs(freq - 2*self.freq2) <= adaptive_tolerance:
                freq2_peak += 0.7 * power
            # Also check for third harmonics
            if abs(freq - 3*self.freq1) <= adaptive_tolerance:
                freq1_peak += 0.3 * power
            if abs(freq - 3*self.freq2) <= adaptive_tolerance:
                freq2_peak += 0.3 * power
        
        return freq1_peak, freq2_peak, peak_dict
    
    def extract_features(self, freq_bins, power_spectrum, eeg_data, signal_quality=None):
        """Extract comprehensive features for classification
        
        Args:
            freq_bins: Frequency bins from FFT
            power_spectrum: Power spectrum for each channel
            eeg_data: Raw EEG data
            signal_quality: Signal quality for each channel (optional)
            
        Returns:
            numpy.ndarray: Feature vector
        """
        features = []
        
        # 1. FFT-based features (expanded)
        for channel_idx in range(power_spectrum.shape[0]):
            # Focus on the frequency range of interest
            mask = (freq_bins >= 4) & (freq_bins <= 30)
            freqs = freq_bins[mask]
            powers = power_spectrum[channel_idx, mask]
            
            # Normalize powers
            if np.max(powers) > 0:
                powers = powers / np.max(powers)
            
            # Extract power at target frequencies and surrounding bins with more detail
            for target_freq in [self.freq1, self.freq2]:
                for offset in [-1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5]:
                    freq = target_freq + offset
                    # Find the closest frequency bin
                    idx = np.argmin(np.abs(freqs - freq))
                    features.append(powers[idx])
                
                # Add harmonics with more detail
                for harmonic in [2, 3]:
                    freq = target_freq * harmonic
                    if freq <= 30:  # Only if within our range
                        for offset in [-0.5, 0, 0.5]:
                            harm_freq = freq + offset
                            idx = np.argmin(np.abs(freqs - harm_freq))
                            features.append(powers[idx])
        
        # 2. Add CCA-based features
        cca_results = self.perform_cca(eeg_data)
        features.extend(cca_results)
        
        # 3. Add ratio of CCA results
        if cca_results[1] > 0:
            features.append(cca_results[0] / cca_results[1])
        else:
            features.append(0)
        
        # 4. Add wavelet features
        wavelet_features = self.compute_wavelet_features(eeg_data)
        features.extend(wavelet_features)
        
        # 5. Add signal quality metrics if available
        if signal_quality is not None:
            features.extend([sq/100 for sq in signal_quality])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])  # Default medium quality
        
        # 6. Add peak detection features
        detection_result = self.detect_peak_frequency(freq_bins, power_spectrum, signal_quality)
        if detection_result:
            freq1_peak, freq2_peak, _ = detection_result
            # Add both absolute values and ratio
            features.append(freq1_peak)
            features.append(freq2_peak)
            if freq2_peak > 0:
                features.append(freq1_peak / freq2_peak)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def simple_classify(self, freq_bins, power_spectrum, signal_quality=None):
        """Simple classification based on peak detection only
        
        Args:
            freq_bins: Frequency bins from FFT
            power_spectrum: Power spectrum for each channel
            signal_quality: Signal quality for each channel (optional)
            
        Returns:
            tuple: (prediction, confidence) where prediction is 0, 1, or "Uncertain"
        """
        detection_result = self.detect_peak_frequency(freq_bins, power_spectrum, signal_quality)
        
        if detection_result is None:
            return "Uncertain", 0.0
        
        freq1_peak, freq2_peak, _ = detection_result
        
        # Simple threshold-based classification
        threshold = 0.1  # Minimum peak power threshold
        
        if freq1_peak > threshold and freq2_peak > threshold:
            # Both frequencies detected, choose the stronger one
            if freq1_peak > freq2_peak:
                confidence = freq1_peak / (freq1_peak + freq2_peak)
                return 0, confidence
            else:
                confidence = freq2_peak / (freq1_peak + freq2_peak)
                return 1, confidence
        elif freq1_peak > threshold:
            confidence = min(0.8, freq1_peak)
            return 0, confidence
        elif freq2_peak > threshold:
            confidence = min(0.8, freq2_peak)
            return 1, confidence
        else:
            return "Uncertain", 0.0
    
    def get_target_frequencies(self):
        """Get the target SSVEP frequencies
        
        Returns:
            tuple: (freq1, freq2)
        """
        return self.freq1, self.freq2
    
    def update_frequencies(self, freq1, freq2):
        """Update target frequencies and regenerate reference signals
        
        Args:
            freq1: New first target frequency
            freq2: New second target frequency
        """
        self.freq1 = freq1
        self.freq2 = freq2
        self.ref_signals = self._generate_reference_signals()
        print(f"Updated target frequencies to {freq1} Hz and {freq2} Hz") 