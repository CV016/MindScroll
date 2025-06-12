"""
EEG Signal Processing Module
Handles filtering, preprocessing, and signal quality assessment
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import yaml
import os

class EEGProcessor:
    """Process EEG signals with filtering and quality assessment"""
    
    def __init__(self, config_path="../../config/settings.yaml"):
        """Initialize EEG processor
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Signal properties
        self.fs = self.config['signal_processing']['fs']
        self.window_length = self.config['signal_processing']['window_length']
        self.buffer_size = int(self.window_length * self.fs)
        
        # Channel configuration
        self.n_channels = len(self.config['channels']['names'])
        self.channel_names = self.config['channels']['names']
        
        # Filtering parameters
        self.bandpass_low = self.config['filtering']['bandpass_low']
        self.bandpass_high = self.config['filtering']['bandpass_high']
        self.ssvep_low = self.config['filtering']['ssvep_bandpass_low']
        self.ssvep_high = self.config['filtering']['ssvep_bandpass_high']
        self.notch_50 = self.config['filtering']['notch_freq_50']
        self.notch_60 = self.config['filtering']['notch_freq_60']
        self.notch_q = self.config['filtering']['notch_q']
        
        # Signal quality
        self.impedance_threshold = self.config['signal_quality']['impedance_threshold']
        self.signal_quality = [0, 0, 0, 0]
        self.impedance_values = [float('inf')] * 4
        
        # Initialize buffer
        self.eeg_buffer = np.zeros((self.n_channels, self.buffer_size))
    
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
                'signal_processing': {'fs': 250, 'window_length': 2},
                'channels': {'names': ['O1', 'O2', 'T3', 'T4']},
                'filtering': {
                    'bandpass_low': 1, 'bandpass_high': 50,
                    'ssvep_bandpass_low': 3, 'ssvep_bandpass_high': 45,
                    'notch_freq_50': 50, 'notch_freq_60': 60, 'notch_q': 30
                },
                'signal_quality': {'impedance_threshold': 50000}
            }
    
    def butter_bandpass(self, lowcut, highcut, order=5):
        """Design a bandpass filter
        
        Args:
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            order: Filter order
            
        Returns:
            tuple: Filter coefficients (b, a)
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data, lowcut, highcut, order=5):
        """Apply a bandpass filter to the data
        
        Args:
            data: Input signal
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            order: Filter order
            
        Returns:
            numpy.ndarray: Filtered signal
        """
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def notch_filter(self, data, freq, q=None):
        """Apply a notch filter to remove power line noise
        
        Args:
            data: Input signal
            freq: Frequency to notch out
            q: Quality factor
            
        Returns:
            numpy.ndarray: Filtered signal
        """
        if q is None:
            q = self.notch_q
            
        nyq = 0.5 * self.fs
        w0 = freq / nyq
        b, a = iirnotch(w0, q)
        y = filtfilt(b, a, data)
        return y
    
    def spatial_filter(self, eeg_data):
        """Apply Laplacian spatial filter to enhance SSVEP response
        Primarily for occipital channels (O1, O2)
        
        Args:
            eeg_data: EEG data array (channels x samples)
            
        Returns:
            numpy.ndarray: Spatially filtered data
        """
        filtered_data = np.zeros_like(eeg_data)
        
        # For BrainBit with channels ordered as [O1, O2, T3, T4]
        # For O1 (subtract average of neighbors)
        filtered_data[0] = eeg_data[0] - 0.5 * (eeg_data[1] + eeg_data[2])
        
        # For O2 (subtract average of neighbors)
        filtered_data[1] = eeg_data[1] - 0.5 * (eeg_data[0] + eeg_data[3])
        
        # Keep temporal channels mostly as-is but with some influence from occipital
        filtered_data[2] = eeg_data[2] - 0.2 * eeg_data[0]
        filtered_data[3] = eeg_data[3] - 0.2 * eeg_data[1]
        
        return filtered_data
    
    def calculate_signal_quality(self, data):
        """Calculate signal quality metrics for each channel
        
        Args:
            data: EEG data array (channels x samples)
            
        Returns:
            list: Signal quality percentages for each channel
        """
        # Update signal quality metrics (0-100%)
        for i in range(self.n_channels):
            # Calculate SNR-like metric
            # 1. Compute power in SSVEP frequency bands
            filtered_ssvep = self.butter_bandpass_filter(data[i], 7, 14)
            ssvep_power = np.mean(filtered_ssvep**2)
            
            # 2. Compute power in the rest of the band (noise)
            filtered_all = self.butter_bandpass_filter(data[i], 1, 45)
            total_power = np.mean(filtered_all**2)
            
            # Skip if total power is too low (likely disconnected)
            if total_power < 1e-12:
                self.signal_quality[i] = 0
                continue
                
            noise_power = abs(total_power - ssvep_power)
            
            # 3. Compute SNR
            if noise_power > 0:
                snr = 10 * np.log10(ssvep_power / noise_power)
                
                # Convert to 0-100% scale (-10dB to +10dB maps to 0-100%)
                snr_percent = max(0, min(100, (snr + 10) * 5))
            else:
                snr_percent = 0
            
            # 4. Incorporate impedance (if available)
            imp_factor = 1.0
            if self.impedance_values[i] < float('inf'):
                imp_factor = max(0.1, min(1.0, self.impedance_threshold / self.impedance_values[i]))
            
            # Combine metrics
            self.signal_quality[i] = snr_percent * imp_factor
        
        return self.signal_quality
    
    def preprocess_eeg(self, eeg_data):
        """Preprocess EEG data with enhanced filtering
        
        Args:
            eeg_data: Raw EEG data array (channels x samples)
            
        Returns:
            numpy.ndarray: Preprocessed EEG data
        """
        # Make a copy of the data
        processed_data = eeg_data.copy()
        
        # Apply filters to each channel
        for i in range(eeg_data.shape[0]):
            # Apply notch filter to remove power line noise (50Hz in Europe, 60Hz in US)
            processed_data[i] = self.notch_filter(processed_data[i], self.notch_50)
            processed_data[i] = self.notch_filter(processed_data[i], self.notch_60)
            
            # Apply bandpass filter (using SSVEP-specific range for better performance)
            processed_data[i] = self.butter_bandpass_filter(
                processed_data[i], self.ssvep_low, self.ssvep_high
            )
        
        # Apply spatial filtering to enhance SSVEP response
        processed_data = self.spatial_filter(processed_data)
        
        # Calculate signal quality
        self.calculate_signal_quality(processed_data)
        
        return processed_data
    
    def update_buffer(self, new_data):
        """Update the EEG buffer with new data
        
        Args:
            new_data: New EEG samples (samples x channels) or (channels x samples)
        """
        # Ensure correct shape (channels x samples)
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)
        
        if new_data.shape[0] != self.n_channels:
            # If data is (samples x channels), transpose it
            if new_data.shape[1] == self.n_channels:
                new_data = new_data.T
            else:
                raise ValueError(f"Invalid data shape: {new_data.shape}")
        
        # Shift the buffer left by the number of new samples
        samples_to_add = new_data.shape[1]
        self.eeg_buffer[:, :-samples_to_add] = self.eeg_buffer[:, samples_to_add:]
        # Add the new samples to the end of the buffer
        self.eeg_buffer[:, -samples_to_add:] = new_data
    
    def compute_fft(self, eeg_data=None):
        """Compute FFT on the EEG data
        
        Args:
            eeg_data: EEG data to analyze. If None, uses current buffer.
            
        Returns:
            tuple: (freq_bins, power_spectrum)
        """
        if eeg_data is None:
            eeg_data = self.eeg_buffer
            
        # Apply Hamming window to reduce spectral leakage
        windowed_data = eeg_data * np.hamming(eeg_data.shape[1])
        
        # Compute FFT
        fft_data = np.fft.rfft(windowed_data)
        
        # Compute frequencies for the FFT bins
        freq_bins = np.fft.rfftfreq(eeg_data.shape[1], 1/self.fs)
        
        # Compute the power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_data)**2
        
        return freq_bins, power_spectrum
    
    def get_buffer_data(self):
        """Get current buffer data
        
        Returns:
            numpy.ndarray: Current EEG buffer
        """
        return self.eeg_buffer.copy()
    
    def get_signal_quality(self):
        """Get current signal quality metrics
        
        Returns:
            list: Signal quality percentages for each channel
        """
        return self.signal_quality.copy()
    
    def set_impedance_values(self, impedance_values):
        """Set impedance values for signal quality calculation
        
        Args:
            impedance_values: List of impedance values for each channel
        """
        self.impedance_values = impedance_values.copy()
    
    def reset_buffer(self):
        """Reset the EEG buffer to zeros"""
        self.eeg_buffer = np.zeros((self.n_channels, self.buffer_size)) 