"""
Main Application Module
Integrates all components of the SSVEP BCI system
"""

import sys
import time
import threading
import numpy as np
import argparse
import traceback

# Import project modules
from hardware.brainbit_interface import BrainBitInterface
from signal_processing.eeg_processor import EEGProcessor
from signal_processing.ssvep_detector import SSVEPDetector
from ml.svm_classifier import SSVEPClassifier
from gui.control_panel import ControlPanel
from gui.stimulus_window import StimulusWindow

class SSVEPBCISystem:
    """Main SSVEP BCI System class that coordinates all components"""
    
    def __init__(self, config_path="../config/settings.yaml"):
        """Initialize the SSVEP BCI system
        
        Args:
            config_path: Path to configuration file
        """
        print("Initializing SSVEP BCI System...")
        
        # Initialize all components
        self.hardware = BrainBitInterface(config_path)
        self.eeg_processor = EEGProcessor(config_path)
        self.ssvep_detector = SSVEPDetector(config_path)
        self.classifier = SSVEPClassifier(config_path)
        
        # GUI components
        self.control_panel = None
        self.stimulus_window = None
        
        # System state
        self.is_running = False
        self.is_collecting = False
        self.is_calibrating = False
        self.current_mode = "IDLE"  # IDLE, CALIB_FREQ1, CALIB_FREQ2, DETECTION
        
        # Data collection thread
        self.collection_thread = None
        self.thread_running = False
        
        # Current predictions
        self.last_prediction = None
        self.last_confidence = 0.0
        
        print("System initialization complete")
    
    def start_system(self, gui_mode=True):
        """Start the BCI system
        
        Args:
            gui_mode: Whether to start with GUI (default: True)
        """
        print("Starting SSVEP BCI System...")
        
        try:
            # Initialize hardware
            if not self.hardware.connect_device():
                print("Warning: Could not connect to BrainBit device")
                print("You can still test the system without hardware")
            
            # Try to load a pre-trained model
            if self.classifier.load_model():
                print("Pre-trained model loaded successfully")
            else:
                print("No pre-trained model found. Calibration will be required.")
            
            if gui_mode:
                self._start_gui_mode()
            else:
                self._start_console_mode()
                
        except Exception as e:
            print(f"Error starting system: {e}")
            traceback.print_exc()
    
    def _start_gui_mode(self):
        """Start the system with GUI"""
        print("Starting GUI mode...")
        
        # Create control panel
        self.control_panel = ControlPanel()
        
        # Register callbacks
        self.control_panel.register_callback('start_collection', self.start_data_collection)
        self.control_panel.register_callback('stop_collection', self.stop_data_collection)
        self.control_panel.register_callback('calibrate', self.start_calibration)
        self.control_panel.register_callback('toggle_stimulus', self.toggle_stimulus_window)
        self.control_panel.register_callback('get_eeg_data', self.get_current_eeg_data)
        self.control_panel.register_callback('get_signal_quality', self.get_signal_quality)
        self.control_panel.register_callback('get_predictions', self.get_latest_prediction)
        
        # Create stimulus window
        self.stimulus_window = StimulusWindow()
        
        # Start the control panel (blocking)
        self.is_running = True
        self.control_panel.run()
    
    def _start_console_mode(self):
        """Start the system in console mode for testing"""
        print("Starting console mode...")
        print("Commands: start, stop, calibrate, detect, quit")
        
        self.is_running = True
        
        while self.is_running:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "start":
                    self.start_data_collection()
                elif command == "stop":
                    self.stop_data_collection()
                elif command == "calibrate":
                    self.start_calibration()
                elif command == "detect":
                    self.start_detection()
                elif command in ["quit", "exit", "q"]:
                    self.shutdown()
                    break
                else:
                    print("Unknown command. Use: start, stop, calibrate, detect, quit")
                    
            except KeyboardInterrupt:
                print("\nShutting down...")
                self.shutdown()
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def start_data_collection(self):
        """Start EEG data collection"""
        if self.is_collecting:
            print("Data collection already running")
            return
        
        print("Starting data collection...")
        
        # Start LSL stream
        if not self.hardware.start_lsl_stream():
            print("Warning: Could not start LSL stream")
        
        # Start collection thread
        self.thread_running = True
        self.collection_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.is_collecting = True
        print("Data collection started")
    
    def stop_data_collection(self):
        """Stop EEG data collection"""
        if not self.is_collecting:
            print("Data collection not running")
            return
        
        print("Stopping data collection...")
        
        # Stop collection thread
        self.thread_running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
        
        # Stop LSL stream
        self.hardware.stop_lsl_stream()
        
        self.is_collecting = False
        self.current_mode = "IDLE"
        print("Data collection stopped")
    
    def _data_collection_loop(self):
        """Main data collection loop"""
        print("Data collection loop started")
        
        while self.thread_running and self.is_collecting:
            try:
                # Get new EEG data
                new_data = self.hardware.get_eeg_data()
                
                if new_data is not None and new_data.size > 0:
                    # Update EEG processor buffer
                    self.eeg_processor.update_buffer(new_data)
                    
                    # Get current buffer for analysis
                    current_buffer = self.eeg_processor.get_buffer_data()
                    
                    # Preprocess the data
                    processed_data = self.eeg_processor.preprocess_eeg(current_buffer)
                    
                    # Get signal quality
                    signal_quality = self.eeg_processor.get_signal_quality()
                    
                    # Perform frequency analysis
                    freq_bins, power_spectrum = self.eeg_processor.compute_fft(processed_data)
                    
                    # Update hardware impedance if available
                    impedance = self.hardware.get_impedance_values()
                    if impedance:
                        self.eeg_processor.set_impedance_values(impedance)
                    
                    # Handle different modes
                    if self.current_mode == "DETECTION":
                        self._handle_detection_mode(freq_bins, power_spectrum, processed_data, signal_quality)
                    elif self.current_mode in ["CALIB_FREQ1", "CALIB_FREQ2"]:
                        self._handle_calibration_mode(freq_bins, power_spectrum, processed_data, signal_quality)
                
                time.sleep(0.05)  # 20 Hz update rate
                
            except Exception as e:
                print(f"Error in data collection loop: {e}")
                time.sleep(0.1)
        
        print("Data collection loop ended")
    
    def _handle_detection_mode(self, freq_bins, power_spectrum, eeg_data, signal_quality):
        """Handle detection mode processing"""
        try:
            # Extract features for classification
            features = self.ssvep_detector.extract_features(
                freq_bins, power_spectrum, eeg_data, signal_quality
            )
            
            # Classify using SVM if trained
            if self.classifier.is_trained:
                prediction, confidence = self.classifier.predict(features)
                self.last_prediction = prediction
                self.last_confidence = confidence
            else:
                # Use simple detection method
                prediction, confidence = self.ssvep_detector.simple_classify(
                    freq_bins, power_spectrum, signal_quality
                )
                self.last_prediction = prediction
                self.last_confidence = confidence
            
        except Exception as e:
            print(f"Error in detection mode: {e}")
    
    def _handle_calibration_mode(self, freq_bins, power_spectrum, eeg_data, signal_quality):
        """Handle calibration mode processing"""
        try:
            # Extract features
            features = self.ssvep_detector.extract_features(
                freq_bins, power_spectrum, eeg_data, signal_quality
            )
            
            # Determine label based on current calibration mode
            if self.current_mode == "CALIB_FREQ1":
                label = 0  # Frequency 1
            elif self.current_mode == "CALIB_FREQ2":
                label = 1  # Frequency 2
            else:
                return
            
            # Check if we have a strong SSVEP response before adding to training
            detection_result = self.ssvep_detector.detect_peak_frequency(
                freq_bins, power_spectrum, signal_quality
            )
            
            if detection_result:
                freq1_peak, freq2_peak, _ = detection_result
                
                # Add to training data if there's a clear response
                min_peak_threshold = 0.1
                if (label == 0 and freq1_peak > min_peak_threshold) or \
                   (label == 1 and freq2_peak > min_peak_threshold):
                    self.classifier.add_training_data(features, label)
                    print(f"Added training sample for class {label}")
            
        except Exception as e:
            print(f"Error in calibration mode: {e}")
    
    def start_calibration(self):
        """Start the calibration process"""
        if not self.is_collecting:
            print("Please start data collection first")
            return
        
        print("Starting calibration process...")
        
        # Clear any existing training data
        self.classifier.clear_training_data()
        
        # Create stimulus window if not exists
        if self.stimulus_window is None:
            self.stimulus_window = StimulusWindow()
        
        if not self.stimulus_window.is_window_open():
            self.stimulus_window.create_window()
        
        # Start calibration sequence
        self._run_calibration_sequence()
    
    def _run_calibration_sequence(self):
        """Run the complete calibration sequence"""
        print("\\n=== Calibration Sequence ===")
        print("Look at the stimulus and try to focus on it when it's flashing")
        
        # Calibration for frequency 1 (8 Hz)
        print(f"\\nStep 1: Calibrating for {self.ssvep_detector.freq1} Hz (scroll up)")
        print("Focus on the TOP box when it flashes...")
        
        self.current_mode = "CALIB_FREQ1"
        self.stimulus_window.set_mode("CALIB_FREQ1")
        
        # Collect data for 30 seconds
        time.sleep(30)
        
        # Calibration for frequency 2 (13 Hz)
        print(f"\\nStep 2: Calibrating for {self.ssvep_detector.freq2} Hz (scroll down)")
        print("Focus on the BOTTOM box when it flashes...")
        
        self.current_mode = "CALIB_FREQ2"
        self.stimulus_window.set_mode("CALIB_FREQ2")
        
        # Collect data for 30 seconds
        time.sleep(30)
        
        # Train the classifier
        print("\\nTraining classifier...")
        if self.classifier.can_train():
            if self.classifier.train():
                print("Calibration successful!")
                
                # Save the trained model
                self.classifier.save_model()
                
                # Switch to detection mode
                self.start_detection()
            else:
                print("Calibration failed. Please try again.")
                self.current_mode = "IDLE"
                self.stimulus_window.set_mode("IDLE")
        else:
            print("Not enough training data collected. Please try calibration again.")
            self.current_mode = "IDLE"
            self.stimulus_window.set_mode("IDLE")
    
    def start_detection(self):
        """Start detection mode"""
        if not self.is_collecting:
            print("Please start data collection first")
            return
        
        print("Starting detection mode...")
        self.current_mode = "DETECTION"
        
        # Create stimulus window if not exists
        if self.stimulus_window is None:
            self.stimulus_window = StimulusWindow()
        
        if not self.stimulus_window.is_window_open():
            self.stimulus_window.create_window()
        
        self.stimulus_window.set_mode("DETECTION")
        print("Detection active. Look at the stimuli to control scrolling.")
    
    def toggle_stimulus_window(self):
        """Toggle the stimulus window"""
        if self.stimulus_window is None:
            self.stimulus_window = StimulusWindow()
        
        self.stimulus_window.toggle_window()
    
    def get_current_eeg_data(self):
        """Get current EEG data for display
        
        Returns:
            numpy.ndarray: Current EEG buffer data
        """
        try:
            return self.eeg_processor.get_buffer_data()
        except Exception:
            return None
    
    def get_signal_quality(self):
        """Get current signal quality metrics
        
        Returns:
            list: Signal quality percentages
        """
        try:
            return self.eeg_processor.get_signal_quality()
        except Exception:
            return [0, 0, 0, 0]
    
    def get_latest_prediction(self):
        """Get the latest prediction and confidence
        
        Returns:
            tuple: (prediction, confidence)
        """
        return self.last_prediction, self.last_confidence
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        print("Shutting down SSVEP BCI System...")
        
        # Stop data collection
        if self.is_collecting:
            self.stop_data_collection()
        
        # Close stimulus window
        if self.stimulus_window:
            self.stimulus_window.close_window()
        
        # Close control panel
        if self.control_panel:
            self.control_panel.close_window()
        
        # Disconnect hardware
        self.hardware.disconnect_device()
        
        self.is_running = False
        print("System shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SSVEP BCI System")
    parser.add_argument("--no-gui", action="store_true", 
                       help="Run in console mode without GUI")
    parser.add_argument("--config", type=str, default="../config/settings.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Create and start the system
        system = SSVEPBCISystem(config_path=args.config)
        system.start_system(gui_mode=not args.no_gui)
        
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        print("Program ended")

if __name__ == "__main__":
    main() 