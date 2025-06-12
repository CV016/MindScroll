"""
BrainBit Hardware Interface
Handles connection to BrainBit EEG devices and LSL streaming
"""

import sys
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet
import yaml
import os

try:
    from neurosdk.scanner import Scanner
    from neurosdk.cmn_types import *
except ImportError as e:
    print(f"Warning: neurosdk not available. {e}")
    Scanner = None

class BrainBitInterface:
    """Interface for BrainBit EEG devices with LSL streaming capability"""
    
    def __init__(self, config_path="../config/settings.yaml"):
        """Initialize BrainBit interface
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Device properties
        self.scanner = None
        self.sensor = None
        self.outlet = None
        self.is_streaming = False
        
        # Configuration
        self.fs = self.config['signal_processing']['fs']
        self.n_channels = len(self.config['channels']['names'])
        self.channel_names = self.config['channels']['names']
        
        # Initialize scanner if neurosdk is available
        if Scanner is not None:
            self._init_scanner()
        else:
            raise ImportError("neurosdk is required for BrainBit interface")
        
        # Initialize LSL outlet
        self._init_lsl_outlet()
    
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
                'signal_processing': {'fs': 250},
                'channels': {'names': ['O1', 'O2', 'T3', 'T4']},
                'lsl': {'stream_name': 'BrainBit', 'stream_type': 'EEG'}
            }
    
    def _init_scanner(self):
        """Initialize the BrainBit scanner"""
        try:
            self.scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack])
            print("BrainBit scanner initialized")
        except Exception as e:
            print(f"Failed to initialize scanner: {e}")
            raise
    
    def _init_lsl_outlet(self):
        """Initialize LSL outlet for streaming"""
        try:
            # Create stream info
            stream_name = self.config['lsl']['stream_name']
            stream_type = self.config['lsl']['stream_type']
            
            self.info = StreamInfo(
                name=stream_name,
                type=stream_type,
                channel_count=self.n_channels,
                nominal_srate=self.fs,
                channel_format='float32',
                source_id='brainbit_serial'
            )
            
            # Add channel information
            channels = self.info.desc().append_child("channels")
            for channel_name in self.channel_names:
                ch = channels.append_child("channel")
                ch.append_child_value("label", channel_name)
            
            # Create outlet
            self.outlet = StreamOutlet(self.info)
            print(f"LSL outlet created: {stream_name} ({stream_type})")
            
        except Exception as e:
            print(f"Failed to create LSL outlet: {e}")
            raise
    
    def on_sensor_found(self, scanner, sensors):
        """Callback for when sensors are found"""
        print(f"Found {len(sensors)} sensor(s)")
        for i, sensor_info in enumerate(sensors):
            print(f"Sensor {i}: {sensor_info.Name} ({sensor_info.SerialNumber})")
    
    def on_signal_received(self, sensor, data):
        """Callback for receiving EEG data"""
        if not self.outlet:
            return
            
        for sample in data:
            # Create sample array with data from all 4 channels
            sample_data = [sample.O1, sample.O2, sample.T3, sample.T4]
            # Push sample to LSL
            self.outlet.push_sample(sample_data)
    
    def on_state_changed(self, sensor, state):
        """Callback for sensor state changes"""
        print(f"Sensor state changed: {state}")
    
    def search_sensors(self, search_time=5):
        """Search for available BrainBit sensors
        
        Args:
            search_time: Time to search in seconds
            
        Returns:
            List of found sensors
        """
        if not self.scanner:
            return []
            
        print(f"Searching for sensors for {search_time} seconds...")
        self.scanner.sensorsChanged = self.on_sensor_found
        self.scanner.start()
        time.sleep(search_time)
        self.scanner.stop()
        return self.scanner.sensors()
    
    def connect_to_sensor(self, sensor_index=0):
        """Connect to a sensor by index
        
        Args:
            sensor_index: Index of sensor to connect to
            
        Returns:
            bool: True if connection successful
        """
        sensors = self.scanner.sensors()
        if not sensors:
            print("No sensors found")
            return False
        
        if sensor_index >= len(sensors):
            print(f"Invalid sensor index: {sensor_index}")
            return False
        
        try:
            self.sensor = self.scanner.create_sensor(sensors[sensor_index])
            self.sensor.sensorStateChanged = self.on_state_changed
            self.sensor.signalDataReceived = self.on_signal_received
            
            # Wait for sensor to connect
            timeout = 10  # seconds
            start_time = time.time()
            
            while self.sensor.state != SensorState.StateInRange:
                if time.time() - start_time > timeout:
                    print("Connection timeout")
                    return False
                    
                time.sleep(0.5)
                if self.sensor.state == SensorState.StateDisconnected:
                    print("Failed to connect to sensor")
                    return False
            
            print(f"Connected to {self.sensor.name} ({self.sensor.serial_number})")
            return True
            
        except Exception as e:
            print(f"Error connecting to sensor: {e}")
            return False
    
    def start_stream(self):
        """Start streaming EEG data to LSL
        
        Returns:
            bool: True if streaming started successfully
        """
        if not self.sensor:
            print("No sensor connected")
            return False
        
        if self.is_streaming:
            print("Already streaming")
            return True
        
        try:
            if self.sensor.is_supported_command(SensorCommand.StartSignal):
                self.sensor.exec_command(SensorCommand.StartSignal)
                self.is_streaming = True
                print("Started streaming EEG data to LSL")
                return True
            else:
                print("StartSignal command not supported")
                return False
        except Exception as e:
            print(f"Error starting stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop streaming EEG data"""
        if not self.sensor or not self.is_streaming:
            return
        
        try:
            if self.sensor.is_supported_command(SensorCommand.StopSignal):
                self.sensor.exec_command(SensorCommand.StopSignal)
                self.is_streaming = False
                print("Stopped streaming")
        except Exception as e:
            print(f"Error stopping stream: {e}")
    
    def disconnect(self):
        """Disconnect from sensor and clean up"""
        self.stop_stream()
        
        if self.sensor:
            try:
                self.sensor.disconnect()
                print("Disconnected from sensor")
            except Exception as e:
                print(f"Error disconnecting: {e}")
            
            del self.sensor
            self.sensor = None
        
        if self.scanner:
            del self.scanner
            self.scanner = None
    
    def get_device_info(self):
        """Get information about connected device
        
        Returns:
            dict: Device information
        """
        if not self.sensor:
            return None
            
        return {
            'name': self.sensor.name,
            'serial_number': self.sensor.serial_number,
            'state': self.sensor.state,
            'is_streaming': self.is_streaming,
            'sample_rate': self.fs,
            'channels': self.channel_names
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def main():
    """Main function for standalone execution"""
    try:
        # Create BrainBit interface
        brain_interface = BrainBitInterface()
        
        # Search for sensors
        sensors = brain_interface.search_sensors(5)
        
        if not sensors:
            print("No sensors found. Exiting.")
            sys.exit(1)
        
        # Connect to first sensor
        if not brain_interface.connect_to_sensor(0):
            print("Failed to connect to sensor. Exiting.")
            sys.exit(1)
        
        # Start streaming
        if not brain_interface.start_stream():
            print("Failed to start streaming. Exiting.")
            sys.exit(1)
        
        print("Streaming EEG data to LSL. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            brain_interface.disconnect()
            print("Exited cleanly")
    
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main() 