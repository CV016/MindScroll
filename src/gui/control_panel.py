"""
Control Panel Module
Main GUI interface for controlling the SSVEP BCI system
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pyautogui
import threading
import time
import yaml
import os

class ControlPanel:
    """Main control panel for the SSVEP BCI system"""
    
    def __init__(self, config_path="../../config/settings.yaml"):
        """Initialize control panel
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Main window
        self.root = None
        self.is_open = False
        
        # Control variables
        self.is_collecting = False
        self.is_scrolling_enabled = True
        self.scroll_sensitivity = 3
        
        # Data tracking
        self.session_predictions = []
        self.session_confidences = []
        self.session_start_time = None
        
        # GUI elements
        self.status_var = None
        self.confidence_var = None
        self.scroll_enabled_var = None
        self.scroll_sens_var = None
        
        # Real-time plot elements
        self.fig = None
        self.canvas = None
        self.plot_axes = None
        
        # Threading
        self.update_thread = None
        self.thread_running = False
        
        # Callbacks
        self.callbacks = {
            'start_collection': None,
            'stop_collection': None,
            'calibrate': None,
            'get_eeg_data': None,
            'get_signal_quality': None,
            'get_predictions': None,
            'toggle_stimulus': None
        }
    
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
                'gui': {
                    'window': {'width': 800, 'height': 600},
                    'colors': {'primary': '#007acc', 'secondary': '#e0e0e0'},
                    'fonts': {'family': 'Arial', 'size_normal': 10, 'size_large': 12}
                },
                'channels': {'names': ['O1', 'O2', 'T3', 'T4']},
                'frequency': {'flicker_freq_1': 8.0, 'flicker_freq_2': 13.0}
            }
    
    def create_window(self):
        """Create the main control panel window"""
        if self.is_open:
            return
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("SSVEP BCI Control Panel")
        self.root.geometry(f"{self.config['gui']['window']['width']}x{self.config['gui']['window']['height']}")
        
        # Configure style
        self.root.configure(bg=self.config['gui']['colors']['secondary'])
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_control_tab()
        self._create_monitoring_tab()
        self._create_settings_tab()
        
        # Set up closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        
        # Start update thread
        self._start_update_thread()
        
        self.is_open = True
        print("Control panel window created")
    
    def _create_control_tab(self):
        """Create the main control tab"""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Control")
        
        # Status section
        status_frame = ttk.LabelFrame(control_frame, text="System Status", padding=10)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Disconnected")
        self.confidence_var = tk.StringVar(value="0%")
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w")
        
        ttk.Label(status_frame, text="Confidence:").grid(row=1, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.confidence_var).grid(row=1, column=1, sticky="w")
        
        # Control buttons section
        buttons_frame = ttk.LabelFrame(control_frame, text="Control", padding=10)
        buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Data collection controls
        ttk.Button(buttons_frame, text="Start Collection", 
                  command=self._on_start_collection).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Stop Collection", 
                  command=self._on_stop_collection).grid(row=0, column=1, padx=5, pady=5)
        
        # Calibration controls
        ttk.Button(buttons_frame, text="Calibrate System", 
                  command=self._on_calibrate).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Toggle Stimulus", 
                  command=self._on_toggle_stimulus).grid(row=1, column=1, padx=5, pady=5)
        
        # Scroll control section
        scroll_frame = ttk.LabelFrame(control_frame, text="Scroll Control", padding=10)
        scroll_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.scroll_enabled_var = tk.BooleanVar(value=self.is_scrolling_enabled)
        ttk.Checkbutton(scroll_frame, text="Enable Scrolling", 
                       variable=self.scroll_enabled_var, 
                       command=self._on_scroll_toggle).grid(row=0, column=0, columnspan=2, sticky="w")
        
        ttk.Label(scroll_frame, text="Sensitivity:").grid(row=1, column=0, sticky="w")
        self.scroll_sens_var = tk.IntVar(value=self.scroll_sensitivity)
        sensitivity_scale = ttk.Scale(scroll_frame, from_=1, to=10, 
                                     variable=self.scroll_sens_var, 
                                     orient=tk.HORIZONTAL,
                                     command=self._on_sensitivity_change)
        sensitivity_scale.grid(row=1, column=1, sticky="ew", padx=5)
        
        # Configure grid weights
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        scroll_frame.columnconfigure(1, weight=1)
    
    def _create_monitoring_tab(self):
        """Create the monitoring/visualization tab"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="Monitor")
        
        # Signal quality section
        quality_frame = ttk.LabelFrame(monitor_frame, text="Signal Quality", padding=10)
        quality_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Create quality indicators for each channel
        self.quality_vars = []
        channel_names = self.config['channels']['names']
        for i, channel in enumerate(channel_names):
            ttk.Label(quality_frame, text=f"{channel}:").grid(row=i, column=0, sticky="w")
            var = tk.StringVar(value="0%")
            self.quality_vars.append(var)
            ttk.Label(quality_frame, textvariable=var).grid(row=i, column=1, sticky="w")
        
        # Real-time plot section
        plot_frame = ttk.LabelFrame(monitor_frame, text="Real-time EEG", padding=10)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig, self.plot_axes = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.suptitle("EEG Channels")
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        for i, ax in enumerate(self.plot_axes.flat):
            if i < len(channel_names):
                ax.set_title(channel_names[i])
                ax.set_ylim(-100, 100)
                ax.grid(True)
            else:
                ax.set_visible(False)
        
        # Configure grid weights
        monitor_frame.columnconfigure(0, weight=1)
        monitor_frame.rowconfigure(1, weight=1)
        quality_frame.columnconfigure(1, weight=1)
    
    def _create_settings_tab(self):
        """Create the settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Frequency settings
        freq_frame = ttk.LabelFrame(settings_frame, text="SSVEP Frequencies", padding=10)
        freq_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Frequency 1 (8 Hz)
        ttk.Label(freq_frame, text="Frequency 1 (Hz):").grid(row=0, column=0, sticky="w")
        self.freq1_var = tk.DoubleVar(value=self.config['frequency']['flicker_freq_1'])
        freq1_spin = ttk.Spinbox(freq_frame, from_=5.0, to=20.0, increment=0.5, 
                                textvariable=self.freq1_var, width=10)
        freq1_spin.grid(row=0, column=1, sticky="w", padx=5)
        
        # Frequency 2 (13 Hz)
        ttk.Label(freq_frame, text="Frequency 2 (Hz):").grid(row=1, column=0, sticky="w")
        self.freq2_var = tk.DoubleVar(value=self.config['frequency']['flicker_freq_2'])
        freq2_spin = ttk.Spinbox(freq_frame, from_=5.0, to=20.0, increment=0.5, 
                                textvariable=self.freq2_var, width=10)
        freq2_spin.grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Button(freq_frame, text="Update Frequencies", 
                  command=self._on_update_frequencies).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Session statistics
        stats_frame = ttk.LabelFrame(settings_frame, text="Session Statistics", padding=10)
        stats_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.session_time_var = tk.StringVar(value="00:00:00")
        self.predictions_count_var = tk.StringVar(value="0")
        self.avg_confidence_var = tk.StringVar(value="0%")
        
        ttk.Label(stats_frame, text="Session Time:").grid(row=0, column=0, sticky="w")
        ttk.Label(stats_frame, textvariable=self.session_time_var).grid(row=0, column=1, sticky="w")
        
        ttk.Label(stats_frame, text="Predictions:").grid(row=1, column=0, sticky="w")
        ttk.Label(stats_frame, textvariable=self.predictions_count_var).grid(row=1, column=1, sticky="w")
        
        ttk.Label(stats_frame, text="Avg Confidence:").grid(row=2, column=0, sticky="w")
        ttk.Label(stats_frame, textvariable=self.avg_confidence_var).grid(row=2, column=1, sticky="w")
        
        ttk.Button(stats_frame, text="Reset Statistics", 
                  command=self._on_reset_stats).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Configure grid weights
        settings_frame.columnconfigure(0, weight=1)
        freq_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(1, weight=1)
    
    def _start_update_thread(self):
        """Start the background update thread"""
        if self.update_thread is not None and self.update_thread.is_alive():
            return
        
        self.thread_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Background update loop for real-time data"""
        while self.thread_running and self.is_open:
            try:
                self._update_displays()
                time.sleep(0.1)  # Update 10 times per second
            except Exception as e:
                print(f"Error in update loop: {e}")
        
        print("Update thread stopped")
    
    def _update_displays(self):
        """Update all display elements"""
        if not self.is_open or self.root is None:
            return
        
        # Update signal quality
        if self.callbacks['get_signal_quality']:
            try:
                signal_quality = self.callbacks['get_signal_quality']()
                if signal_quality:
                    for i, var in enumerate(self.quality_vars):
                        if i < len(signal_quality):
                            var.set(f"{signal_quality[i]:.1f}%")
            except Exception as e:
                print(f"Error updating signal quality: {e}")
        
        # Update EEG plots
        if self.callbacks['get_eeg_data']:
            try:
                eeg_data = self.callbacks['get_eeg_data']()
                if eeg_data is not None and eeg_data.size > 0:
                    self._update_plots(eeg_data)
            except Exception as e:
                print(f"Error updating plots: {e}")
        
        # Update session statistics
        self._update_session_stats()
        
        # Process scroll commands if enabled
        if self.is_scrolling_enabled and self.callbacks['get_predictions']:
            try:
                prediction, confidence = self.callbacks['get_predictions']()
                if prediction is not None and prediction != "Uncertain":
                    self._handle_scroll_command(prediction, confidence)
            except Exception as e:
                print(f"Error processing scroll commands: {e}")
    
    def _update_plots(self, eeg_data):
        """Update the real-time EEG plots"""
        try:
            # Display only the last 2 seconds of data
            samples_to_show = min(500, eeg_data.shape[1])
            data_to_plot = eeg_data[:, -samples_to_show:]
            
            for i, ax in enumerate(self.plot_axes.flat):
                if i < min(4, eeg_data.shape[0]):
                    ax.clear()
                    ax.plot(data_to_plot[i])
                    ax.set_title(self.config['channels']['names'][i])
                    ax.set_ylim(-100, 100)
                    ax.grid(True)
            
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def _update_session_stats(self):
        """Update session statistics"""
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.session_time_var.set(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        self.predictions_count_var.set(str(len(self.session_predictions)))
        
        if self.session_confidences:
            avg_conf = np.mean(self.session_confidences) * 100
            self.avg_confidence_var.set(f"{avg_conf:.1f}%")
    
    def _handle_scroll_command(self, prediction, confidence):
        """Handle scroll commands based on predictions"""
        # Record prediction
        self.session_predictions.append(prediction)
        self.session_confidences.append(confidence)
        
        # Update confidence display
        self.confidence_var.set(f"{confidence*100:.1f}%")
        
        # Perform scroll action if confidence is high enough
        min_confidence = 0.7  # Minimum confidence threshold
        if confidence >= min_confidence:
            try:
                if prediction == 0:  # Scroll up
                    pyautogui.scroll(self.scroll_sensitivity)
                    self.status_var.set("Scrolling Up")
                elif prediction == 1:  # Scroll down
                    pyautogui.scroll(-self.scroll_sensitivity)
                    self.status_var.set("Scrolling Down")
            except Exception as e:
                print(f"Error performing scroll: {e}")
        else:
            self.status_var.set("Uncertain")
    
    # Callback methods
    def _on_start_collection(self):
        """Handle start collection button"""
        if self.callbacks['start_collection']:
            self.callbacks['start_collection']()
        self.is_collecting = True
        self.session_start_time = time.time()
        self.status_var.set("Collecting Data")
    
    def _on_stop_collection(self):
        """Handle stop collection button"""
        if self.callbacks['stop_collection']:
            self.callbacks['stop_collection']()
        self.is_collecting = False
        self.status_var.set("Stopped")
    
    def _on_calibrate(self):
        """Handle calibration button"""
        if self.callbacks['calibrate']:
            self.callbacks['calibrate']()
        self.status_var.set("Calibrating...")
    
    def _on_toggle_stimulus(self):
        """Handle toggle stimulus button"""
        if self.callbacks['toggle_stimulus']:
            self.callbacks['toggle_stimulus']()
    
    def _on_scroll_toggle(self):
        """Handle scroll enable/disable"""
        self.is_scrolling_enabled = self.scroll_enabled_var.get()
        status = "Enabled" if self.is_scrolling_enabled else "Disabled"
        print(f"Scrolling {status}")
    
    def _on_sensitivity_change(self, value):
        """Handle scroll sensitivity change"""
        self.scroll_sensitivity = int(float(value))
        print(f"Scroll sensitivity set to {self.scroll_sensitivity}")
    
    def _on_update_frequencies(self):
        """Handle frequency update"""
        freq1 = self.freq1_var.get()
        freq2 = self.freq2_var.get()
        print(f"Updating frequencies to {freq1} Hz and {freq2} Hz")
        # This would typically call a callback to update the system
    
    def _on_reset_stats(self):
        """Reset session statistics"""
        self.session_predictions = []
        self.session_confidences = []
        self.session_start_time = time.time()
        self.status_var.set("Statistics Reset")
    
    def register_callback(self, event_name, callback):
        """Register a callback for specific events
        
        Args:
            event_name: Name of the event ('start_collection', 'stop_collection', etc.)
            callback: Function to call when the event occurs
        """
        if event_name in self.callbacks:
            self.callbacks[event_name] = callback
            print(f"Callback registered for {event_name}")
        else:
            print(f"Unknown callback event: {event_name}")
    
    def close_window(self):
        """Close the control panel window"""
        self.thread_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception as e:
                print(f"Error closing control panel: {e}")
            finally:
                self.root = None
                self.is_open = False
                print("Control panel window closed")
    
    def show_window(self):
        """Show the control panel window"""
        if not self.is_open:
            self.create_window()
        
        if self.root:
            self.root.deiconify()
            self.root.lift()
    
    def hide_window(self):
        """Hide the control panel window"""
        if self.root:
            self.root.withdraw()
    
    def run(self):
        """Run the control panel (blocking)"""
        if not self.is_open:
            self.create_window()
        
        if self.root:
            self.root.mainloop()
    
    def update_status(self, status):
        """Update the status display
        
        Args:
            status: New status string
        """
        if self.status_var:
            self.status_var.set(status)
    
    def is_window_open(self):
        """Check if the control panel window is open
        
        Returns:
            bool: True if window is open
        """
        return self.is_open 