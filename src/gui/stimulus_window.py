"""
Stimulus Window Module
Creates and manages the SSVEP stimulus display window
"""

import tkinter as tk
import yaml
import os

class StimulusWindow:
    """Manages the SSVEP stimulus display window"""
    
    def __init__(self, config_path="../../config/settings.yaml"):
        """Initialize stimulus window
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Window properties
        self.window = None
        self.canvas = None
        self.is_open = False
        
        # Stimulus properties
        self.freq1 = self.config['frequency']['flicker_freq_1']
        self.freq2 = self.config['frequency']['flicker_freq_2']
        self.box_width = self.config['gui']['stimulus']['box_width']
        self.box_height = self.config['gui']['stimulus']['box_height']
        self.box_margin = self.config['gui']['stimulus']['box_margin']
        
        # Colors
        self.bg_color = self.config['gui']['colors']['background']
        self.on_color = self.config['gui']['colors']['stimulus_on']
        self.off_color = self.config['gui']['colors']['stimulus_off']
        self.inactive_color = self.config['gui']['colors']['stimulus_inactive']
        
        # Fonts
        self.font_family = self.config['gui']['fonts']['family']
        self.font_size = self.config['gui']['fonts']['size_large']
        
        # Flicker state tracking
        self.top_box_state = {"is_on": True}
        self.bottom_box_state = {"is_on": True}
        
        # Current mode
        self.current_mode = "IDLE"
    
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
                'frequency': {'flicker_freq_1': 8.0, 'flicker_freq_2': 13.0},
                'gui': {
                    'stimulus': {'box_width': 200, 'box_height': 100, 'box_margin': 50},
                    'colors': {
                        'background': 'black', 'stimulus_on': 'white',
                        'stimulus_off': 'black', 'stimulus_inactive': 'gray'
                    },
                    'fonts': {'family': 'Arial', 'size_large': 20}
                }
            }
    
    def create_window(self):
        """Create the stimulus window"""
        if self.is_open:
            return
        
        # Create new window
        self.window = tk.Tk()
        self.window.title("SSVEP Stimulus - Dual Frequency")
        
        # Get screen dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        # Configure window for fullscreen
        self.window.geometry(f"{screen_width}x{screen_height}")
        self.window.attributes("-topmost", True)
        self.window.configure(bg=self.bg_color)
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.window,
            width=screen_width,
            height=screen_height,
            bg=self.bg_color,
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create stimulus boxes
        self._create_stimulus_boxes(screen_width, screen_height)
        
        # Create labels
        self._create_labels(screen_width, screen_height)
        
        # Set up window closing behavior
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        # Start flicker timing
        self._start_flicker()
        
        self.is_open = True
        print("Stimulus window created")
    
    def _create_stimulus_boxes(self, screen_width, screen_height):
        """Create the stimulus boxes"""
        # Calculate positions
        center_x = screen_width // 2
        
        # Top box (for freq1 - scroll up)
        top_y = self.box_margin
        self.top_box = self.canvas.create_rectangle(
            center_x - self.box_width // 2, top_y,
            center_x + self.box_width // 2, top_y + self.box_height,
            fill=self.on_color, outline=self.on_color, tags="top_box"
        )
        
        # Bottom box (for freq2 - scroll down)
        bottom_y = screen_height - self.box_margin - self.box_height
        self.bottom_box = self.canvas.create_rectangle(
            center_x - self.box_width // 2, bottom_y,
            center_x + self.box_width // 2, bottom_y + self.box_height,
            fill=self.on_color, outline=self.on_color, tags="bottom_box"
        )
    
    def _create_labels(self, screen_width, screen_height):
        """Create text labels for the stimulus boxes"""
        center_x = screen_width // 2
        
        # Create label backgrounds for better visibility
        label_bg_top = self.canvas.create_rectangle(
            center_x - 150, 200,
            center_x + 150, 240,
            fill=self.bg_color, outline=self.on_color
        )
        
        label_bg_bottom = self.canvas.create_rectangle(
            center_x - 150, screen_height - 240,
            center_x + 150, screen_height - 200,
            fill=self.bg_color, outline=self.on_color
        )
        
        # Create text labels
        self.canvas.create_text(
            center_x, 220,
            text=f"{self.freq1} Hz - Scroll Up",
            fill=self.on_color,
            font=(self.font_family, self.font_size)
        )
        
        self.canvas.create_text(
            center_x, screen_height - 220,
            text=f"{self.freq2} Hz - Scroll Down",
            fill=self.on_color,
            font=(self.font_family, self.font_size)
        )
    
    def _start_flicker(self):
        """Start the flicker timing for both boxes"""
        # Calculate periods
        period_ms_top = int(1000 / (2 * self.freq1))  # Half period for on/off cycle
        period_ms_bottom = int(1000 / (2 * self.freq2))  # Half period for on/off cycle
        
        # Start flicker cycles
        self.window.after(100, lambda: self._flicker_top_box(period_ms_top))
        self.window.after(100, lambda: self._flicker_bottom_box(period_ms_bottom))
    
    def _flicker_top_box(self, period_ms):
        """Handle flickering of the top box"""
        if not self.is_open or self.window is None:
            return
        
        try:
            # Toggle state
            self.top_box_state["is_on"] = not self.top_box_state["is_on"]
            
            # Determine color based on mode and state
            if self.current_mode in ["DETECTION", "CALIB_FREQ1"]:
                color = self.on_color if self.top_box_state["is_on"] else self.off_color
            else:
                color = self.inactive_color
            
            # Update box color
            self.canvas.itemconfig("top_box", fill=color, outline=color)
            
            # Schedule next update
            self.window.after(period_ms, lambda: self._flicker_top_box(period_ms))
            
        except Exception as e:
            print(f"Error in top box flicker: {e}")
    
    def _flicker_bottom_box(self, period_ms):
        """Handle flickering of the bottom box"""
        if not self.is_open or self.window is None:
            return
        
        try:
            # Toggle state
            self.bottom_box_state["is_on"] = not self.bottom_box_state["is_on"]
            
            # Determine color based on mode and state
            if self.current_mode in ["DETECTION", "CALIB_FREQ2"]:
                color = self.on_color if self.bottom_box_state["is_on"] else self.off_color
            else:
                color = self.inactive_color
            
            # Update box color
            self.canvas.itemconfig("bottom_box", fill=color, outline=color)
            
            # Schedule next update
            self.window.after(period_ms, lambda: self._flicker_bottom_box(period_ms))
            
        except Exception as e:
            print(f"Error in bottom box flicker: {e}")
    
    def set_mode(self, mode):
        """Set the stimulus mode
        
        Args:
            mode: Current operation mode ("IDLE", "CALIB_FREQ1", "CALIB_FREQ2", "DETECTION")
        """
        self.current_mode = mode
        print(f"Stimulus window mode set to: {mode}")
        
        if not self.is_open:
            return
        
        # Update box visibility based on mode
        try:
            if mode == "CALIB_FREQ1":
                # Only top box active
                self.canvas.itemconfig("bottom_box", fill=self.inactive_color, outline=self.inactive_color)
            elif mode == "CALIB_FREQ2":
                # Only bottom box active
                self.canvas.itemconfig("top_box", fill=self.inactive_color, outline=self.inactive_color)
            elif mode == "DETECTION":
                # Both boxes active (will be handled by flicker functions)
                pass
            else:  # IDLE
                # Both boxes inactive
                self.canvas.itemconfig("top_box", fill=self.inactive_color, outline=self.inactive_color)
                self.canvas.itemconfig("bottom_box", fill=self.inactive_color, outline=self.inactive_color)
        except Exception as e:
            print(f"Error setting stimulus mode: {e}")
    
    def close_window(self):
        """Close the stimulus window"""
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception as e:
                print(f"Error closing stimulus window: {e}")
            finally:
                self.window = None
                self.canvas = None
                self.is_open = False
                print("Stimulus window closed")
    
    def toggle_window(self):
        """Toggle the stimulus window open/closed"""
        if self.is_open:
            self.close_window()
        else:
            self.create_window()
    
    def update_frequencies(self, freq1, freq2):
        """Update stimulus frequencies
        
        Args:
            freq1: New first frequency
            freq2: New second frequency
        """
        self.freq1 = freq1
        self.freq2 = freq2
        
        if self.is_open:
            # Restart flicker with new frequencies
            self._start_flicker()
            # Update labels if they exist
            # Note: This would require recreating the labels or storing references
        
        print(f"Stimulus frequencies updated to {freq1} Hz and {freq2} Hz")
    
    def get_window_handle(self):
        """Get the Tkinter window handle
        
        Returns:
            tkinter.Tk: Window handle or None if not open
        """
        return self.window
    
    def is_window_open(self):
        """Check if the stimulus window is open
        
        Returns:
            bool: True if window is open
        """
        return self.is_open
    
    def bring_to_front(self):
        """Bring the stimulus window to front"""
        if self.window is not None:
            try:
                self.window.lift()
                self.window.focus_force()
            except Exception as e:
                print(f"Error bringing window to front: {e}")
    
    def hide_window(self):
        """Hide the window without destroying it"""
        if self.window is not None:
            try:
                self.window.withdraw()
            except Exception as e:
                print(f"Error hiding window: {e}")
    
    def show_window(self):
        """Show the hidden window"""
        if self.window is not None:
            try:
                self.window.deiconify()
                self.bring_to_front()
            except Exception as e:
                print(f"Error showing window: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.create_window()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_window() 