# SSVEP-Based Brain-Computer Interface (BCI)

A real-time SSVEP (Steady-State Visual Evoked Potential) brain-computer interface system for controlling computer functions using EEG signals.

## Overview

This project implements a BCI system that:

- Detects SSVEP responses at 8Hz and 13Hz frequencies
- Uses machine learning (SVM) for classification
- Provides real-time scroll control
- Works with BrainBit EEG devices

## Hardware Requirements

- **BrainBit EEG Device** (BrainBit or BrainBit Black)
- Windows/Linux/macOS computer
- USB connection for the EEG device

## Software Requirements

- Python 3.8 or higher
- Lab Streaming Layer (LSL)
- BrainBit SDK (neurosdk)

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd bcih-project
```

2. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

3. **Install Lab Streaming Layer:**

   - Download LSL from: https://github.com/sccn/labstreaminglayer
   - Follow platform-specific installation instructions

4. **Install BrainBit SDK:**
   - Download from BrainBit official website
   - Follow manufacturer's installation guide

## Usage

### Step 1: Start the BrainBit Bridge

```bash
python BrainBitLSL.py
```

This creates an LSL stream from your BrainBit device.

### Step 2: Run the Main Application

```bash
python analysis.py
```

### Step 3: Calibration Process

1. Click "Toggle Stimuli Window" to open the flicker window
2. Click "Calibrate 8Hz" and look at the top flickering box for 30 seconds
3. Click "Calibrate 13Hz" and look at the bottom flickering box for 30 seconds
4. Click "Train Model" to train the classifier

### Step 4: Real-time Detection

1. Click "Start Detection"
2. Look at the top box (8Hz) to scroll up
3. Look at the bottom box (13Hz) to scroll down

## Project Structure

- `analysis.py` - Main application (most complete implementation)
- `BrainBitLSL.py` - Bridge between BrainBit device and LSL
- `final.py` - Alternative simplified implementation
- `SVM.py` - CCA-based implementation
- `scroller.py` - Lightweight scroller implementation
- `flash.py` - Simple pygame-based stimulus
- `power.py` - Basic power spectrum analyzer

## Key Features

- **Real-time EEG processing** with multiple filtering stages
- **Signal quality monitoring** for each electrode
- **Dual-frequency SSVEP detection** (8Hz and 13Hz)
- **Machine learning classification** using SVM
- **Adaptive thresholding** based on signal quality
- **Smooth scrolling control** with cooldown periods
- **Comprehensive visualization** of signals and detection

## Troubleshooting

### Common Issues:

1. **"No EEG stream found"**

   - Ensure BrainBitLSL.py is running
   - Check BrainBit device connection
   - Verify LSL installation

2. **Poor signal quality**

   - Clean electrode contacts
   - Apply conductive gel if needed
   - Check electrode placement (O1, O2, T3, T4)

3. **Inconsistent detection**
   - Collect more calibration data
   - Ensure stable gaze on stimuli
   - Check for environmental interference

### Signal Quality Guidelines:

- **Good**: >70% signal quality (green indicators)
- **Acceptable**: 50-70% (yellow indicators)
- **Poor**: <50% (red indicators) - recalibrate

## Configuration

Key parameters can be modified in the code:

- `FLICKER_FREQ_1 = 8.0` - First stimulus frequency
- `FLICKER_FREQ_2 = 13.0` - Second stimulus frequency
- `WINDOW_LENGTH = 2` - Analysis window length (seconds)
- `scroll_cooldown = 1.0` - Time between scroll actions

## File Descriptions

### Primary Files:

- **analysis.py**: Most comprehensive implementation with full GUI
- **BrainBitLSL.py**: Hardware interface for BrainBit devices

### Alternative Implementations:

- **final.py**: Simplified version with basic features
- **SVM.py**: CCA-based approach with enhanced signal processing
- **scroller.py**: Minimal implementation focused on scrolling

### Utility Files:

- **flash.py**: Simple pygame stimulus for testing
- **power.py**: Basic frequency analysis tool



## Acknowledgments

- BrainBit for EEG hardware support
- Lab Streaming Layer community
- SSVEP research community
