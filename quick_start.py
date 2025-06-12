#!/usr/bin/env python3
"""
Quick Start Script for SSVEP BCI Project
This script helps users get started with the BCI system.
"""

import sys
import os
import subprocess
import importlib
import time

# Required packages for the project
REQUIRED_PACKAGES = [
    'numpy', 'matplotlib', 'scipy', 'sklearn', 
    'pylsl', 'neurosdk', 'pywt', 'joblib', 'pyautogui'
]

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_packages():
    """Check if required packages are installed"""
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages using pip"""
    if not missing_packages:
        return True
    
    print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        print("   Please run: pip install -r requirements.txt")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'analysis.py', 'BrainBitLSL.py', 'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def start_brainbit_bridge():
    """Start the BrainBit LSL bridge"""
    print("\n🚀 Starting BrainBit Bridge...")
    print("   This will search for your BrainBit device and create an LSL stream")
    print("   Make sure your BrainBit is connected and turned on")
    
    response = input("   Continue? (y/n): ").lower().strip()
    if response != 'y':
        return False
    
    try:
        # Start BrainBit bridge in a separate process
        process = subprocess.Popen([sys.executable, 'BrainBitLSL.py'])
        print("✅ BrainBit bridge started")
        print("   Process ID:", process.pid)
        time.sleep(3)  # Give it time to start
        return True
    except Exception as e:
        print(f"❌ Failed to start BrainBit bridge: {e}")
        return False

def start_main_application():
    """Start the main SSVEP application"""
    print("\n🧠 Starting SSVEP BCI Application...")
    print("   This will open the main interface for calibration and detection")
    
    response = input("   Continue? (y/n): ").lower().strip()
    if response != 'y':
        return False
    
    try:
        subprocess.run([sys.executable, 'analysis.py'])
        return True
    except Exception as e:
        print(f"❌ Failed to start main application: {e}")
        return False

def print_usage_instructions():
    """Print basic usage instructions"""
    instructions = """
📋 USAGE INSTRUCTIONS:

1. HARDWARE SETUP:
   - Connect your BrainBit EEG device
   - Place electrodes: O1, O2 (occipital), T3, T4 (temporal)
   - Ensure good electrode contact (low impedance)

2. SOFTWARE WORKFLOW:
   - BrainBit Bridge: Creates LSL stream from device
   - Main Application: Provides GUI for calibration and detection

3. CALIBRATION PROCESS:
   - Click "Toggle Stimuli Window" to open flicker display
   - Click "Calibrate 8Hz" → Look at TOP box for 30 seconds
   - Click "Calibrate 13Hz" → Look at BOTTOM box for 30 seconds
   - Click "Train Model" to create classifier

4. DETECTION MODE:
   - Click "Start Detection"
   - Look at TOP box (8Hz) → Scroll UP
   - Look at BOTTOM box (13Hz) → Scroll DOWN

5. TROUBLESHOOTING:
   - Check signal quality indicators in GUI
   - Ensure stable fixation on stimuli
   - Recalibrate if detection is poor

6. ALTERNATIVE FILES:
   - final.py: Simplified version
   - SVM.py: CCA-based approach
   - flash.py: Simple stimulus for testing
"""
    print(instructions)

def main():
    """Main function to run the quick start process"""
    print("=" * 60)
    print("🧠 SSVEP BCI PROJECT - QUICK START")
    print("=" * 60)
    
    # Check system requirements
    print("\n🔍 Checking System Requirements...")
    if not check_python_version():
        sys.exit(1)
    
    # Check files
    print("\n📁 Checking Required Files...")
    if not check_files():
        print("❌ Missing required files. Please ensure all files are present.")
        sys.exit(1)
    
    # Check packages
    print("\n📦 Checking Python Packages...")
    missing_packages = check_packages()
    
    if missing_packages:
        install_response = input(f"\n❓ Install missing packages? (y/n): ").lower().strip()
        if install_response == 'y':
            if not install_missing_packages(missing_packages):
                sys.exit(1)
        else:
            print("❌ Cannot proceed without required packages")
            sys.exit(1)
    
    print("\n✅ All requirements satisfied!")
    
    # Show usage instructions
    show_instructions = input("\n❓ Show usage instructions? (y/n): ").lower().strip()
    if show_instructions == 'y':
        print_usage_instructions()
    
    # Start the application
    print("\n🚀 STARTING APPLICATION...")
    
    # Option to start components
    print("\nChoose how to proceed:")
    print("1. Start BrainBit Bridge only")
    print("2. Start Main Application only (if bridge is already running)")
    print("3. Start both (recommended for first time)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        start_brainbit_bridge()
    elif choice == '2':
        start_main_application()
    elif choice == '3':
        if start_brainbit_bridge():
            input("\nPress Enter when BrainBit bridge is ready...")
            start_main_application()
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
    
    print("\n✅ Quick start completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 