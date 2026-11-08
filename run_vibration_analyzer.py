"""
run_vibration_analyzer.py - Master Entrypoint for Real-Time Mechanical Vibration Analyzer

Usage:
    python run_vibration_analyzer.py          # Launch 60 FPS Native PyQt6 Desktop Instrument
    python run_vibration_analyzer.py --web    # Launch Reactive Streamlit Web Dashboard
"""

import sys
import os

# Ensure package root is in Python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    if "--web" in sys.argv:
        import subprocess
        target_script = os.path.join(os.path.dirname(__file__), "vibration_analyzer", "web_app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", target_script] + [a for a in sys.argv[1:] if a != "--web"])
    else:
        from vibration_analyzer.app import main
        main()
