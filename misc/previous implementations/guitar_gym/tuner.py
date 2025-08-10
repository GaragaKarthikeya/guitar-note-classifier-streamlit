#!/usr/bin/env python3
"""
Optimized Live Guitar Spectrum Analyzer + Basic Tuner
- Uses sounddevice for audio capture
- PyQtGraph for GPU-accelerated plotting
- Audio callback is light; GUI polls spectrum via QTimer
- Parabolic interpolation for sub-bin peak frequency
- Harmonic validation and persistence to avoid false detections
- Expose tuning/note + cents

Modular implementation with separate components for:
- Audio processing
- Note detection
- UI rendering
- Configuration
"""

import sys
from pyqtgraph.Qt import QtWidgets

from guitar_tuner.config import TunerConfig
from guitar_tuner.audio import AudioProcessor
from guitar_tuner.notes import NoteDetector
from guitar_tuner.ui import SpectrumTunerUI

def main():
    # Initialize application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and configure components
    config = TunerConfig()
    audio_processor = AudioProcessor(config)
    note_detector = NoteDetector(config)
    
    # Create UI
    ui = SpectrumTunerUI(config, audio_processor, note_detector)
    ui.show()
    
    # Start audio processing
    try:
        stream = audio_processor.start_stream()
    except Exception as e:
        print(f"Failed to start audio stream: {e}", file=sys.stderr)
        return
    
    # Start Qt event loop
    try:
        if hasattr(app, 'exec'):
            app.exec()
        else:
            app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            audio_processor.stop_stream()
        except Exception:
            pass

if __name__ == "__main__":
    main()
