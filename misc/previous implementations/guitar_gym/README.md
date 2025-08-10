# Guitar Gym - Real-time Guitar Tuner

A real-time guitar tuner and spectrum analyzer with precise note detection and visual tuning feedback.

## Features

- Real-time audio spectrum analysis optimized for guitar frequencies
- Accurate note detection with harmonic validation
- Visual tuning feedback with cents deviation indicator
- Color-coded reference lines for all six guitar strings
- Intelligent decay detection to avoid "pseudo notes" during note decay
- Responsive design for playing melodies

## Technical Details

- Uses Fast Fourier Transform (FFT) for spectral analysis
- Parabolic interpolation for sub-bin frequency accuracy
- Harmonic validation to confirm detected notes
- Decay detection with note locking during sustained notes
- PyQtGraph for GPU-accelerated visualization

## Project Structure

The project has been modularized for better maintainability:

- `guitar_tuner/` - Main package
  - `__init__.py` - Package initialization
  - `audio.py` - Audio processing and FFT computation
  - `notes.py` - Note detection and frequency analysis
  - `ui.py` - User interface and visualization
  - `config.py` - Centralized configuration
- `tuner.py` - Main application entry point

## Usage

Run the application with:

```bash
python tuner.py
```

## Requirements

- Python 3.6+
- sounddevice
- numpy
- scipy
- pyqtgraph

## Installation

```bash
pip install numpy scipy pyqtgraph sounddevice
```

## License

This project is open source.
