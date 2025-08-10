"""
Configuration settings for the guitar tuner.
"""

class TunerConfig:
    def __init__(self):
        # Maximum processing power settings - simple but powerful
        self.CHUNK = 16384         # Maximum chunk size for ultimate frequency resolution
        self.RATE = 44100          # sample rate
        self.FREQ_MIN = 60         # Hz, wide range for maximum coverage
        self.FREQ_MAX = 800        # Hz, wide range for maximum coverage
        
        # Simple, aggressive spectrum analysis
        self.SMOOTH_ALPHA = 0.5    # Less smoothing for more responsive detection
        
        # Simple peak detection - just find the strongest peaks
        self.MIN_PEAK_HEIGHT = 0.5   # Very low threshold - let processing power handle the rest
        self.MIN_PEAK_PROMINENCE = 0.3  # Very low threshold
        self.PEAK_DISTANCE_BINS = 2  # Minimal distance constraint
        
        # Simple harmonic validation
        self.HARMONIC_COUNT = 2    # Just check 2nd and 3rd harmonic - simple but effective
        self.HARMONIC_RATIO_THRESHOLD = 0.05  # Very low threshold - let amplitude decide
        
        # Simple note detection - minimal persistence, maximum responsiveness
        self.FREQ_TOLERANCE = 0.02  # Very tight tolerance with high resolution
        self.PERSISTENCE_FRAMES = 1  # No persistence - immediate response
        self.STRONGER_FACTOR = 1.1   # Minimal requirement for override
        
        # Simple decay detection
        self.DECAY_THRESHOLD = 0.5   # Simple threshold
        self.RISE_THRESHOLD = 1.05    # Very sensitive to new notes
        self.MIN_AMP_THRESHOLD = 1.0 # Low threshold - detect everything
        self.DECAY_HOLD_TIME = 0.5   # Short hold time
        
        # Standard guitar tuning (Hz) and reference notes
        self.GUITAR_NOTES = {
            'E2 (6)': 82.4069,
            'A2 (5)': 110.0000,
            'D3 (4)': 146.8324,
            'G3 (3)': 195.9977,
            'B3 (2)': 246.9417,
            'E4 (1)': 329.6276
        }
