"""
Audio processing module for the guitar tuner.
Handles audio input, FFT processing, and spectrum analysis.
"""

import time
import threading
import numpy as np
import sounddevice as sd

class AudioProcessor:
    def __init__(self, config):
        """Initialize with maximum processing power configuration."""
        self.config = config
        # Maximum FFT size for ultimate frequency resolution
        self.nfft = config.CHUNK * 8  # 8x zero-padding for maximum resolution
        
        # Prepare static arrays
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / config.RATE)
        self.min_index = np.searchsorted(self.freqs, config.FREQ_MIN)
        self.max_index = np.searchsorted(self.freqs, config.FREQ_MAX, side='right') - 1
        self.display_freqs = self.freqs[self.min_index:self.max_index+1]
        self.display_bins = len(self.display_freqs)
        
        # Thread-shared buffer
        self.spec_lock = threading.Lock()
        self.shared_spectrum = np.zeros(self.display_bins, dtype=np.float32)
        self.shared_timestamp = 0.0
        
        # Simple but effective window
        self.window = np.hanning(self.config.CHUNK).astype(np.float32)
    
    def start_stream(self):
        """Start the audio input stream."""
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.config.RATE,
            blocksize=self.config.CHUNK,
            dtype='float32'
        )
        self.stream.start()
        return self.stream
    
    def stop_stream(self):
        """Stop the audio input stream."""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def audio_callback(self, indata, frames, time_info, status):
        """Maximum processing power audio callback - simple but intensive."""
        if status:
            print(f"Audio status: {status}")
            
        try:
            samples = indata[:, 0].astype(np.float32)
        except Exception:
            return
            
        if len(samples) < self.config.CHUNK:
            samples = np.pad(samples, (0, self.config.CHUNK - len(samples)))
        elif len(samples) > self.config.CHUNK:
            samples = samples[:self.config.CHUNK]
            
        # Apply window
        windowed_samples = samples * self.window
        
        # Maximum zero-padding for ultimate frequency resolution
        padded_samples = np.zeros(self.nfft, dtype=np.float32)
        padded_samples[:self.config.CHUNK] = windowed_samples
        
        # Compute ultra-high-resolution spectrum
        sp = np.abs(np.fft.rfft(padded_samples, n=self.nfft))
        
        # Minimal smoothing to preserve peaks
        if len(sp) >= 3:
            sp = np.convolve(sp, np.ones(3)/3.0, mode='same')
        
        # Select display range
        disp = sp[self.min_index:self.max_index+1]
        
        with self.spec_lock:
            self.shared_spectrum = disp.astype(np.float32)
            self.shared_timestamp = time.time()
    
    def get_spectrum(self):
        """Get a copy of the current spectrum and timestamp."""
        with self.spec_lock:
            spec = self.shared_spectrum.copy()
            ts = self.shared_timestamp
        return spec, ts
