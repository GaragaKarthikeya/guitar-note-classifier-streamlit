#!/usr/bin/env python3
"""
Optimized Live Guitar Spectrum Analyzer + Basic Tuner
- Uses sounddevice for audio capture
- PyQtGraph for GPU-accelerated plotting
- Audio callback is light; GUI polls spectrum via QTimer
- Parabolic interpolation for sub-bin peak frequency
- Harmonic validation and persistence to avoid false detections
- Expose tuning/note + cents

Author: ChatGPT (structured & explicit)
"""

import sys
import time
import threading
from collections import deque
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from scipy.signal import find_peaks

# -------------------------
# User-tweakable settings
# -------------------------
CHUNK = 2048          # samples per block (2048 -> ~21 FPS at 44.1kHz)
RATE = 44100          # sample rate
FREQ_MIN = 75         # Hz, lower display limit - slightly below low E (82.4 Hz)
FREQ_MAX = 700        # Hz, upper display limit - reduced from 3000 to focus on guitar range
SMOOTH_ALPHA = 0.6    # exponential smoothing factor for the displayed spectrum (0..1) - increased from 0.35 for more responsiveness
MIN_PEAK_HEIGHT = 2   # minimum amplitude for find_peaks (tweak by mic/gain) - reduced from 4
MIN_PEAK_PROMINENCE = 1 # reduced from 2
PEAK_DISTANCE_BINS = 2  # min distance between peaks (bins) - reduced from 3
HARMONIC_COUNT = 2    # check 2nd..(HARMONIC_COUNT+1)th harmonic - reduced from 3
HARMONIC_RATIO_THRESHOLD = 0.08  # harmonic amplitude must be >= this fraction of fundamental - reduced from 0.12
FREQ_TOLERANCE = 0.06  # 6% frequency tolerance for matching expected harmonic/fundamental - increased from 4%
PERSISTENCE_FRAMES = 1  # # of frames a new note must persist before replacing previous - reduced from 2
STRONGER_FACTOR = 1.2   # if new candidate amplitude > previous * STRONGER_FACTOR, accept immediately - reduced from 1.4
# Decay detection settings
DECAY_THRESHOLD = 0.6   # If amplitude falls below 60% of peak, consider decaying (less aggressive)
RISE_THRESHOLD = 1.1    # New note must be only 10% louder to break lock (more responsive)
MIN_AMP_THRESHOLD = 2.5 # Lower threshold to detect quieter notes
DECAY_HOLD_TIME = 1.0   # Hold decaying note for max 1 second (down from 3s)

# Standard guitar tuning (Hz) and harmonics for reference
GUITAR_NOTES = {
    'E2 (6)': 82.4069,  # Added string numbers for easier reference
    'A2 (5)': 110.0000,
    'D3 (4)': 146.8324,
    'G3 (3)': 195.9977,
    'B3 (2)': 246.9417,
    'E4 (1)': 329.6276
}
# We'll build a list of note frequencies across many octaves for tuning mapping
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# -------------------------
# Derived static arrays
# -------------------------
# Frequency bins for rfft with CHUNK zero-padding? We use n=CHUNK for speed (you can use n=CHUNK*2)
nfft = CHUNK
freqs = np.fft.rfftfreq(nfft, 1.0 / RATE)  # length nfft//2 + 1
# indices for display range
min_index = np.searchsorted(freqs, FREQ_MIN)
max_index = np.searchsorted(freqs, FREQ_MAX, side='right') - 1
display_freqs = freqs[min_index:max_index+1]
display_bins = len(display_freqs)

# -------------------------
# Thread-shared buffer
# -------------------------
spec_lock = threading.Lock()
shared_spectrum = np.zeros(display_bins, dtype=np.float32)
shared_timestamp = 0.0

# -------------------------
# Helper functions
# -------------------------
def parabolic_interpolation(mag, idx):
    """
    Parabolic interpolation for a peak at bin idx in magnitude array mag.
    Returns refined bin index offset (in bins) and estimated amplitude.
    """
    if idx <= 0 or idx >= len(mag)-1:
        return idx, mag[idx]
    alpha = mag[idx-1]
    beta = mag[idx]
    gamma = mag[idx+1]
    denom = (alpha - 2*beta + gamma)
    if denom == 0:
        return idx, beta
    delta = 0.5 * (alpha - gamma) / denom
    peak_bin = idx + delta
    # Parabolic peak amplitude estimate:
    peak_amp = beta - 0.25*(alpha - gamma)*delta
    return peak_bin, peak_amp

def freq_to_note_cents(f):
    """
    Convert frequency f (Hz) to nearest note and cents offset.
    Return (note_name, octave, cents)
    """
    if f <= 0:
        return None, None, None
    # A4 = 440 Hz is MIDI 69
    midi = 69 + 12 * np.log2(f / 440.0)
    midi_round = int(np.round(midi))
    cents = (midi - midi_round) * 100
    note_index = (midi_round % 12)
    octave = (midi_round // 12) - 1
    name = NOTE_NAMES[note_index]
    return f"{name}{octave}", midi_round, cents

def harmonic_validation(fund_freq, spectrum, bins, freq_bins):
    """
    Check presence of harmonics for the given fundamental frequency.
    fund_freq: Hz
    spectrum: displayed magnitude array
    bins: length of displayed spectrum
    freq_bins: array of bin center frequencies
    Returns (score, details) where score is fraction of harmonics found (0..1).
    """
    found = 0
    # amplitude at fundamental: use nearest bin amplitude
    idx = np.argmin(np.abs(freq_bins - fund_freq))
    fund_amp = spectrum[idx] if idx < bins else 0.0
    if fund_amp <= 0:
        return 0.0, {}
    details = {}
    for h in range(2, 2 + HARMONIC_COUNT):
        target = fund_freq * h
        if target > freq_bins[-1]:
            details[h] = False
            continue
        # find nearest bin
        idx_h = np.argmin(np.abs(freq_bins - target))
        amp_h = spectrum[idx_h]
        ratio = amp_h / (fund_amp + 1e-12)
        details[h] = (ratio >= HARMONIC_RATIO_THRESHOLD)
        if details[h]:
            found += 1
    score = found / float(HARMONIC_COUNT)
    return score, details

# -------------------------
# Audio callback (light)
# -------------------------
window = np.hanning(nfft).astype(np.float32)

def audio_callback(indata, frames, time_info, status):
    global shared_spectrum, shared_timestamp
    if status:
        # Print non-fatal warnings to stderr
        print("Audio status:", status, file=sys.stderr)
    try:
        samples = indata[:, 0].astype(np.float32)
    except Exception:
        return  # no data
    if len(samples) < nfft:
        # zero-pad (shouldn't normally happen with blocksize == CHUNK)
        samples = np.pad(samples, (0, nfft - len(samples)))
    # Apply window
    samples *= window
    # Compute magnitude spectrum (no extra zero-padding for speed)
    sp = np.abs(np.fft.rfft(samples, n=nfft))
    # Simple small smoothing across bins to reduce bin noise (convolution)
    sp = np.convolve(sp, np.ones(3)/3.0, mode='same')
    # Select display range
    disp = sp[min_index:max_index+1]
    with spec_lock:
        # copy into shared buffer
        shared_spectrum = disp.astype(np.float32)
        shared_timestamp = time.time()

# -------------------------
# GUI / Main thread
# -------------------------
class SpectrumTuner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Guitar Spectrum + Tuner")
        self.resize(1100, 640)
        # pg config - try to use OpenGL for faster plotting, but fallback to standard rendering if not available
        try:
            # First try importing the OpenGL module to check if it's available
            import OpenGL.GL
            pg.setConfigOptions(useOpenGL=True, enableExperimental=True)
        except ImportError:
            # If OpenGL is not available, use standard rendering
            print("PyOpenGL not available, using standard rendering...")
            pg.setConfigOptions(useOpenGL=False)
            
        self.win = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.win)
        
        # Create a more visually appealing background
        self.win.setBackground('k')  # Black background - applied to the GraphicsLayoutWidget, not the plot
        
        self.plot = self.win.addPlot(row=0, col=0, title="Guitar Tuner - Frequency Spectrum")
        self.plot.setLabel('left', "Amplitude")
        self.plot.setLabel('bottom', "Frequency", units='Hz')
        self.plot.showGrid(x=True, y=True, alpha=0.4)  # Slightly more visible grid
        self.plot.setXRange(FREQ_MIN, FREQ_MAX, padding=0.03)
        self.plot.setYRange(0, 100)  # Reduced Y range for better sensitivity to quieter signals
        
        # Define colors for each guitar string for better visual distinction
        string_colors = {
            'E2 (6)': (255, 0, 0),      # Red for low E
            'A2 (5)': (255, 165, 0),    # Orange for A
            'D3 (4)': (255, 255, 0),    # Yellow for D
            'G3 (3)': (0, 255, 0),      # Green for G
            'B3 (2)': (0, 165, 255),    # Blue for B
            'E4 (1)': (128, 0, 255)     # Purple for high E
        }

        # mark guitar fundamental lines with improved visibility
        for note, freq in GUITAR_NOTES.items():
            color = string_colors[note]
            line = pg.InfiniteLine(pos=freq, angle=90, pen=pg.mkPen(color, width=2))
            lbl = pg.TextItem(text=note, color=color, anchor=(0.5, 1.0))
            lbl.setPos(freq, 0)
            self.plot.addItem(line)
            self.plot.addItem(lbl)

        # spectrum curve
        self.curve = self.plot.plot(display_freqs, np.zeros_like(display_freqs), pen=pg.mkPen('y', width=2))

        # Enhanced UI for detected note
        self.note_text = pg.TextItem(text="", color='w', anchor=(0,0), html=None)
        self.note_text.setPos(FREQ_MIN + 10, 90)  # Position adjusted for new Y range
        self.plot.addItem(self.note_text)
        
        # Add a tuning indicator that will show how close the note is to being in tune
        self.tuning_indicator = pg.PlotDataItem([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='w')
        self.plot.addItem(self.tuning_indicator)
        
        # Add a reference line for the currently detected note
        self.current_note_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.current_note_line.hide()  # Hide initially until a note is detected
        self.plot.addItem(self.current_note_line)

        # internal state for smoothing & persistence
        self.smoothed = np.zeros_like(display_freqs, dtype=np.float32)
        self.prev_detect = None
        self.prev_amp = 0.0
        self.candidate = None
        self.candidate_count = 0
        
        # Add these to track note decay
        self.peak_amp = 0.0          # Track the peak amplitude of current note
        self.min_amp_since_peak = 0.0 # Track minimum amplitude since peak
        self.in_decay_mode = False   # Whether we're in a note decay phase
        self.locked_note = None      # The note we've locked onto during decay
        self.locked_freq = None      # Frequency of the locked note
        self.decay_start_time = 0    # When the decay started

        # Timer to poll shared spectrum and update UI
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1000 * (CHUNK / float(RATE)) * 0.9))  # slightly faster than block interval
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def find_peaks_and_identify(self, spectrum):
        # find peaks
        peaks, props = find_peaks(spectrum,
                                 height=MIN_PEAK_HEIGHT,
                                 prominence=MIN_PEAK_PROMINENCE,
                                 distance=PEAK_DISTANCE_BINS)
        if len(peaks) == 0:
            return None
        # refine peaks by parabolic interpolation to get sub-bin accuracy
        peak_candidates = []
        for p in peaks:
            peak_bin_global = p  # relative to displayed array
            abs_bin = min_index + peak_bin_global  # global bin index in freqs array
            # parabolic interpolation on original rfft mag -> but spectrum passed is already sliced & smoothed
            refined_bin, amp = parabolic_interpolation(spectrum, p)
            # convert refined bin to frequency
            refined_freq = np.interp(refined_bin, np.arange(len(display_freqs)), display_freqs)
            peak_candidates.append((refined_freq, amp, p))
        # choose the strongest by amplitude
        peak_candidates.sort(key=lambda x: x[1], reverse=True)
        # return top candidate
        f, a, raw_idx = peak_candidates[0]
        return f, a

    def update_frame(self):
        global shared_spectrum, shared_timestamp
        with spec_lock:
            spec = shared_spectrum.copy()
            ts = shared_timestamp
        # if no recent audio, skip
        if ts == 0:
            return
        # apply exponential smoothing
        self.smoothed = SMOOTH_ALPHA * spec + (1.0 - SMOOTH_ALPHA) * self.smoothed

        # update plot curve (only when significant change to avoid churn)
        self.curve.setData(display_freqs, self.smoothed)

        # peak detection & identification
        candidate = self.find_peaks_and_identify(self.smoothed)
        if candidate is None:
            # No peak found
            self.candidate = None
            self.candidate_count = 0
            
            # Reset decay tracking if amplitude is very low
            if self.prev_amp < MIN_AMP_THRESHOLD:
                self.in_decay_mode = False
                self.locked_note = None
                self.locked_freq = None
            
            # Update UI - but keep the locked note if we're in decay mode
            if not self.in_decay_mode:
                self.note_text.setText("")
                self.current_note_line.hide()
                self.tuning_indicator.setData([], [])
            else:
                # Keep showing the locked note with fading opacity, but with shorter hold time
                elapsed_decay = time.time() - self.decay_start_time
                if elapsed_decay > DECAY_HOLD_TIME:  # Release the lock after the configured hold time
                    self.in_decay_mode = False
                    self.locked_note = None
                    self.locked_freq = None
                    self.note_text.setText("")
                    self.current_note_line.hide()
                    self.tuning_indicator.setData([], [])
            return

        freq_cand, amp_cand = candidate
        
        # Decay mode logic - more responsive to melodic playing
        if self.in_decay_mode:
            # Track minimum amplitude since peak
            self.min_amp_since_peak = min(self.min_amp_since_peak, amp_cand)
            
            # Check for new note detection - more sensitive during melody playing
            new_note_detected = False
            
            # Case 1: Note got louder again (new attack)
            if amp_cand > self.min_amp_since_peak * RISE_THRESHOLD:
                new_note_detected = True
                
            # Case 2: Different note with reasonable amplitude
            if self.locked_freq and abs(freq_cand - self.locked_freq)/self.locked_freq > FREQ_TOLERANCE:
                # If frequency changed more than tolerance and has decent amplitude
                if amp_cand > MIN_AMP_THRESHOLD * 1.5:
                    new_note_detected = True
                    
            # Case 3: Sharp increase in amplitude even if same frequency
            if amp_cand > self.prev_amp * 1.5:  # 50% increase from previous frame
                new_note_detected = True
                
            # Exit decay mode if new note detected
            if new_note_detected:
                self.in_decay_mode = False
                self.locked_note = None
                self.locked_freq = None
                # Reset peak tracking for new note
                self.peak_amp = amp_cand
            else:
                # Still in decay - use the locked note for very different frequencies
                if self.locked_freq and abs(freq_cand - self.locked_freq)/self.locked_freq > FREQ_TOLERANCE * 2:
                    # Very different frequency - stick with locked note but update amplitude
                    candidate = (self.locked_freq, amp_cand)
                    freq_cand, amp_cand = candidate
                # For similar frequencies, allow small drift as the note decays
                self.prev_amp = amp_cand

        # Track amplitude peaks and decays - more melody-friendly
        if not self.in_decay_mode and self.prev_detect is not None:
            # Entering decay mode?
            if self.prev_amp > self.peak_amp:
                self.peak_amp = self.prev_amp  # Update peak if louder
            
            # Reset peak tracking if note has changed significantly
            if self.prev_detect and abs(freq_cand - self.prev_detect)/self.prev_detect > FREQ_TOLERANCE * 1.5:
                # Different note detected - reset peak amplitude tracking
                self.peak_amp = amp_cand
            
            # Check for start of decay - only engage for notes that reached good amplitude
            if amp_cand < self.peak_amp * DECAY_THRESHOLD and self.peak_amp > MIN_AMP_THRESHOLD * 2:
                # Only engage decay mode for notes that were reasonably strong
                self.in_decay_mode = True
                self.locked_freq = self.prev_detect
                self.locked_note = freq_to_note_cents(self.locked_freq)[0]
                self.decay_start_time = time.time()
                self.min_amp_since_peak = amp_cand
        # harmonic validation score (0..1)
        score, details = harmonic_validation(freq_cand, self.smoothed, display_bins, display_freqs)

        # Make decision: much less strict requirements for harmonic confirmation or amplitude
        accept_candidate = False
        if score >= 0.15:  # even lower threshold for harmonic presence - reduced from 0.25
            accept_candidate = True
        elif amp_cand > self.prev_amp * STRONGER_FACTOR:
            accept_candidate = True
        elif amp_cand > 10 and score >= 0.05:  # drastically reduced thresholds from 20 and 0.10
            # even quieter signals with minimal harmonic evidence
            accept_candidate = True
        elif amp_cand > 5:  # accept very quiet signals with no harmonic validation
            accept_candidate = True

        # persistence logic
        if not accept_candidate:
            # do not promote candidate; keep previous detection if any
            # We still allow loud single fundamentals to pass as above
            final_freq = None
            final_amp = 0.0
        else:
            # candidate accepted for consideration
            if self.prev_detect is None:
                # no previous note -> adopt after N frames
                if self.candidate is None or abs(self.candidate - freq_cand) > (freq_cand * FREQ_TOLERANCE):
                    # reset candidate if different freq
                    self.candidate = freq_cand
                    self.candidate_count = 1
                else:
                    self.candidate_count += 1
                if self.candidate_count >= PERSISTENCE_FRAMES:
                    final_freq = freq_cand
                    final_amp = amp_cand
                else:
                    final_freq = None
                    final_amp = 0.0
            else:
                # there is a previous detection - much more relaxed rules
                if abs(freq_cand - self.prev_detect) / self.prev_detect < FREQ_TOLERANCE * 1.5:  # 50% more tolerant
                    # same note family => update immediately
                    final_freq = freq_cand
                    final_amp = amp_cand
                    self.candidate = freq_cand
                    self.candidate_count = 0
                else:
                    # new distinct candidate - accept almost immediately with reduced requirements
                    if amp_cand > self.prev_amp * STRONGER_FACTOR * 0.8:  # 20% more sensitive than STRONGER_FACTOR
                        # less strict amplitude threshold -> immediate replace
                        final_freq = freq_cand
                        final_amp = amp_cand
                        self.candidate = None
                        self.candidate_count = 0
                    else:
                        # minimal persistence requirement
                        if self.candidate is None or abs(self.candidate - freq_cand) > (freq_cand * FREQ_TOLERANCE * 1.5):
                            self.candidate = freq_cand
                            # With PERSISTENCE_FRAMES = 1, this should always be enough
                            self.candidate_count = 1
                        else:
                            self.candidate_count += 1
                        if self.candidate_count >= PERSISTENCE_FRAMES:
                            final_freq = freq_cand
                            final_amp = amp_cand
                        else:
                            # With reduced persistence requirement, this shouldn't happen often
                            final_freq = None
                            final_amp = 0.0

        # finalize - enhanced visual feedback for tuning
        if 'final_freq' in locals() and final_freq:
            # update previous detection
            self.prev_detect = final_freq
            self.prev_amp = final_amp
            note_name, midi, cents = freq_to_note_cents(final_freq)
            
            # Find closest guitar note for tuning reference
            closest_guitar_note = None
            min_distance = float('inf')
            for note, freq in GUITAR_NOTES.items():
                distance = abs(freq - final_freq) / freq
                if distance < min_distance:
                    min_distance = distance
                    closest_guitar_note = (note, freq)
            
            # Color-coded text based on how in-tune the note is
            if abs(cents) < 5:
                color = "lime"  # Very in tune (green)
                tune_status = "✓ IN TUNE"
            elif abs(cents) < 15:
                color = "yellow"  # Close to in tune (yellow)
                tune_status = "ALMOST"
            else:
                color = "red"  # Out of tune (red)
                tune_status = "TUNE" + ("↑" if cents > 0 else "↓")
            
            # Update the current note line and tuning indicator
            self.current_note_line.setPos(final_freq)
            self.current_note_line.show()
            
            # Format with HTML for better visibility
            text = f"<span style='font-size: 16pt; color: white;'><b>{note_name}</b></span><br>"
            text += f"<span style='font-size: 12pt; color: {color};'><b>{tune_status}</b></span><br>"
            text += f"<span style='font-size: 10pt; color: #aaaaaa;'>{final_freq:.1f} Hz ({cents:+.1f}¢)</span>"
            
            # If it's close to a guitar note, show which string
            if closest_guitar_note and min_distance < 0.1:  # Within 10% of a guitar note
                note_name, freq = closest_guitar_note
                text += f"<br><span style='color: #88ff88;'>Guitar string: {note_name}</span>"
            
            self.note_text.setHtml(text)
            
            # Show tuning indicator - horizontal position shows frequency, vertical shows cents offset
            x_pos = [final_freq]
            y_pos = [50 + cents/2]  # Center at 50, scale cents for visibility
            self.tuning_indicator.setData(x_pos, y_pos)
            
        elif candidate is not None:
            # Show even unconfirmed candidate information
            freq_cand, amp_cand = candidate
            note_name, midi, cents = freq_to_note_cents(freq_cand)
            
            text = f"<span style='font-size: 14pt; color: #aaaaaa;'><b>{note_name}</b></span><br>"
            text += f"<span style='font-size: 10pt; color: #888888;'>{freq_cand:.1f} Hz ({cents:+.1f}¢)</span>"
            self.note_text.setHtml(text)
            
            # Update position of the current note line but with less opacity
            self.current_note_line.setPos(freq_cand)
            self.current_note_line.setPen(pg.mkPen('w', width=1, style=QtCore.Qt.PenStyle.DotLine))
            self.current_note_line.show()
            
            # Show indicator for candidate
            x_pos = [freq_cand]
            y_pos = [50 + cents/2]
            self.tuning_indicator.setData(x_pos, y_pos)
            
        else:
            # No detection
            if self.prev_detect:
                note_name, midi, cents = freq_to_note_cents(self.prev_detect)
                text = f"<span style='font-size: 12pt; color: #666666;'>Last: {note_name}</span><br>"
                text += f"<span style='font-size: 10pt; color: #555555;'>{self.prev_detect:.1f} Hz</span>"
                self.note_text.setHtml(text)
                # Fade out the line
                self.current_note_line.setPen(pg.mkPen('w', width=1, style=QtCore.Qt.PenStyle.DotLine))
            else:
                self.note_text.setHtml("<span style='color: #666666;'>Play a note...</span>")
                self.current_note_line.hide()
            
            # Clear the tuning indicator
            self.tuning_indicator.setData([], [])

# -------------------------
# Run app
# -------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SpectrumTuner()
    win.show()

    # Open input stream (blocksize = CHUNK), using the light audio callback
    stream = sd.InputStream(callback=audio_callback,
                            channels=1,
                            samplerate=RATE,
                            blocksize=CHUNK,
                            dtype='float32')
    try:
        stream.start()
    except Exception as e:
        print("Failed to start audio stream:", e, file=sys.stderr)
        return

    # Qt event loop
    try:
        if hasattr(app, 'exec'):
            app.exec()
        else:
            app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
