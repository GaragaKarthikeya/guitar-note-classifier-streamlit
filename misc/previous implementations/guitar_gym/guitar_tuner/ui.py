"""
GUI module for the guitar tuner.
Handles the graphical interface, spectrum visualization, and tuning indicators.
"""

import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

class SpectrumTunerUI(QtWidgets.QMainWindow):
    def __init__(self, config, audio_processor, note_detector):
        """Initialize the UI with the given configuration and processors."""
        super().__init__()
        self.config = config
        self.audio_processor = audio_processor
        self.note_detector = note_detector
        
        # Get frequency data from audio processor
        self.display_freqs = audio_processor.display_freqs
        self.display_bins = audio_processor.display_bins
        
        # Setup UI components
        self._setup_window()
        self._setup_plot()
        self._setup_guitar_references()
        self._setup_tuning_indicators()
        
        # Internal state for note detection and tracking
        self._initialize_state()
        
        # Timer to poll spectrum and update UI
        self._setup_timer()
    
    def _setup_window(self):
        """Set up the main window."""
        self.setWindowTitle("Live Guitar Spectrum + Tuner")
        self.resize(1100, 640)
        
        # Try to use OpenGL for faster plotting
        try:
            import OpenGL.GL
            pg.setConfigOptions(useOpenGL=True, enableExperimental=True)
        except ImportError:
            print("PyOpenGL not available, using standard rendering...")
            pg.setConfigOptions(useOpenGL=False)
        
        self.win = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.win)
        self.win.setBackground('k')  # Black background
    
    def _setup_plot(self):
        """Set up the main plot area."""
        self.plot = self.win.addPlot(row=0, col=0, title="Guitar Tuner - Frequency Spectrum")
        self.plot.setLabel('left', "Amplitude")
        self.plot.setLabel('bottom', "Frequency", units='Hz')
        self.plot.showGrid(x=True, y=True, alpha=0.4)
        self.plot.setXRange(self.config.FREQ_MIN, self.config.FREQ_MAX, padding=0.03)
        self.plot.setYRange(0, 100)  # Reduced Y range for better sensitivity
        
        # Spectrum curve
        self.curve = self.plot.plot(
            self.display_freqs, 
            np.zeros_like(self.display_freqs), 
            pen=pg.mkPen('y', width=2)
        )
    
    def _setup_guitar_references(self):
        """Set up reference lines for guitar strings."""
        # Define colors for each guitar string
        self.string_colors = {
            'E2 (6)': (255, 0, 0),      # Red for low E
            'A2 (5)': (255, 165, 0),    # Orange for A
            'D3 (4)': (255, 255, 0),    # Yellow for D
            'G3 (3)': (0, 255, 0),      # Green for G
            'B3 (2)': (0, 165, 255),    # Blue for B
            'E4 (1)': (128, 0, 255)     # Purple for high E
        }
        
        # Add reference lines for guitar strings
        for note, freq in self.config.GUITAR_NOTES.items():
            color = self.string_colors[note]
            line = pg.InfiniteLine(pos=freq, angle=90, pen=pg.mkPen(color, width=2))
            lbl = pg.TextItem(text=note, color=color, anchor=(0.5, 1.0))
            lbl.setPos(freq, 0)
            self.plot.addItem(line)
            self.plot.addItem(lbl)
    
    def _setup_tuning_indicators(self):
        """Set up the tuning indicators."""
        # Text display for the detected note
        self.note_text = pg.TextItem(text="", color='w', anchor=(0,0), html=None)
        self.note_text.setPos(self.config.FREQ_MIN + 10, 90)
        self.plot.addItem(self.note_text)
        
        # Tuning indicator that shows cents deviation
        self.tuning_indicator = pg.PlotDataItem(
            [], [], 
            pen=None, 
            symbol='o', 
            symbolSize=10, 
            symbolBrush='w'
        )
        self.plot.addItem(self.tuning_indicator)
        
        # Reference line for the currently detected note
        self.current_note_line = pg.InfiniteLine(
            pos=0, 
            angle=90, 
            pen=pg.mkPen('w', width=2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.current_note_line.hide()  # Hide until a note is detected
        self.plot.addItem(self.current_note_line)
    
    def _initialize_state(self):
        """Initialize the internal state for note tracking."""
        self.smoothed = np.zeros_like(self.display_freqs, dtype=np.float32)
        self.prev_detect = None
        self.prev_amp = 0.0
        self.candidate = None
        self.candidate_count = 0
        
        # Decay detection state
        self.peak_amp = 0.0
        self.min_amp_since_peak = 0.0
        self.in_decay_mode = False
        self.locked_note = None
        self.locked_freq = None
        self.decay_start_time = 0
    
    def _setup_timer(self):
        """Set up the timer for updating the UI."""
        self.timer = QtCore.QTimer()
        interval = int(1000 * (self.config.CHUNK / float(self.config.RATE)) * 0.9)
        self.timer.setInterval(interval)  # slightly faster than block interval
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()
    
    def update_frame(self):
        """Update the display based on the latest audio data."""
        # Get spectrum data from audio processor
        spec, ts = self.audio_processor.get_spectrum()
        
        # If no recent audio, skip
        if ts == 0:
            return
            
        # Apply exponential smoothing
        self.smoothed = self.config.SMOOTH_ALPHA * spec + (1.0 - self.config.SMOOTH_ALPHA) * self.smoothed
        
        # Update plot curve
        self.curve.setData(self.display_freqs, self.smoothed)
        
        # Enhanced peak detection & identification with octave certainty
        candidate = self.note_detector.find_peaks_and_identify(self.smoothed, self.display_freqs)
        
        if candidate is None:
            self._handle_no_peak()
            return
        
        freq_cand, amp_cand, is_octave_certain = candidate
        
        # Handle decay mode logic
        if self.in_decay_mode:
            self._handle_decay_mode(freq_cand, amp_cand)
        
        # Track amplitude peaks and decays
        if not self.in_decay_mode and self.prev_detect is not None:
            self._track_peaks_and_decays(freq_cand, amp_cand)
        
        # Enhanced harmonic validation
        score, details, _ = self.note_detector.harmonic_validation(
            freq_cand, 
            self.smoothed, 
            self.display_bins, 
            self.display_freqs
        )
        
        # Use the octave certainty information from the peak detection
        # is_octave_certain was already determined during peak detection
        
        # Make decision on accepting the candidate
        accept_candidate = self._decide_on_candidate(freq_cand, amp_cand, score)
        
        # Apply persistence logic
        final_freq, final_amp = self._apply_persistence_logic(freq_cand, amp_cand, accept_candidate)
        
        # Update UI based on detection
        if 'final_freq' in locals() and final_freq:
            self._update_ui_with_detection(final_freq, final_amp, is_octave_certain)
        elif candidate is not None:
            self._show_candidate_info(freq_cand, amp_cand, is_octave_certain)
        else:
            self._show_no_detection_info()
    
    def _handle_no_peak(self):
        """Handle the case when no peak is found."""
        # Reset candidate tracking
        self.candidate = None
        self.candidate_count = 0
        
        # Reset decay tracking if amplitude is very low
        if self.prev_amp < self.config.MIN_AMP_THRESHOLD:
            self.in_decay_mode = False
            self.locked_note = None
            self.locked_freq = None
        
        # Update UI based on decay mode
        if not self.in_decay_mode:
            self.note_text.setText("")
            self.current_note_line.hide()
            self.tuning_indicator.setData([], [])
        else:
            # Keep showing the locked note with fading opacity, but with shorter hold time
            elapsed_decay = time.time() - self.decay_start_time
            if elapsed_decay > self.config.DECAY_HOLD_TIME:
                self.in_decay_mode = False
                self.locked_note = None
                self.locked_freq = None
                self.note_text.setText("")
                self.current_note_line.hide()
                self.tuning_indicator.setData([], [])
    
    def _handle_decay_mode(self, freq_cand, amp_cand):
        """Handle decay mode logic."""
        # Track minimum amplitude since peak
        self.min_amp_since_peak = min(self.min_amp_since_peak, amp_cand)
        
        # Check for new note detection - more sensitive during melody playing
        new_note_detected = False
        
        # Case 1: Note got louder again (new attack)
        if amp_cand > self.min_amp_since_peak * self.config.RISE_THRESHOLD:
            new_note_detected = True
        
        # Case 2: Different note with reasonable amplitude
        if self.locked_freq and abs(freq_cand - self.locked_freq)/self.locked_freq > self.config.FREQ_TOLERANCE:
            # If frequency changed more than tolerance and has decent amplitude
            if amp_cand > self.config.MIN_AMP_THRESHOLD * 1.5:
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
            if self.locked_freq and abs(freq_cand - self.locked_freq)/self.locked_freq > self.config.FREQ_TOLERANCE * 2:
                # Very different frequency - stick with locked note but update amplitude
                self.prev_amp = amp_cand
                return self.locked_freq, amp_cand
            # For similar frequencies, allow small drift as the note decays
            self.prev_amp = amp_cand
    
    def _track_peaks_and_decays(self, freq_cand, amp_cand):
        """Track amplitude peaks and detect decay."""
        # Update peak if louder
        if self.prev_amp > self.peak_amp:
            self.peak_amp = self.prev_amp
        
        # Reset peak tracking if note has changed significantly
        if self.prev_detect and abs(freq_cand - self.prev_detect)/self.prev_detect > self.config.FREQ_TOLERANCE * 1.5:
            # Different note detected - reset peak amplitude tracking
            self.peak_amp = amp_cand
        
        # Check for start of decay - only engage for notes that reached good amplitude
        if amp_cand < self.peak_amp * self.config.DECAY_THRESHOLD and self.peak_amp > self.config.MIN_AMP_THRESHOLD * 2:
            # Only engage decay mode for notes that were reasonably strong
            self.in_decay_mode = True
            self.locked_freq = self.prev_detect
            self.locked_note = self.note_detector.freq_to_note_cents(self.locked_freq)[0]
            self.decay_start_time = time.time()
            self.min_amp_since_peak = amp_cand
    
    def _decide_on_candidate(self, freq_cand, amp_cand, score):
        """Decide whether to accept the candidate note with octave jump prevention."""
        # First check if this would be an octave jump
        if self.prev_detect is not None:
            freq_ratio = freq_cand / self.prev_detect
            # Check for octave jumps (ratio around 2.0 or 0.5)
            is_octave_jump = (abs(freq_ratio - 2.0) < 0.1) or (abs(freq_ratio - 0.5) < 0.05)
            
            if is_octave_jump:
                # For octave jumps, require much stronger evidence
                if score < 0.3 or amp_cand < self.prev_amp * 1.5:
                    return False  # Reject likely octave jump
        
        # Normal acceptance criteria
        if score >= 0.15:  # Reduced threshold for harmonic presence
            return True
        elif amp_cand > self.prev_amp * self.config.STRONGER_FACTOR:
            return True
        elif amp_cand > 10 and score >= 0.05:
            return True
        elif amp_cand > 5:  # Accept very quiet signals with no harmonic validation
            return True
        return False
    
    def _apply_persistence_logic(self, freq_cand, amp_cand, accept_candidate):
        """Apply persistence logic to avoid quick oscillations."""
        if not accept_candidate:
            # Do not promote candidate; keep previous detection if any
            return None, 0.0
        
        # Candidate accepted for consideration
        if self.prev_detect is None:
            # No previous note -> adopt after N frames
            if self.candidate is None or abs(self.candidate - freq_cand) > (freq_cand * self.config.FREQ_TOLERANCE):
                # Reset candidate if different freq
                self.candidate = freq_cand
                self.candidate_count = 1
            else:
                self.candidate_count += 1
            
            if self.candidate_count >= self.config.PERSISTENCE_FRAMES:
                final_freq = freq_cand
                final_amp = amp_cand
                return final_freq, final_amp
            else:
                return None, 0.0
        else:
            # There is a previous detection - more relaxed rules
            if abs(freq_cand - self.prev_detect) / self.prev_detect < self.config.FREQ_TOLERANCE * 1.5:
                # Same note family => update immediately
                final_freq = freq_cand
                final_amp = amp_cand
                self.candidate = freq_cand
                self.candidate_count = 0
                return final_freq, final_amp
            else:
                # New distinct candidate
                if amp_cand > self.prev_amp * self.config.STRONGER_FACTOR * 0.8:
                    # Less strict amplitude threshold -> immediate replace
                    final_freq = freq_cand
                    final_amp = amp_cand
                    self.candidate = None
                    self.candidate_count = 0
                    return final_freq, final_amp
                else:
                    # Minimal persistence requirement
                    if self.candidate is None or abs(self.candidate - freq_cand) > (freq_cand * self.config.FREQ_TOLERANCE * 1.5):
                        self.candidate = freq_cand
                        self.candidate_count = 1
                    else:
                        self.candidate_count += 1
                    
                    if self.candidate_count >= self.config.PERSISTENCE_FRAMES:
                        final_freq = freq_cand
                        final_amp = amp_cand
                        return final_freq, final_amp
                    else:
                        return None, 0.0
    
    def _update_ui_with_detection(self, final_freq, final_amp, is_octave_certain=True):
        """Update the UI with the detected note."""
        # Update previous detection
        self.prev_detect = final_freq
        self.prev_amp = final_amp
        
        # Get note info
        note_name, midi, cents = self.note_detector.freq_to_note_cents(final_freq)
        
        # Find closest guitar note for tuning reference
        closest_guitar_note = None
        min_distance = float('inf')
        for note, freq in self.config.GUITAR_NOTES.items():
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
        text = f"<span style='font-size: 16pt; color: white;'><b>{note_name}</b></span>"
        
        # Add an indicator if the octave detection might not be reliable
        if not is_octave_certain:
            text += " <span style='font-size: 10pt; color: #ffaa00;'>(octave estimate)</span>"
            
        text += f"<br><span style='font-size: 12pt; color: {color};'><b>{tune_status}</b></span><br>"
        text += f"<span style='font-size: 10pt; color: #aaaaaa;'>{final_freq:.1f} Hz ({cents:+.1f}¢)</span>"
        
        # If it's close to a guitar note, show which string
        if closest_guitar_note and min_distance < 0.1:
            note_name, freq = closest_guitar_note
            text += f"<br><span style='color: #88ff88;'>Guitar string: {note_name}</span>"
        
        self.note_text.setHtml(text)
        
        # Show tuning indicator
        x_pos = [final_freq]
        y_pos = [50 + cents/2]  # Center at 50, scale cents for visibility
        self.tuning_indicator.setData(x_pos, y_pos)
    
    def _show_candidate_info(self, freq_cand, amp_cand, is_octave_certain=True):
        """Show information about an unconfirmed candidate."""
        note_name, midi, cents = self.note_detector.freq_to_note_cents(freq_cand)
        
        text = f"<span style='font-size: 14pt; color: #aaaaaa;'><b>{note_name}</b></span>"
        
        # Add an indicator if the octave detection might not be reliable
        if not is_octave_certain:
            text += " <span style='font-size: 10pt; color: #ffaa00;'>(octave estimate)</span>"
            
        text += f"<br><span style='font-size: 10pt; color: #888888;'>{freq_cand:.1f} Hz ({cents:+.1f}¢)</span>"
        self.note_text.setHtml(text)
        
        # Update position of the current note line with less opacity
        self.current_note_line.setPos(freq_cand)
        self.current_note_line.setPen(pg.mkPen('w', width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.current_note_line.show()
        
        # Show indicator for candidate
        x_pos = [freq_cand]
        y_pos = [50 + cents/2]
        self.tuning_indicator.setData(x_pos, y_pos)
    
    def _show_no_detection_info(self):
        """Show information when no note is detected."""
        if self.prev_detect:
            note_name, midi, cents = self.note_detector.freq_to_note_cents(self.prev_detect)
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
