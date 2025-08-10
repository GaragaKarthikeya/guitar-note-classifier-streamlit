"""
Note detection module for the guitar tuner.
Handles frequency detection, note mapping, and tuning analysis.
"""

import numpy as np
from scipy.signal import find_peaks

class NoteDetector:
    def __init__(self, config):
        """Initialize the note detector with the given configuration."""
        self.config = config
        self.NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    def parabolic_interpolation(self, mag, idx):
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
    
    def freq_to_note_cents(self, f):
        """
        Simple, direct frequency to note conversion with debugging.
        Maximum processing power, minimal complexity.
        """
        if f <= 0:
            return None, None, None
            
        # Direct MIDI calculation - no complex logic
        midi_float = 69 + 12 * np.log2(f / 440.0)
        midi_round = int(np.round(midi_float))
        cents = (midi_float - midi_round) * 100
        
        # Simple octave calculation
        note_index = midi_round % 12
        octave = (midi_round // 12) - 1
        
        name = self.NOTE_NAMES[note_index]
        
        # Debug print for troubleshooting
        # print(f"Debug: {f:.1f}Hz -> MIDI {midi_round} -> {name}{octave}")
        
        return f"{name}{octave}", midi_round, cents
    
    def harmonic_validation(self, fund_freq, spectrum, bins, freq_bins):
        """
        Simple harmonic validation - maximum processing power, minimal complexity.
        """
        # Simple amplitude check at fundamental
        idx = np.argmin(np.abs(freq_bins - fund_freq))
        fund_amp = spectrum[idx] if idx < bins else 0.0
        
        if fund_amp <= self.config.MIN_AMP_THRESHOLD:
            return 0.0, {}, False
            
        found = 0
        details = {}
        
        # Simple harmonic check - just 2nd and 3rd
        for h in range(2, 2 + self.config.HARMONIC_COUNT):
            target = fund_freq * h
            if target > freq_bins[-1]:
                details[h] = False
                continue
                
            # Find nearest bin - simple approach
            idx_h = np.argmin(np.abs(freq_bins - target))
            amp_h = spectrum[idx_h]
            ratio = amp_h / (fund_amp + 1e-12)
            
            details[h] = (ratio >= self.config.HARMONIC_RATIO_THRESHOLD)
            if details[h]:
                found += 1
                
        score = found / float(self.config.HARMONIC_COUNT)
        is_octave_certain = score > 0.3  # Simple threshold
        
        return score, details, is_octave_certain
    
    def find_peaks_and_identify(self, spectrum, display_freqs):
        """
        Simple peak detection with fundamental frequency emphasis.
        Addresses octave detection issues by preferring lower frequencies.
        """
        # Simple peak detection with minimal thresholds
        peaks, props = find_peaks(spectrum,
                                height=self.config.MIN_PEAK_HEIGHT,
                                prominence=self.config.MIN_PEAK_PROMINENCE,
                                distance=self.config.PEAK_DISTANCE_BINS)
        
        if len(peaks) == 0:
            return None
            
        # Get all peaks with their amplitudes
        peak_candidates = []
        for p in peaks:
            # Simple parabolic interpolation
            refined_bin, amp = self.parabolic_interpolation(spectrum, p)
            refined_freq = np.interp(refined_bin, np.arange(len(display_freqs)), display_freqs)
            
            # Fundamental frequency emphasis: prefer lower frequencies (guitar fundamentals)
            # This helps avoid detecting harmonics instead of fundamentals
            if refined_freq < 100:  # Low E region - give extra weight
                freq_weight = 2.0
            elif refined_freq < 200:  # A, D region - moderate weight
                freq_weight = 1.5
            elif refined_freq < 350:  # G, B, high E region - normal weight
                freq_weight = 1.0
            else:  # Higher frequencies - likely harmonics, reduce weight
                freq_weight = 0.5
            
            # Simple harmonic validation
            h_score, h_details, is_octave_certain = self.harmonic_validation(
                refined_freq, spectrum, len(display_freqs), display_freqs)
            
            # Combined score: emphasize fundamentals over harmonics
            combined_score = amp * freq_weight * (1.0 + h_score * 0.5)
            peak_candidates.append((refined_freq, amp, combined_score, is_octave_certain))
            
        # Sort by combined score and return the best
        peak_candidates.sort(key=lambda x: x[2], reverse=True)
        f, a, _, is_octave_certain = peak_candidates[0]
        return f, a, is_octave_certain
