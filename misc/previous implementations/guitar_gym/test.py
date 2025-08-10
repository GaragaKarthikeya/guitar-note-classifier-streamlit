#!/usr/bin/env python3
"""
Guitar Note Detector with Rich Terminal GUI
Author: GaragaKarthikeya
Date: 2025-08-09
"""

import numpy as np
import pyaudio
import threading
import time
from collections import deque
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align

class RichGuitarDetector:
    def __init__(self, rate=44100, chunk_size=8192):
        self.rate = rate
        self.chunk_size = chunk_size
        
        # Guitar frequencies
        self.guitar_notes = {
            82.41: 'E2', 110.0: 'A2', 146.83: 'D3', 
            196.0: 'G3', 246.94: 'B3', 329.63: 'E4',
            87.31: 'F2', 92.50: 'F#2', 98.00: 'G2',
            103.83: 'G#2', 116.54: 'A#2', 123.47: 'B2',
            130.81: 'C3', 138.59: 'C#3', 155.56: 'D#3',
            164.81: 'E3', 174.61: 'F3', 185.00: 'F#3',
            207.65: 'G#3', 220.00: 'A3', 233.08: 'A#3',
            261.63: 'C4', 277.18: 'C#4', 293.66: 'D4',
            311.13: 'D#4', 349.23: 'F4', 369.99: 'F#4',
            392.00: 'G4', 415.30: 'G#4'
        }
        
        # State
        self.is_running = False
        self.current_note = None
        self.current_frequency = 0
        self.confidence = 0
        self.volume_level = 0
        self.note_history = deque(maxlen=20)
        
        # Audio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Rich console
        self.console = Console()
        
    def detect_pitch_autocorrelation(self, audio_data):
        """Autocorrelation pitch detection"""
        windowed = audio_data * np.hanning(len(audio_data))
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        min_period = int(self.rate / 500)  # Max freq 500Hz
        max_period = int(self.rate / 70)   # Min freq 70Hz
        
        if max_period >= len(autocorr):
            return 0, 0
            
        search_range = autocorr[min_period:max_period]
        if len(search_range) == 0:
            return 0, 0
            
        peak_idx = np.argmax(search_range) + min_period
        confidence = autocorr[peak_idx] / autocorr[0] if autocorr[0] > 0 else 0
        frequency = self.rate / peak_idx if peak_idx > 0 else 0
        
        return frequency, confidence
    
    def frequency_to_note(self, frequency):
        """Convert frequency to note"""
        if frequency < 70 or frequency > 500:
            return None, 0
            
        closest_freq = min(self.guitar_notes.keys(), 
                          key=lambda x: abs(x - frequency))
        cents_off = 1200 * np.log2(frequency / closest_freq)
        note_confidence = max(0, 1 - abs(cents_off) / 50)
        
        if note_confidence > 0.5:
            return self.guitar_notes[closest_freq], note_confidence
        return None, 0
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio processing callback"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.volume_level = np.sqrt(np.mean(audio_data**2))
        
        if self.volume_level > 0.001:
            frequency, confidence = self.detect_pitch_autocorrelation(audio_data)
            
            if confidence > 0.3:
                note, note_confidence = self.frequency_to_note(frequency)
                if note:
                    self.current_note = note
                    self.current_frequency = frequency
                    self.confidence = confidence * note_confidence
                    self.note_history.append((note, time.time()))
                else:
                    self.current_note = None
            else:
                self.current_note = None
        else:
            self.current_note = None
            
        return (in_data, pyaudio.paContinue)
    
    def get_tuning_status(self):
        """Get tuning status for current note"""
        if not self.current_note or self.current_frequency == 0:
            return "---", "white"
            
        target_freq = next((freq for freq, note in self.guitar_notes.items() 
                           if note == self.current_note), 0)
        
        if target_freq == 0:
            return "---", "white"
            
        cents = 1200 * np.log2(self.current_frequency / target_freq)
        
        if abs(cents) < 5:
            return "🎯 IN TUNE", "green"
        elif cents > 0:
            return f"📈 +{cents:.0f} cents (SHARP)", "red"
        else:
            return f"📉 {cents:.0f} cents (FLAT)", "yellow"
    
    def create_volume_bar(self):
        """Create volume level bar"""
        volume_percent = min(100, self.volume_level * 100)
        bar_length = 30
        filled = int(volume_percent * bar_length / 100)
        
        bar = "█" * filled + "░" * (bar_length - filled)
        color = "green" if volume_percent > 10 else "red"
        
        return Text(f"🎤 {bar} {volume_percent:.0f}%", style=color)
    
    def create_note_display(self):
        """Create main note display panel"""
        if self.current_note:
            note_text = Text(self.current_note, style="bold cyan", justify="center")
            note_text.stylize("bold", 0, len(self.current_note))
            
            freq_text = f"🎵 {self.current_frequency:.1f} Hz"
            conf_text = f"🎯 {self.confidence:.0%} confidence"
        else:
            note_text = Text("♪ ♫ ♪", style="dim white", justify="center")
            freq_text = "🎵 --- Hz"
            conf_text = "🎯 --- confidence"
        
        content = Align.center(f"\n{note_text}\n\n{freq_text}\n{conf_text}\n")
        
        return Panel(
            content,
            title="🎸 DETECTED NOTE",
            border_style="cyan",
            height=7
        )
    
    def create_tuning_panel(self):
        """Create tuning status panel"""
        tuning_text, color = self.get_tuning_status()
        
        # Create tuning meter
        if self.current_note and self.current_frequency > 0:
            target_freq = next((freq for freq, note in self.guitar_notes.items() 
                               if note == self.current_note), 0)
            if target_freq > 0:
                cents = 1200 * np.log2(self.current_frequency / target_freq)
                cents = max(-50, min(50, cents))
                
                meter_width = 40
                center = meter_width // 2
                position = center + int(cents * center / 50)
                
                meter = ["─"] * meter_width
                meter[center] = "│"
                meter[position] = "●"
                
                meter_str = "".join(meter)
                meter_display = f"FLAT ←{meter_str}→ SHARP"
            else:
                meter_display = "No tuning reference"
        else:
            meter_display = "Play a note to see tuning"
        
        content = f"\n{tuning_text}\n\n{meter_display}\n"
        
        return Panel(
            content,
            title="🎯 TUNING STATUS",
            border_style=color,
            height=7
        )
    
    def create_history_panel(self):
        """Create note history panel"""
        if self.note_history:
            recent_notes = list(self.note_history)[-8:]
            notes_text = " → ".join([note for note, _ in recent_notes])
        else:
            notes_text = "No notes detected yet..."
        
        return Panel(
            f"\n{notes_text}\n",
            title="📜 RECENT NOTES",
            border_style="magenta",
            height=5
        )
    
    def create_info_panel(self):
        """Create info panel"""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column(style="cyan")
        info_table.add_column(style="white")
        
        info_table.add_row("⏰ Time:", current_time)
        info_table.add_row("🎼 Sample Rate:", f"{self.rate} Hz")
        info_table.add_row("📊 Buffer Size:", f"{self.chunk_size}")
        info_table.add_row("🎸 Total Notes:", str(len(self.guitar_notes)))
        
        return Panel(
            info_table,
            title="ℹ️ SYSTEM INFO",
            border_style="blue",
            height=8
        )
    
    def create_layout(self):
        """Create the main layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        layout["left"].split_column(
            Layout(name="note_display"),
            Layout(name="tuning_status")
        )
        
        layout["right"].split_column(
            Layout(name="history"),
            Layout(name="info")
        )
        
        # Header
        title = Text("🎸 GUITAR NOTE DETECTOR 🎸", style="bold green")
        layout["header"].update(Align.center(title))
        
        # Main panels
        layout["note_display"].update(self.create_note_display())
        layout["tuning_status"].update(self.create_tuning_panel())
        layout["history"].update(self.create_history_panel())
        layout["info"].update(self.create_info_panel())
        
        # Footer
        volume_bar = self.create_volume_bar()
        footer_text = Text("\nPress Ctrl+C to quit", style="dim")
        footer_content = Align.center(f"{volume_bar}\n{footer_text}")
        layout["footer"].update(footer_content)
        
        return layout
    
    def start(self):
        """Start the detector with Rich interface"""
        try:
            self.console.print("[bold green]🎸 Starting Guitar Note Detector...[/]")
            self.console.print("[cyan]🎤 Initializing audio...[/]")
            
            # Setup audio
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.is_running = True
            self.stream.start_stream()
            
            self.console.print("[bold green]✅ Audio started! Listening for guitar notes...[/]")
            time.sleep(2)
            
            # Start Rich live display
            with Live(self.create_layout(), refresh_per_second=10, screen=True) as live:
                while self.is_running:
                    live.update(self.create_layout())
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.console.print("\n[bold red]🛑 Stopping detector...[/]")
            self.stop()
        except Exception as e:
            self.console.print(f"[bold red]❌ Error: {e}[/]")
            self.stop()
    
    def stop(self):
        """Stop the detector"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.console.print("[bold green]✅ Guitar Note Detector stopped![/]")

def main():
    """Main function"""
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]🎸 Guitar Note Detector[/]\n"
        "[white]Make sure your microphone is connected![/]",
        border_style="green"
    ))
    
    detector = RichGuitarDetector()
    detector.start()

if __name__ == "__main__":
    main()