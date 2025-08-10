#!/usr/bin/env python3
"""
Streamlit Guitar Note Recorder - Enhanced UI with intuitive recording!
User: GaragaKarthikeya
Date: 2025-08-10
"""

import streamlit as st
import torch
import librosa
import numpy as np
import wave
import time
import os
import tempfile
import io
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="🎸 Guitar Note Classifier",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #ff6b6b;
        --secondary-color: #4ecdc4;
        --accent-color: #45b7d1;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background: var(--background-gradient);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255,107,107,0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .prediction-note {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .confidence-meter {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Guitar fretboard styling */
    .fretboard {
        background: #8B4513;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Recording interface styling */
    .recording-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        text-align: center;
        margin: 1rem 0;
        border: 3px solid #ff6b6b;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ========================================================================================
# SAME MODEL AS BEFORE
# ========================================================================================

class RealisticGuitarNetwork(torch.nn.Module):
    def __init__(self, input_size=206, num_classes=37, dropout_rate=0.5):
        super(RealisticGuitarNetwork, self).__init__()
        
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128), 
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate - 0.1)
        )
        
        self.classifier = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

def extract_frequency_features(audio_data, sr):
    """Extract features (same as before)"""
    try:
        # FFT Analysis
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft)
        magnitude = magnitude[:len(magnitude)//2]
        
        freqs = np.fft.fftfreq(len(audio_data), 1/sr)[:len(magnitude)]
        
        # Musical range: 80-2000 Hz
        min_freq, max_freq = 80, 2000
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        musical_freqs = freqs[freq_mask]
        musical_magnitude = magnitude[freq_mask]
        
        # 128 frequency bins
        num_bins = 128
        bin_edges = np.linspace(min_freq, max_freq, num_bins + 1)
        freq_bins = np.zeros(num_bins)
        
        for i in range(num_bins):
            bin_mask = (musical_freqs >= bin_edges[i]) & (musical_freqs < bin_edges[i+1])
            if np.any(bin_mask):
                freq_bins[i] = np.mean(musical_magnitude[bin_mask])
        
        if np.max(freq_bins) > 0:
            freq_bins = freq_bins / np.max(freq_bins)
        
        # Mel features (64)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=64, 
                                                 fmin=80, fmax=2000)
        mel_features = np.mean(mel_spec, axis=1)
        if np.max(mel_features) > 0:
            mel_features = mel_features / np.max(mel_features)
        
        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_chroma=12)
        chroma_features = np.mean(chroma, axis=1)
        
        # Spectral centroid (1)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        centroid_feature = [np.mean(spectral_centroid)]
        
        # Fundamental frequency (1)
        try:
            f0 = librosa.yin(audio_data, fmin=80, fmax=800, sr=sr)
            f0_clean = f0[f0 > 0]
            fundamental_freq = [np.median(f0_clean)] if len(f0_clean) > 0 else [0.0]
        except:
            fundamental_freq = [0.0]
        
        # Combine all features (206 total)
        combined_features = np.concatenate([
            freq_bins, mel_features, chroma_features, 
            centroid_feature, fundamental_freq
        ])
        
        return combined_features.astype(np.float32)
        
    except Exception as e:
        st.error(f"❌ Feature extraction error: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    model_path = "realistic_guitar_classifier_final.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = RealisticGuitarNetwork().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = checkpoint['scaler']
        reverse_mapping = checkpoint['reverse_mapping']
        
        return model, scaler, reverse_mapping, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def classify_audio(audio_data, sample_rate, model, scaler, reverse_mapping, device):
    """Classify the audio data with enhanced feedback"""
    
    # Enhanced audio quality check
    rms = np.sqrt(np.mean(audio_data**2))
    max_amplitude = np.max(np.abs(audio_data))
    zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
    
    with st.expander("📊 Audio Quality Analysis", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔊 RMS Level", f"{rms:.4f}")
            if rms < 0.001:
                st.error("Too quiet!")
            elif rms > 0.1:
                st.warning("Too loud!")
            else:
                st.success("Good level ✅")
        
        with col2:
            st.metric("📈 Peak Amplitude", f"{max_amplitude:.4f}")
            if max_amplitude > 0.95:
                st.error("Clipping detected!")
            elif max_amplitude < 0.01:
                st.warning("Very weak signal")
            else:
                st.success("Clean signal ✅")
        
        with col3:
            st.metric("🌊 Zero Crossings", zero_crossings)
            if zero_crossings < 100:
                st.info("Steady tone")
            elif zero_crossings > 1000:
                st.warning("Noisy signal")
            else:
                st.success("Clear note ✅")
    
    # Extract features with progress
    with st.spinner("🔄 Extracting audio features..."):
        features = extract_frequency_features(audio_data, sample_rate)
    
    if features is None:
        st.error("❌ Feature extraction failed!")
        return None, 0.0, []
    
    st.success(f"✅ Successfully extracted {len(features)} audio features")
    
    # Normalize and predict with detailed feedback
    with st.spinner("🧠 Running neural network inference..."):
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            output = model(features_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top 5 predictions
            top5_conf, top5_pred = torch.topk(probabilities, 5, dim=1)
    
    # Format results
    predicted_note = reverse_mapping[top5_pred[0][0].item()]
    confidence = top5_conf[0][0].item()
    
    top5_results = []
    for i in range(5):
        note = reverse_mapping[top5_pred[0][i].item()]
        conf = top5_conf[0][i].item()
        top5_results.append((note, conf))
    
    return predicted_note, confidence, top5_results

def display_results(predicted_note, confidence, top5_results):
    """Display prediction results with enhanced UI"""
    
    if predicted_note:
        # Main prediction card
        st.markdown(f"""
        <div class="result-card">
            <h2>🎯 Detected Note</h2>
            <div class="prediction-note">{predicted_note}</div>
            <div class="confidence-meter">
                <h4>Confidence: {confidence*100:.1f}%</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence visualization
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if confidence >= 0.9:
                st.success("🔥 EXCELLENT DETECTION!")
            elif confidence >= 0.7:
                st.success("✅ HIGH CONFIDENCE")
            elif confidence >= 0.5:
                st.warning("🟡 MODERATE CONFIDENCE")
            else:
                st.error("🟠 LOW CONFIDENCE - Try again")
            
            # Enhanced progress bar
            st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
        
        # Top 5 predictions with enhanced styling
        st.subheader("📊 Alternative Predictions")
        
        for i, (note, conf) in enumerate(top5_results):
            rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#4169E1", "#9370DB"]
            rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
            
            with col1:
                st.markdown(f"<h3 style='color:{rank_colors[i]}'>{rank_icons[i]}</h3>", 
                           unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{note}**")
            with col3:
                st.progress(conf, text=f"{conf*100:.1f}%")
            with col4:
                if i == 0:
                    st.success("Primary")
                elif conf > 0.1:
                    st.info("Alternative")
                else:
                    st.warning("Low prob.")
        
        # Musical context and tips
        display_musical_context(predicted_note, confidence)
        
    else:
        st.error("❌ Could not detect note - try recording again")
        st.info("💡 Please ensure you recorded a clear guitar note and try again.")

def display_musical_context(predicted_note, confidence):
    """Display musical context and information about the detected note"""
    
    st.subheader("🎼 Musical Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Note frequency information
        note_frequencies = {
            'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83,
            'A2': 110.00, 'A#2': 116.54, 'B2': 123.47, 'C3': 130.81, 'C#3': 138.59,
            'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 'F#3': 185.00,
            'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
            'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
            'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00,
            'A#4': 466.16, 'B4': 493.88, 'C5': 523.25, 'C#5': 554.37, 'D5': 587.33,
            'D#5': 622.25, 'E5': 659.25
        }
        
        if predicted_note in note_frequencies:
            freq = note_frequencies[predicted_note]
            st.metric("🎵 Frequency", f"{freq:.2f} Hz")
        
        # Guitar string context
        string_map = {
            'E2': '6th string (Low E)',
            'A2': '5th string (A)',
            'D3': '4th string (D)',
            'G3': '3rd string (G)',
            'B3': '2nd string (B)',
            'E4': '1st string (High E)'
        }
        
        if predicted_note in string_map:
            st.success(f"🎸 **Open String:** {string_map[predicted_note]}")
        else:
            st.info("🎸 **Fretted Note:** Play this on the fretboard")
    
    with col2:
        # Special notes and tuning reference
        if predicted_note == 'A4':
            st.success("🎯 **A4 - Concert Pitch!**")
            st.info("Perfect reference for tuning (440 Hz)")
        
        if predicted_note in ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']:
            st.success("🎵 **Open String Detected!**")
            st.info("Great for tuning your guitar")
        
        # Practice suggestions
        st.markdown("**🎯 Practice Tips:**")
        st.markdown(f"""
        - Practice scales containing **{predicted_note}**
        - Work on intonation for this pitch
        - Try harmonics at this frequency
        """)
        
        if confidence > 0.8:
            st.success("🔥 **Excellent pitch accuracy!**")

def main():
    # Enhanced header with hero section
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🎸 Guitar Note Classifier</div>
        <div class="subtitle">AI-Powered Real-time Guitar Note Detection</div>
        <p>🧠 Deep Learning • 🎵 37 Note Classes • 🎯 High Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats in header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎵 Supported Notes", "37")
    with col2:
        st.metric("🎸 Guitar Strings", "6")
    with col3:
        st.metric("⚡ Processing", "Real-time")
    with col4:
        st.metric("🧠 Accuracy", "92%")
    
    # Call-to-action section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
               color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; 
               text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
        <h2 style="margin-bottom: 1rem; font-size: 2.5rem;">🎸 Ready to Identify Your Guitar Note?</h2>
        <p style="font-size: 1.3rem; margin-bottom: 1rem; opacity: 0.9;">
            Our AI-powered neural network can instantly recognize any guitar note you play!
        </p>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h3 style="margin-bottom: 0.5rem;">🚀 How it works:</h3>
            <p style="font-size: 1.1rem; margin: 0;">
                Record → AI Analysis → Instant Results in seconds!
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Audio settings in an expander
        with st.expander("🎤 Recording Settings", expanded=True):
            sample_rate = st.selectbox(
                "Sample Rate", 
                [22050, 44100], 
                index=0,
                help="Higher sample rate = better quality, slower processing"
            )
            
            st.markdown("**🎯 Recording Guide:**")
            st.markdown("""
            1. 🔇 **Find a quiet space**
            2. 🎸 **Tune your guitar**
            3. 🎤 **Click record button**
            4. 🎵 **Play ONE note clearly**
            5. 🛑 **Stop recording**
            6. 🧠 **Get instant AI results!**
            """)
        
        # Model information
        with st.expander("🧠 AI Model Info"):
            model, scaler, reverse_mapping, device = load_model()
            
            total_params = sum(p.numel() for p in model.parameters())
            st.metric("Neural Network Size", f"{total_params:,} parameters")
            st.metric("Processing Device", str(device).upper())
            st.metric("Note Classes", f"{len(reverse_mapping)} notes")
            
            if 'cuda' in str(device):
                st.success("🚀 GPU Acceleration ENABLED")
            else:
                st.info("💻 CPU Processing")
            
            st.markdown("**🎯 Model Accuracy:**")
            st.progress(0.92, text="92% accuracy on test data")
        
        # Guitar tuning reference
        with st.expander("🎸 Guitar Reference"):
            st.markdown("**Standard Tuning:**")
            tuning_notes = [
                ("1st String (Thinnest)", "E4", "#ff6b6b"),
                ("2nd String", "B3", "#4ecdc4"),
                ("3rd String", "G3", "#45b7d1"),
                ("4th String", "D3", "#96ceb4"),
                ("5th String", "A2", "#feca57"),
                ("6th String (Thickest)", "E2", "#ff9ff3")
            ]
            
            for string, note, color in tuning_notes:
                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 0.5rem; 
                           margin: 0.3rem 0; border-radius: 8px; text-align: center;">
                    <strong>{string}: {note}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**🎵 Frequency Reference:**")
            st.markdown("""
            - **A4**: 440 Hz (tuning standard)
            - **E2**: 82 Hz (lowest guitar note)
            - **E4**: 330 Hz (highest open string)
            """)
        
        # Quick tips for better results
        with st.expander("💡 Pro Tips"):
            st.markdown("""
            **For Best Results:**
            
            🎸 **Guitar Setup:**
            - Use a well-tuned guitar
            - Fresh strings work better
            - Avoid old, dead strings
            
            🎤 **Recording Technique:**
            - Hold notes for 2-3 seconds
            - Play with consistent volume
            - Avoid touching other strings
            - Use a pick for clarity
            
            🏠 **Environment:**
            - Record in a quiet room
            - Avoid echo/reverb
            - Close windows/doors
            - Turn off fans/AC if possible
            """)
    
    # Main content - simplified to focus on recording
    st.markdown("### 🎤 Step 1: Record Your Guitar Note")
    
    # Visual step indicator
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 2rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="background: #ff6b6b; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem;">1</div>
            <div style="color: #666; font-size: 1.1rem;">Record</div>
            <div style="width: 50px; height: 2px; background: #ddd;"></div>
            <div style="background: #4ecdc4; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem;">2</div>
            <div style="color: #666; font-size: 1.1rem;">AI Analysis</div>
            <div style="width: 50px; height: 2px; background: #ddd;"></div>
            <div style="background: #45b7d1; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem;">3</div>
            <div style="color: #666; font-size: 1.1rem;">Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prominent recording section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #ff8e8e); 
               color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
               text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
        <h2 style="margin-bottom: 1rem;">🎤 Click Below to Start Recording!</h2>
        <p style="font-size: 1.2rem; margin-bottom: 0;">Play a single guitar note clearly and hold it for 2-3 seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recording instructions and tips
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: rgba(255,107,107,0.1); padding: 1rem; border-radius: 10px; 
                   border-left: 4px solid #ff6b6b;">
            <h4 style="color: #ff6b6b;">🎯 Quick Tips</h4>
            <ul style="margin-bottom: 0;">
                <li>Quiet room</li>
                <li>Single note only</li>
                <li>Hold for 2-3 seconds</li>
                <li>Play clearly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Main recording interface with enhanced styling
        st.markdown("""
        <div class="recording-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎤</div>
            <h3 style="color: #333; margin-bottom: 1rem;">Audio Recorder</h3>
            <p style="color: #666; margin-bottom: 1.5rem;">Click the microphone button below</p>
        """, unsafe_allow_html=True)
        
        # Audio input widget with better styling
        audio_value = st.audio_input(
            "",
            help="🎵 Click the microphone button to start recording your guitar note",
            label_visibility="collapsed"
        )
        
        if not audio_value:
            st.markdown("""
            <div style="background: rgba(255,107,107,0.1); color: #ff6b6b; 
                       padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <strong>👆 Waiting for your recording...</strong><br>
                Click the red microphone button above to start!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(76,205,196,0.1); color: #4ecdc4; 
                       padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <strong>✅ Recording captured!</strong><br>
                Processing your guitar note now...
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: rgba(78,205,196,0.1); padding: 1rem; border-radius: 10px; 
                   border-left: 4px solid #4ecdc4;">
            <h4 style="color: #4ecdc4;">🎸 Best Notes</h4>
            <ul style="margin-bottom: 0;">
                <li>Open strings (E, A, D, G, B, E)</li>
                <li>Fretted notes</li>
                <li>Harmonics</li>
                <li>Any single pitch</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Process recorded audio
    if audio_value is not None:
        try:
            # Add visual feedback for processing
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); 
                       color: white; padding: 1.5rem; border-radius: 10px; 
                       text-align: center; margin: 2rem 0;">
                <h3>🔄 Processing Your Recording...</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🎵 Loading your audio..."):
                # Load audio from the recorded data
                audio_data, file_sample_rate = librosa.load(audio_value, sr=sample_rate)
            
            # Success indicator
            st.success("✅ Audio recorded successfully!")
            
            # Audio info in a clean layout
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Duration", f"{len(audio_data)/file_sample_rate:.2f}s")
            with col2:
                st.metric("🔊 Sample Rate", f"{file_sample_rate} Hz")
            with col3:
                st.metric("📊 Quality", "Good" if len(audio_data) > 1000 else "Short")
            
            # Audio playback section
            st.markdown("### 🎧 Your Recording")
            st.audio(audio_value)
            
            # Automatic analysis (no button needed)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b, #ff8e8e); 
                       color: white; padding: 1.5rem; border-radius: 10px; 
                       text-align: center; margin: 2rem 0;">
                <h3>🧠 AI Analysis in Progress...</h3>
                <p>Our neural network is analyzing your guitar note</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Classify audio automatically
            with st.spinner("🤖 Neural network is analyzing your note..."):
                predicted_note, confidence, top5_results = classify_audio(
                    audio_data, sample_rate, model, scaler, reverse_mapping, device
                )
            
            # Display enhanced results
            display_results(predicted_note, confidence, top5_results)
            
            # Celebration and next action
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); 
                       color: white; padding: 2rem; border-radius: 15px; 
                       text-align: center; margin: 2rem 0;">
                <h3>🎉 Analysis Complete!</h3>
                <p>Want to try another note? Record again below!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prominent "Record Again" section
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🎤 Record Another Note", type="primary", use_container_width=True):
                    st.rerun()
                
                st.markdown("""
                <div style="text-align: center; margin-top: 1rem; color: #666;">
                    <small>💡 Try different strings, frets, or harmonics!</small>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Processing error: {e}")
            st.markdown("""
            <div style="background: rgba(220,53,69,0.1); padding: 1rem; border-radius: 10px; 
                       border-left: 4px solid #dc3545; margin: 1rem 0;">
                <h4 style="color: #dc3545;">🔧 Troubleshooting</h4>
                <ul>
                    <li>Check microphone permissions</li>
                    <li>Ensure you're playing a single note</li>
                    <li>Try recording in a quieter environment</li>
                    <li>Make sure your guitar is in tune</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: rgba(0,0,0,0.05); 
               border-radius: 10px; margin-top: 2rem;">
        <h4>🎸 Guitar Note Classifier</h4>
        <p>Built with ❤️ using Streamlit & PyTorch</p>
        <p>👨‍💻 Created by <strong>GaragaKarthikeya</strong> | 
           📅 {datetime.now().strftime('%Y')}</p>
        <p>🔬 <em>Powered by Deep Learning & Signal Processing</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
