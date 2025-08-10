#!/usr/bin/env python3
"""
Streamlit Guitar Note Recorder - Real-time guitar note classification!
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

# ========================================================================================
# STREAMLIT GUITAR CLASSIFIER
# ========================================================================================

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
    """Classify the audio data"""
    
    # Check audio quality
    rms = np.sqrt(np.mean(audio_data**2))
    max_amplitude = np.max(np.abs(audio_data))
    
    with st.expander("📊 Audio Quality Check"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔊 RMS Level", f"{rms:.4f}")
        with col2:
            st.metric("📈 Max Amplitude", f"{max_amplitude:.4f}")
        
        if rms < 0.001:
            st.warning("⚠️ Very quiet audio - try playing louder!")
        elif rms > 0.1:
            st.warning("⚠️ Very loud audio - might be clipping!")
        else:
            st.success("✅ Good audio level")
    
    # Extract features
    with st.spinner("🔄 Extracting features..."):
        features = extract_frequency_features(audio_data, sample_rate)
    
    if features is None:
        st.error("❌ Feature extraction failed!")
        return None, 0.0, []
    
    st.success(f"✅ {len(features)} features extracted")
    
    # Normalize and predict
    with st.spinner("🧠 Running neural network..."):
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
    """Display prediction results in Streamlit"""
    
    st.subheader("🎯 Prediction Results")
    
    if predicted_note:
        # Main prediction
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric("🏆 Predicted Note", predicted_note)
        
        with col2:
            st.metric("📊 Confidence", f"{confidence*100:.1f}%")
        
        with col3:
            if confidence >= 0.9:
                st.success("🔥 VERY HIGH")
            elif confidence >= 0.7:
                st.success("✅ HIGH")
            elif confidence >= 0.5:
                st.warning("🟡 MODERATE")
            else:
                st.error("🟠 LOW")
        
        # Progress bar for confidence
        st.progress(confidence)
        
        # Top 5 predictions
        st.subheader("📈 Top 5 Predictions")
        
        markers = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, (note, conf) in enumerate(top5_results):
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.write(markers[i])
            with col2:
                st.write(f"**{note}**")
            with col3:
                st.progress(conf)
                st.write(f"{conf*100:.1f}%")
        
        # Special notes and context
        if predicted_note == 'A4':
            st.info("🎯 A4 DETECTED! Perfect for guitar tuning! (440 Hz)")
        
        # Guitar string context
        string_map = {
            'E2': '6th string (low E)',
            'A2': '5th string (A)',
            'D3': '4th string (D)',
            'G3': '3rd string (G)',
            'B3': '2nd string (B)',
            'E4': '1st string (high E)'
        }
        
        if predicted_note in string_map:
            st.info(f"🎸 Guitar context: {string_map[predicted_note]}")
    
    else:
        st.error("❌ Could not detect note - try recording again")

# ========================================================================================
# MAIN STREAMLIT APP
# ========================================================================================

def main():
    # Header
    st.title("🎸 Guitar Note Classifier")
    st.markdown("**Real-time guitar note detection using deep learning**")
    st.markdown(f"👤 User: GaragaKarthikeya | 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Audio settings
        st.subheader("🎤 Recording Settings")
        duration = st.slider("Recording Duration (seconds)", 1, 10, 5)
        sample_rate = st.selectbox("Sample Rate", [22050, 44100], index=0)
        
        st.subheader("📊 Model Info")
        model, scaler, reverse_mapping, device = load_model()
        
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Parameters", f"{total_params:,}")
        st.metric("Device", str(device))
        st.metric("Classes", len(reverse_mapping))
        
        # Instructions
        st.subheader("📝 Instructions")
        st.markdown("""
        1. **Upload** an audio file OR **record** live
        2. **Play** a single guitar note clearly
        3. **Get** instant prediction with confidence!
        
        **Supported Notes:**
        - All chromatic notes from E2 to E5
        - Perfect for guitar tuning
        - Works with electric and acoustic guitars
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["🎤 Live Recording", "📁 File Upload"])
    
    with tab1:
        st.header("🎤 Live Audio Recording")
        st.markdown("Use your device's microphone to record a guitar note.")
        
        # Use Streamlit's audio input widget
        audio_value = st.audio_input("Record a guitar note")
        
        if audio_value is not None:
            try:
                # Load audio from the recorded data
                audio_data, file_sample_rate = librosa.load(audio_value, sr=sample_rate)
                
                # Display audio info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duration", f"{len(audio_data)/file_sample_rate:.2f}s")
                with col2:
                    st.metric("Sample Rate", f"{file_sample_rate} Hz")
                
                # Display audio player
                st.audio(audio_value)
                
                # Classify audio
                predicted_note, confidence, top5_results = classify_audio(
                    audio_data, sample_rate, model, scaler, reverse_mapping, device
                )
                
                # Display results
                display_results(predicted_note, confidence, top5_results)
                
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
                st.info("💡 Please ensure you recorded a clear guitar note.")
    
    with tab2:
        st.header("📁 Audio File Upload")
        st.markdown("Upload an audio file (WAV, MP3, etc.) to classify the guitar note.")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file...",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="Upload an audio file containing a single guitar note"
        )
        
        if uploaded_file is not None:
            try:
                # Load audio file
                audio_data, file_sample_rate = librosa.load(uploaded_file, sr=sample_rate)
                
                # Display audio info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{len(audio_data)/file_sample_rate:.2f}s")
                with col2:
                    st.metric("Sample Rate", f"{file_sample_rate} Hz")
                with col3:
                    st.metric("Samples", f"{len(audio_data):,}")
                
                # Audio player
                st.audio(uploaded_file, format='audio/wav')
                
                # Classify button
                if st.button("🧠 Analyze Audio", type="primary", use_container_width=True):
                    # Classify audio
                    predicted_note, confidence, top5_results = classify_audio(
                        audio_data, file_sample_rate, model, scaler, reverse_mapping, device
                    )
                    
                    # Display results
                    display_results(predicted_note, confidence, top5_results)
                
            except Exception as e:
                st.error(f"❌ Error loading audio file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("🎸 **Guitar Note Classifier** | Built with Streamlit & PyTorch | By GaragaKarthikeya")

if __name__ == "__main__":
    main()
