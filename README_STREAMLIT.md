# 🎸 Guitar Note Classifier - Streamlit App

Real-time guitar note detection using deep learning, built with Streamlit and PyTorch.

## 🌟 Features

- **Live Recording**: Record guitar notes directly from your microphone
- **File Upload**: Upload audio files for analysis
- **Real-time Classification**: Get instant predictions with confidence scores
- **Top 5 Predictions**: See alternative note predictions
- **Guitar Context**: Identify which guitar string corresponds to detected notes
- **Audio Quality Analysis**: Monitor recording levels and quality

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo>
   cd final_model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub**
   - Create a new GitHub repository
   - Push your code including `streamlit_app.py`, `requirements.txt`, and `realistic_guitar_classifier_final.pth`

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `final_model/streamlit_app.py`
   - Click "Deploy"

### Alternative Deployment Options

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-guitar-classifier
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📁 File Structure

```
final_model/
├── streamlit_app.py              # Main Streamlit application
├── realistic_guitar_classifier_final.pth  # Trained PyTorch model
├── requirements.txt              # Python dependencies
├── README.md                    # This file
└── live.py                     # Original CLI version
```

## 🎯 Model Details

- **Architecture**: Deep Neural Network with BatchNorm and Dropout
- **Input Features**: 206 features (FFT, Mel-spectrogram, Chroma, etc.)
- **Output Classes**: 37 guitar notes (E2 to E5)
- **Confidence Scoring**: Softmax probability distribution

## 🎸 Supported Notes

The model recognizes all chromatic notes in the guitar range:
- **E2** (6th string, low E)
- **A2** (5th string)
- **D3** (4th string)
- **G3** (3rd string)
- **B3** (2nd string)
- **E4** (1st string, high E)
- Plus all sharps/flats in between

## 🔧 Technical Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **Streamlit**: 1.28+
- **Microphone**: Required for live recording
- **Browser**: Modern browser with audio support

## 💡 Usage Tips

1. **For best results**: Play single notes clearly and let them ring out
2. **Avoid background noise**: Record in a quiet environment
3. **Check audio levels**: Use the built-in quality checker
4. **Try different positions**: Different frets of the same string should give same note

## 🐛 Troubleshooting

### Microphone Issues
- Check browser permissions for microphone access
- Ensure microphone is not being used by other applications
- Try refreshing the page and allowing permissions again

### Model Loading Issues
- Ensure `realistic_guitar_classifier_final.pth` is in the same directory
- Check file permissions and disk space
- Verify PyTorch installation

### Audio Quality Issues
- Increase recording volume if RMS level is too low
- Reduce volume if amplitude is clipping (>1.0)
- Try different microphone positions

## 📊 Performance

- **Real-time processing**: < 2 seconds from recording to prediction
- **Accuracy**: High accuracy on clean guitar recordings
- **Model size**: ~500KB PyTorch model file

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📜 License

This project is for educational and personal use.

---

**Built by GaragaKarthikeya** | Powered by Streamlit & PyTorch 🎸
