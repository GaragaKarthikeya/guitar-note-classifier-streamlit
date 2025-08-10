#!/bin/bash

# 🎸 Guitar Note Classifier - Streamlit Deployment Script
# User: GaragaKarthikeya

echo "🎸 Guitar Note Classifier - Streamlit Deployment"
echo "================================================="

# Check if required files exist
echo "📋 Checking required files..."

if [ ! -f "realistic_guitar_classifier_final.pth" ]; then
    echo "❌ Model file 'realistic_guitar_classifier_final.pth' not found!"
    echo "   Please make sure the model file is in the current directory."
    exit 1
fi

if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Streamlit app file 'streamlit_app.py' not found!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ Requirements file 'requirements.txt' not found!"
    exit 1
fi

echo "✅ All required files found!"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies!"
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Run the app
echo ""
echo "🚀 Starting Streamlit app..."
echo "   App will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py
