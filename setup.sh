#!/bin/bash

echo "================================================================"
echo "     Audio Steganography Detection - Complete Setup"
echo "================================================================"
echo ""

echo "[1/5] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing dependencies. Please check requirements.txt"
    exit 1
fi

echo ""
echo "[2/5] Creating directory structure..."
mkdir -p models visualizations data/clean data/stego

echo ""
echo "[3/5] Generating dataset..."
python src/generate_dataset.py

echo ""
echo "[4/5] Training models..."
python src/model_train.py

echo ""
echo "[5/5] Running test prediction..."
python src/predict.py data/stego/sample_0.wav

echo ""
echo "================================================================"
echo "                    Setup Complete!"
echo "================================================================"
echo ""
echo "To launch the web application, run:"
echo "    streamlit run app/streamlit_app.py"
echo ""
echo "To train specific models, run:"
echo "    python src/model_train.py"
echo ""
echo "To make predictions, run:"
echo "    python src/predict.py path/to/audio.wav"
echo ""
