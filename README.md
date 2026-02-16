# Audio Steganography Detection System

Advanced machine learning-based system for detecting hidden messages in audio files using multiple detection algorithms and comprehensive feature analysis. This professional-grade application offers a sleek web interface, robust model training capabilities, and detailed analysis tools.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## 🌟 Features

### Detection Capabilities

- **Multi-Model Support**: Random Forest, Gradient Boosting, SVM, and CNN classifiers.
- **Comprehensive Feature Extraction**: Extracts over 60 audio features including:
  - **Statistical**: Mean, std, skewness, kurtosis, energy.
  - **Spectral**: Centroid, bandwidth, rolloff, contrast.
  - **MFCC**: Mel-frequency cepstral coefficients (20 coeffs + delta/delta-delta).
  - **LSB Analysis**: Least Significant Bit entropy and chi-square statistics.
  - **Temporal**: Onset strength, tempo.

### Professional Web Interface

- **Interactive UI**: Built with Streamlit for a smooth user experience.
- **Real-Time Analysis**: Instant steganography detection with confidence scores.
- **Visualizations**:
  - Waveform and Spectrogram displays.
  - MFCC Heatmaps.
  - LSB Distribution Histograms.
  - Feature Importance Charts.
- **Batch Processing**: Analyze multiple files at once and export results to CSV.

### Robust Model Performance

- **Hyperparameter Tuning**: Automated grid search with cross-validation.
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Reporting**: Generates detailed evaluation reports (JSON, CSV) and plots (Confusion Matrix, ROC Curve).

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kumarsaravana404/Stegnography.git
cd Stegnography
pip install -r requirements.txt
```

### 2. Implementation

You can run the automated setup script for Windows:

```bash
.\setup.bat
```

Or perform manual steps:

**Generate Dataset:**
Create synthetic audio samples for training (creates clean and stego files).

```bash
python src/generate_dataset.py
```

**Train Models:**
Train the machine learning models.

```bash
python src/model_train.py
```

### 3. Launch Application

Start the web interface:

```bash
python -m streamlit run app/streamlit_app.py
```

Open your browser to **http://localhost:8501** (or the port displayed in the terminal).

---

## 🎯 Usage

### Web Interface

1. **Upload**: Drag and drop audio files (.wav, .mp3, .flac, .ogg).
2. **Select Model**: Choose between Random Forest, SVM, or Gradient Boosting from the sidebar.
3. **Analyze**: View the detection result ("Stego" or "Clean") and confidence score.
4. **Visualize**: Explore the tabs to see waveforms, spectrograms, and feature distributions.

### Command Line

Detect steganography in a single file:

```bash
python src/predict.py path/to/audio.wav
```

### Python API

Use the detector in your own scripts:

```python
from src.model_train import SteganographyDetector

# Load model
detector = SteganographyDetector()
detector.load_model('models/random_forest_model.pkl')

# Predict
prediction, confidence = detector.predict('audio_file.wav')
print(f"Result: {'Stego' if prediction == 1 else 'Clean'} ({confidence*100:.2f}%)")
```

---

## 📂 Project Structure

```
Stegnography/
├── app/
│   └── streamlit_app.py           # Main web application
├── src/
│   ├── feature_extraction.py      # Core feature extraction logic
│   ├── model_train.py             # ML model training and management
│   ├── cnn_train.py               # Deep learning model training
│   ├── evaluate_models.py         # Model evaluation and reporting
│   ├── predict.py                 # Prediction utility
│   ├── generate_dataset.py        # Dataset generation script
│   └── stego_tools_integration.py # Image steganography tools
├── data/
│   ├── clean/                     # Directory for clean audio samples
│   └── stego/                     # Directory for stego audio samples
├── models/                        # Saved trained models
├── visualizations/                # Generated plots and charts
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Configuration

The system is highly configurable via `config.yaml` (if present) or internal constants. Key configurations include:

- **Model Parameters**: Adjust number of estimators, depth, learning rate.
- **Feature Settings**: `n_mfcc` (default 20), `n_fft` (2048), `hop_length` (512).
- **Web App**: Set port and upload limits.

---

## 🐳 Docker Support

Run the application in a containerized environment:

```bash
docker-compose up --build
```

Access the app at `http://localhost:8501`.

---

## 🐛 Troubleshooting & Known Issues

- **Lint Warnings**: You may see IDE warnings about implicit relative imports (e.g., in `train.py`). These are cosmetic and do not affect runtime functionality as the scripts handle path appending.
- **Model Not Found**: If the app complains about missing models, ensure you have run `python src/model_train.py` or `python src/train.py` first.
- **Dependencies**: If you encounter issues with `librosa` or `soundfile`, try installing `libsndfile` on your system or reinstalling the packages.

---

## 🔮 Future Enhancements

- **Real-Time Stream Analysis**: WebRTC integration for live audio analysis.
- **Advanced Deep Learning**: Transformer-based models for improved accuracy on complex audio.
- **Video Steganography**: Extending detection to video containers.
- **Mobile Support**: Optimized UI for mobile devices.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Kumar Saravana**

- GitHub: [@kumarsaravana404](https://github.com/kumarsaravana404)

---

**Made with ❤️ for cybersecurity and digital forensics**
