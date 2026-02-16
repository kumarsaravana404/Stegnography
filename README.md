# Audio Steganography Detection System

Advanced machine learning-based system for detecting hidden messages in audio files using multiple detection algorithms and comprehensive feature analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

### Detection Capabilities

- **Multi-Model Support**: Random Forest, Gradient Boosting, and SVM classifiers
- **Comprehensive Feature Extraction**: 60+ audio features including:
  - Statistical features (mean, std, skewness, kurtosis)
  - Spectral features (centroid, bandwidth, rolloff, contrast)
  - MFCC (Mel-frequency cepstral coefficients)
  - LSB analysis (entropy, chi-square test)
  - Temporal features (onset strength, tempo)

### Web Interface

- **Professional Streamlit UI** with:
  - Real-time audio file upload and analysis
  - Interactive visualizations (waveform, spectrogram, MFCC)
  - Batch processing capabilities
  - Model performance metrics display
  - LSB distribution analysis

### Model Performance

- **Hyperparameter Tuning**: Grid search with cross-validation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualization**: Confusion matrices, ROC curves, feature importance plots

## 📁 Project Structure

```
Stegnography/
├── app/
│   └── streamlit_app.py          # Web application
├── src/
│   ├── feature_extraction.py     # Feature extraction module
│   ├── model_train.py            # Model training with multiple algorithms
│   ├── generate_dataset.py       # Dataset generation
│   ├── stego_tools_integration.py # Image steganography tools
│   ├── features.py               # Legacy feature extraction
│   ├── train.py                  # Legacy training script
│   └── predict.py                # Legacy prediction script
├── data/
│   ├── clean/                    # Clean audio samples
│   └── stego/                    # Steganographic audio samples
├── models/                       # Trained models
├── visualizations/               # Generated plots and charts
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/kumarsaravana404/Stegnography.git
cd Stegnography
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### Generate Dataset

Create synthetic audio samples for training:

```bash
python src/generate_dataset.py
```

This creates 20 clean audio files and 20 corresponding steganographic files.

### Train Models

Train all available models:

```bash
python src/model_train.py
```

Or train a specific model:

```python
from src.model_train import train_model

detector = train_model(
    data_clean_dir='data/clean',
    data_stego_dir='data/stego',
    model_type='random_forest',  # or 'gradient_boosting', 'svm'
    tune_hyperparams=True,
    save_path='models/my_model.pkl'
)
```

### Run Web Application

Launch the Streamlit web interface:

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Command-Line Prediction

Predict a single file:

```bash
python src/predict.py path/to/audio/file.wav
```

## 🎯 Usage Examples

### Feature Extraction

```python
from src.feature_extraction import AudioFeatureExtractor

extractor = AudioFeatureExtractor(n_mfcc=20)
features = extractor.extract_features('audio_file.wav')
feature_names = extractor.get_feature_names()
```

### Model Training

```python
from src.model_train import SteganographyDetector

# Initialize detector
detector = SteganographyDetector(model_type='random_forest')

# Train
detector.train(X_train, y_train, tune_hyperparams=True)

# Evaluate
metrics = detector.evaluate(X_test, y_test)

# Save
detector.save_model('models/my_model.pkl')
```

### Prediction

```python
from src.model_train import SteganographyDetector

# Load model
detector = SteganographyDetector()
detector.load_model('models/random_forest_model.pkl')

# Predict
prediction, confidence = detector.predict('test_audio.wav')
print(f"Prediction: {'Stego' if prediction == 1 else 'Clean'}")
print(f"Confidence: {confidence*100:.2f}%")
```

### Image Steganography

```python
from src.stego_tools_integration import ImageSteganography

img_stego = ImageSteganography()

# Encode text in image
img_stego.encode_text('cover.png', 'Secret message', 'stego.png')

# Decode text from image
text = img_stego.decode_text('stego.png')

# Detect steganography
is_stego, confidence = img_stego.detect_steganography('stego.png')
```

## 📊 Model Performance

| Model             | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ----------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest     | 100%     | 100%      | 100%   | 100%     | 100%    |
| Gradient Boosting | ~98%     | ~98%      | ~98%   | ~98%     | ~99%    |
| SVM               | ~95%     | ~95%      | ~95%   | ~95%     | ~97%    |

_Note: Performance on synthetic dataset. Real-world performance may vary._

## 🔬 Technical Details

### Feature Extraction

The system extracts 60+ features from audio files:

1. **Statistical Features** (7): Mean, std, max, min, skewness, kurtosis, energy
2. **Spectral Features** (12): Centroid, bandwidth, rolloff, contrast, ZCR statistics
3. **MFCC Features** (40): 20 coefficients + their standard deviations
4. **LSB Features** (6): Mean, std, entropy, chi-square, difference statistics
5. **Temporal Features** (3): Onset strength, tempo

### Detection Algorithms

- **Random Forest**: Ensemble of 200 decision trees with max depth 20
- **Gradient Boosting**: Sequential ensemble with 100 estimators
- **SVM**: RBF kernel with probability estimates

### LSB Steganography Detection

The system specifically targets LSB (Least Significant Bit) steganography by:

- Analyzing LSB distribution patterns
- Computing entropy of LSB sequences
- Performing chi-square tests for randomness
- Detecting consecutive LSB pattern anomalies

## 🎨 Web Interface Features

### Detection Tab

- Upload audio files (WAV, MP3, FLAC, OGG)
- Real-time detection with confidence scores
- Interactive visualizations:
  - Waveform display
  - Spectrogram analysis
  - MFCC heatmap
  - LSB distribution histogram
  - Feature distribution charts

### Batch Analysis Tab

- Process multiple files simultaneously
- Export results as CSV
- Summary statistics

### About Tab

- System documentation
- Technical details
- References and resources

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, TensorFlow
- **Audio Processing**: librosa, soundfile
- **Web Framework**: Streamlit
- **Visualization**: matplotlib, seaborn, plotly
- **Image Processing**: Pillow, OpenCV
- **Data Science**: numpy, pandas, scipy

## 📝 API Reference

### AudioFeatureExtractor

```python
class AudioFeatureExtractor:
    def __init__(self, n_mfcc=20, n_fft=2048, hop_length=512)
    def extract_features(self, file_path: str) -> Optional[np.ndarray]
    def get_feature_names(self) -> list
```

### SteganographyDetector

```python
class SteganographyDetector:
    def __init__(self, model_type='random_forest')
    def train(self, X_train, y_train, hyperparams=None, tune_hyperparams=False)
    def evaluate(self, X_test, y_test) -> Dict[str, float]
    def predict(self, file_path: str) -> Tuple[int, float]
    def save_model(self, filepath: str, include_metadata=True)
    def load_model(self, filepath: str)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Kumar Saravana

- GitHub: [@kumarsaravana404](https://github.com/kumarsaravana404)

## 🙏 Acknowledgments

- Inspired by [Steganography-Tools](https://github.com/Priyansh-15/Steganography-Tools) by Priyansh Sharma
- Built with open-source libraries: librosa, scikit-learn, streamlit

## 📚 References

1. Librosa: Audio and Music Signal Analysis in Python
2. Scikit-learn: Machine Learning in Python
3. LSB Steganography Techniques
4. Audio Steganography Detection Methods

## 🔮 Future Enhancements

- [ ] Deep learning models (CNN, LSTM)
- [ ] Support for more steganography techniques
- [ ] Video steganography detection
- [ ] Real-time audio stream analysis
- [ ] Mobile application
- [ ] API endpoint for integration

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for cybersecurity and digital forensics**
