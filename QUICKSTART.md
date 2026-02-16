# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup

Run the automated setup script:

**Windows:**

```bash
setup.bat
```

**Linux/Mac:**

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Install all dependencies
- Create necessary directories
- Generate sample dataset
- Train all models
- Run a test prediction

### Step 2: Launch Web Application

```bash
streamlit run app/streamlit_app.py
```

Open your browser to http://localhost:8501

### Step 3: Start Detecting!

Upload an audio file and see the results instantly.

---

## 📖 Detailed Usage

### Training Custom Models

```bash
# Train all models
python src/model_train.py

# Or use Python API
from src.model_train import train_model

detector = train_model(
    data_clean_dir='data/clean',
    data_stego_dir='data/stego',
    model_type='random_forest',
    tune_hyperparams=True
)
```

### Command-Line Detection

```bash
# Detect steganography in a file
python src/predict.py path/to/audio.wav

# Use specific model
python src/predict.py audio.wav --model models/random_forest_model.pkl
```

### Batch Processing

Use the web interface's "Batch Analysis" tab or:

```python
from src.model_train import SteganographyDetector

detector = SteganographyDetector()
detector.load_model('models/random_forest_model.pkl')

files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
for file in files:
    prediction, confidence = detector.predict(file)
    print(f"{file}: {'Stego' if prediction == 1 else 'Clean'} ({confidence*100:.1f}%)")
```

---

## 🎯 Common Tasks

### Generate More Training Data

```bash
python src/generate_dataset.py
```

Follow the prompts to specify:

- Number of samples
- Audio complexity

### Visualize Model Performance

Models automatically generate visualizations in the `visualizations/` folder:

- Confusion matrices
- ROC curves
- Feature importance plots

### Extract Features Only

```python
from src.feature_extraction import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
features = extractor.extract_features('audio.wav')
print(f"Extracted {len(features)} features")
```

---

## 🔧 Troubleshooting

### "No module named 'librosa'"

Install dependencies:

```bash
pip install -r requirements.txt
```

### "No models found"

Train a model first:

```bash
python src/model_train.py
```

### Web app won't start

Make sure Streamlit is installed:

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

---

## 📊 Understanding Results

### Confidence Scores

- **90-100%**: Very high confidence
- **75-90%**: High confidence
- **50-75%**: Moderate confidence
- **Below 50%**: Low confidence (uncertain)

### Model Selection

- **Random Forest**: Best overall accuracy, fast
- **Gradient Boosting**: Good accuracy, slower
- **SVM**: Good for small datasets

---

## 🎓 Next Steps

1. **Collect Real Data**: Replace synthetic data with real audio samples
2. **Fine-tune Models**: Experiment with hyperparameters
3. **Add More Features**: Extend feature extraction
4. **Deploy**: Use the web interface for demonstrations

---

For more details, see the main [README.md](README.md)
