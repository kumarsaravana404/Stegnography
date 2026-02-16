# 🚀 Quick Start - Get Running in 3 Steps!

## ⚡ Fast Track (Recommended)

### Step 1: Install Core Dependencies

```bash
pip install streamlit numpy pandas scikit-learn librosa soundfile matplotlib seaborn plotly joblib scipy Pillow pyyaml
```

### Step 2: Generate Sample Data & Train Model

```bash
# Generate dataset (creates 20 clean + 20 stego audio files)
python src/generate_dataset.py

# Train the Random Forest model
python src/model_train.py
```

### Step 3: Launch Web App

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to **http://localhost:8501** 🎉

---

## 🎯 What You Can Do

### Web Interface

- Upload audio files (.wav, .mp3, .flac, .ogg)
- Select detection model (Random Forest, SVM, Gradient Boosting)
- View real-time detection results
- See comprehensive visualizations:
  - Waveform
  - Spectrogram
  - MFCC heatmap
  - LSB distribution
  - Feature charts
- Batch process multiple files
- Export results as CSV

### Command Line

```bash
# Detect steganography in a file
python src/predict.py path/to/audio.wav

# Use specific model
python src/predict.py audio.wav --model models/random_forest_model.pkl
```

### Python API

```python
from src.model_train import SteganographyDetector

# Load model
detector = SteganographyDetector()
detector.load_model('models/random_forest_model.pkl')

# Predict
prediction, confidence = detector.predict('audio.wav')
print(f"Result: {'Stego' if prediction == 1 else 'Clean'}")
print(f"Confidence: {confidence*100:.2f}%")
```

---

## 📊 Available Models

| Model             | Accuracy | Speed  | Best For       |
| ----------------- | -------- | ------ | -------------- |
| Random Forest     | 100%\*   | Fast   | General use    |
| Gradient Boosting | ~98%\*   | Medium | High accuracy  |
| SVM               | ~95%\*   | Slow   | Small datasets |
| CNN               | ~97%\*   | Slow   | Deep learning  |

\*Performance on synthetic dataset

---

## 🛠️ Troubleshooting

### "streamlit not found"

```bash
pip install streamlit
```

### "No module named 'librosa'"

```bash
pip install librosa soundfile
```

### "No models found"

```bash
# Train a model first
python src/model_train.py
```

### "No data found"

```bash
# Generate dataset first
python src/generate_dataset.py
```

---

## 📁 Project Structure

```
Stegnography/
├── app/
│   └── streamlit_app.py          # Web application
├── src/
│   ├── feature_extraction.py     # 60+ audio features
│   ├── model_train.py            # ML model training
│   ├── cnn_train.py              # Deep learning
│   ├── evaluate_models.py        # Model evaluation
│   ├── predict.py                # Predictions
│   └── generate_dataset.py       # Dataset generation
├── models/                       # Trained models (auto-created)
├── data/
│   ├── clean/                    # Clean audio samples
│   └── stego/                    # Steganographic samples
└── visualizations/               # Auto-generated plots
```

---

## 🎓 Advanced Usage

### Train All Models

```bash
python src/model_train.py
```

### Train CNN Model

```bash
python src/cnn_train.py
```

### Evaluate All Models

```bash
python src/evaluate_models.py
```

### Generate More Data

```bash
python src/generate_dataset.py
# Follow prompts to specify number of samples
```

---

## 🐳 Docker (Alternative)

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8501
```

---

## 📚 Full Documentation

- **README.md** - Complete project documentation
- **IMPLEMENTATION_GUIDE.md** - Detailed implementation guide
- **ENHANCEMENTS.md** - What's new
- **config.yaml** - Configuration options

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Generate dataset
3. ✅ Train model
4. ✅ Launch web app
5. 🎉 Start detecting steganography!

---

## 💡 Tips

- **First time?** Use the web interface - it's the easiest way to get started
- **Need speed?** Use Random Forest model (fastest, most accurate)
- **Want deep learning?** Train the CNN model (takes longer but impressive)
- **Batch processing?** Use the "Batch Analysis" tab in the web app

---

## 📞 Need Help?

- Check **IMPLEMENTATION_GUIDE.md** for detailed instructions
- See **README.md** for API reference
- Open an issue on GitHub

---

**Made with ❤️ for cybersecurity and digital forensics**

**Repository**: https://github.com/kumarsaravana404/Stegnography
