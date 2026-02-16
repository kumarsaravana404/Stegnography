# Audio Steganography Detection Project

This project uses machine learning to detect if an audio file contains hidden messages (steganography). The detection is based on analyzing statistical anomalies in the Least Significant Bits (LSB) and other audio features.

## Project Structure

- `data/`: Contains the audio dataset.
  - `clean/`: Original audio files (negative class).
  - `stego/`: Audio files with hidden data (positive class).
- `src/`: Source code.
  - `generate_dataset.py`: Generates synthetic clean and stego audio files.
  - `features.py`: Extracts features (LSB stats, Spectral features, MFCC).
  - `train.py`: Trains a Random Forest Classifier.
  - `predict.py`: Predicts if a new file has steganography.
- `model.pkl`: The trained machine learning model.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Workflow

### 1. Generate Dataset (Optional)

If you don't have your own dataset, you can generate a synthetic one. This script creates clean sine waves and embeds random LSB noise to simulate steganography.

```bash
python src/generate_dataset.py
```

### 2. Train the Model

Train the Random Forest classifier on the data in `data/clean` and `data/stego`.

```bash
python src/train.py
```

This will save the trained model to `model.pkl`.

### 3. Predict on New Files

To check if an audio file contains hidden data:

```bash
python src/predict.py path/to/audio/file.wav
```

## Example

After generating data and training, try predicting on one of the generated samples:

```bash
python src/predict.py data/stego/sample_0.wav
# Output: Result: STEGANOGRAPHY DETECTED!
```
