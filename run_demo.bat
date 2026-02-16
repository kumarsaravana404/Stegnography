@echo off
echo ===================================================
echo     Audio Steganography Detection Project Setup
echo ===================================================

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Generating synthetic dataset...
python src/generate_dataset.py

echo.
echo Training the model...
python src/train.py

echo.
echo Running a test prediction on a steganographic sample...
python src/predict.py data/stego/sample_0.wav

echo.
echo Running a test prediction on a clean sample...
python src/predict.py data/clean/sample_0.wav

echo.
echo ===================================================
echo               All steps completed!
echo ===================================================
pause
