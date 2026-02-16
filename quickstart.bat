@echo off
echo ================================================================
echo     Quick Start - Audio Steganography Detection
echo ================================================================
echo.

echo Installing core dependencies (this may take a few minutes)...
pip install streamlit numpy pandas scikit-learn librosa soundfile matplotlib seaborn plotly joblib scipy Pillow pyyaml

echo.
echo Creating directories...
if not exist "models" mkdir models
if not exist "visualizations" mkdir visualizations
if not exist "data\clean" mkdir data\clean
if not exist "data\stego" mkdir data\stego

echo.
echo Generating sample dataset...
python src/generate_dataset.py

echo.
echo Training Random Forest model...
python src/model_train.py

echo.
echo ================================================================
echo                    Setup Complete!
echo ================================================================
echo.
echo To launch the web application, run:
echo     streamlit run app/streamlit_app.py
echo.
echo Or press any key to launch it now...
pause

streamlit run app/streamlit_app.py
