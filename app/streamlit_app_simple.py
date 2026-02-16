"""
Simple Streamlit App for Legacy Model
Works with the trained model.pkl from train.py
"""

import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import librosa
import librosa.display
import joblib
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import extract_features

# Page configuration
st.set_page_config(
    page_title="Audio Steganography Detector",
    page_icon="🔊",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stego-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .clean-detected {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = "models/random_forest_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def predict_audio(model, file_path):
    """Predict if audio contains steganography"""
    features = extract_features(file_path)
    if features is None:
        raise ValueError("Could not extract features from audio")

    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction]

    return prediction, confidence


def plot_waveform(y, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig


def plot_spectrogram(y, sr):
    """Plot spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_lsb_analysis(y):
    """Analyze and plot LSB distribution"""
    if y.dtype != np.int16:
        y_int = (y * 32767).astype(np.int16)
    else:
        y_int = y

    lsb = y_int & 1

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=lsb, nbinsx=2, name="LSB Distribution", marker_color="indianred")
    )
    fig.update_layout(
        title="LSB Distribution Analysis",
        xaxis_title="LSB Value",
        yaxis_title="Frequency",
        height=400,
    )
    return fig


def main():
    # Header
    st.markdown(
        '<div class="main-header">🔊 Audio Steganography Detection System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>
            Advanced ML-based detection of hidden messages in audio files
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load model
    model = load_model()

    if model is None:
        st.error(
            "❌ No model found! Please train a model first using `python src/train.py`"
        )
        st.info("Run this command in your terminal: `python src/train.py`")
        st.stop()

    st.success("✅ Model loaded successfully!")

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("### 🎨 Visualization Options")
    show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)
    show_lsb = st.sidebar.checkbox("Show LSB Analysis", value=True)

    # Main content
    tab1, tab2 = st.tabs(["🔍 Detection", "ℹ️ About"])

    with tab1:
        st.markdown("### Upload Audio File for Detection")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg"],
            help="Upload an audio file to detect steganography",
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Load audio
                y, sr = librosa.load(temp_path, sr=None)

                # Display audio player
                st.audio(uploaded_file, format="audio/wav")

                # Audio info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duration", f"{len(y)/sr:.2f}s")
                with col2:
                    st.metric("Sample Rate", f"{sr} Hz")
                with col3:
                    st.metric("Samples", f"{len(y):,}")
                with col4:
                    st.metric("Channels", "Mono")

                # Perform detection
                with st.spinner("Analyzing audio file..."):
                    prediction, confidence = predict_audio(model, temp_path)

                # Display result
                st.markdown("---")
                st.markdown("### Detection Result")

                if prediction == 1:
                    st.markdown(
                        f"""
                    <div class="stego-detected">
                        <h2>⚠️ STEGANOGRAPHY DETECTED</h2>
                        <p style='font-size: 1.2rem;'>
                            This audio file likely contains hidden data.
                        </p>
                        <p style='font-size: 1.5rem; font-weight: bold;'>
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="clean-detected">
                        <h2>✅ CLEAN AUDIO</h2>
                        <p style='font-size: 1.2rem;'>
                            No steganography detected in this file.
                        </p>
                        <p style='font-size: 1.5rem; font-weight: bold;'>
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Confidence gauge
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Detection Confidence"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {
                                "color": "darkred" if prediction == 1 else "darkgreen"
                            },
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 75], "color": "gray"},
                                {"range": [75, 100], "color": "darkgray"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Visualizations
                st.markdown("---")
                st.markdown("### Audio Analysis")

                if show_waveform:
                    st.pyplot(plot_waveform(y, sr))

                if show_spectrogram:
                    st.pyplot(plot_spectrogram(y, sr))

                if show_lsb:
                    st.plotly_chart(plot_lsb_analysis(y), use_container_width=True)

            except Exception as e:
                st.error(f"Error processing audio file: {e}")

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    with tab2:
        st.markdown("### About This Application")

        st.markdown(
            """
        ### 🎯 Purpose
        This application uses machine learning to detect hidden messages (steganography) 
        in audio files. It analyzes various audio features including:
        
        - **Statistical Features**: Mean, standard deviation, max, min
        - **Spectral Features**: Spectral centroid, bandwidth, rolloff
        - **MFCC**: Mel-frequency cepstral coefficients
        - **LSB Analysis**: Least Significant Bit patterns
        - **Zero-Crossing Rate**: Signal crossing analysis
        
        ### 📊 How It Works
        1. **Feature Extraction**: Extracts 19 features from the audio file
        2. **Prediction**: Uses Random Forest classifier to detect steganography
        3. **Visualization**: Provides detailed analysis and visualizations
        
        ### 🛠️ Technologies Used
        - **Streamlit**: Web application framework
        - **Librosa**: Audio analysis library
        - **Scikit-learn**: Machine learning models
        - **Plotly**: Interactive visualizations
        
        ### 📝 Model Information
        - **Type**: Random Forest Classifier
        - **Features**: 19 audio features
        - **Training**: Trained on clean and steganographic audio samples
        """
        )


if __name__ == "__main__":
    main()
