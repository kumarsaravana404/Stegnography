"""
Streamlit Web Application for Audio Steganography Detection
Professional UI with visualization and multi-model support
"""

import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import librosa.display
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_extraction import AudioFeatureExtractor
from model_train import SteganographyDetector

# Page configuration
st.set_page_config(
    page_title="Audio Steganography Detector",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
def load_models():
    """Load all available models"""
    import joblib

    models = {}
    model_dir = "models"

    if not os.path.exists(model_dir):
        return models

    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl") and not filename.endswith("_metadata.json"):
            model_name = filename.replace("_model.pkl", "").replace(".pkl", "")
            model_path = os.path.join(model_dir, filename)

            try:
                # Try loading as SteganographyDetector first
                detector = SteganographyDetector()
                detector.load_model(model_path)
                models[model_name] = detector
            except Exception as e1:
                try:
                    # Try loading as legacy/wrapped model
                    detector = joblib.load(model_path)
                    # Check if it has the required methods
                    if hasattr(detector, "predict") and hasattr(detector, "model"):
                        models[model_name] = detector
                    else:
                        st.warning(f"Model {filename} doesn't have required methods")
                except Exception as e2:
                    st.warning(f"Could not load model {filename}: {e1}, {e2}")

    return models


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


def plot_mfcc(y, sr):
    """Plot MFCC"""
    fig, ax = plt.subplots(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=ax)
    ax.set_title("MFCC")
    ax.set_ylabel("MFCC Coefficients")
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig


def plot_feature_distribution(features, feature_names):
    """Plot feature distribution"""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=feature_names[:20],  # Top 20 features
            y=features[:20],
            marker_color="lightblue",
        )
    )

    fig.update_layout(
        title="Feature Values (Top 20)",
        xaxis_title="Feature",
        yaxis_title="Value",
        height=400,
        xaxis_tickangle=-45,
    )

    return fig


def plot_lsb_analysis(y):
    """Analyze and plot LSB distribution"""
    # Convert to 16-bit integer
    if y.dtype != np.int16:
        y_int = (y * 32767).astype(np.int16)
    else:
        y_int = y

    # Extract LSB
    lsb = y_int & 1

    # Create histogram
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

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    # Load models
    models = load_models()

    if not models:
        st.error(
            "No models found! Please train a model first using `python src/model_train.py`"
        )
        st.stop()

    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Detection Model",
        options=list(models.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )

    selected_model = models[model_name]

    # Display model info
    st.sidebar.markdown("### 📊 Model Information")
    if selected_model.metrics:
        st.sidebar.metric(
            "Accuracy", f"{selected_model.metrics.get('accuracy', 0):.2%}"
        )
        st.sidebar.metric("F1 Score", f"{selected_model.metrics.get('f1', 0):.2%}")
        st.sidebar.metric("ROC AUC", f"{selected_model.metrics.get('roc_auc', 0):.2%}")

    # Visualization options
    st.sidebar.markdown("### 🎨 Visualization Options")
    show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)
    show_mfcc = st.sidebar.checkbox("Show MFCC", value=False)
    show_lsb = st.sidebar.checkbox("Show LSB Analysis", value=True)
    show_features = st.sidebar.checkbox("Show Feature Distribution", value=False)

    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📈 Batch Analysis", "ℹ️ About"])

    with tab1:
        st.markdown(
            '<div class="sub-header">Upload Audio File for Detection</div>',
            unsafe_allow_html=True,
        )

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
                    prediction, confidence = selected_model.predict(temp_path)

                # Display result
                st.markdown("---")
                st.markdown(
                    '<div class="sub-header">Detection Result</div>',
                    unsafe_allow_html=True,
                )

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
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90,
                            },
                        },
                    )
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Visualizations
                st.markdown("---")
                st.markdown(
                    '<div class="sub-header">Audio Analysis</div>',
                    unsafe_allow_html=True,
                )

                if show_waveform:
                    st.pyplot(plot_waveform(y, sr))

                if show_spectrogram:
                    st.pyplot(plot_spectrogram(y, sr))

                if show_mfcc:
                    st.pyplot(plot_mfcc(y, sr))

                if show_lsb:
                    st.plotly_chart(plot_lsb_analysis(y), use_container_width=True)

                if show_features:
                    extractor = AudioFeatureExtractor()
                    features = extractor.extract_features(temp_path)
                    if features is not None:
                        feature_names = extractor.get_feature_names()
                        st.plotly_chart(
                            plot_feature_distribution(features, feature_names),
                            use_container_width=True,
                        )

            except Exception as e:
                st.error(f"Error processing audio file: {e}")

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    with tab2:
        st.markdown(
            '<div class="sub-header">Batch Analysis</div>', unsafe_allow_html=True
        )

        st.info("Upload multiple audio files for batch processing")

        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=["wav", "mp3", "flac", "ogg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")

                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    prediction, confidence = selected_model.predict(temp_path)
                    results.append(
                        {
                            "Filename": file.name,
                            "Result": "Stego" if prediction == 1 else "Clean",
                            "Confidence": f"{confidence*100:.2f}%",
                        }
                    )
                except Exception as e:
                    results.append(
                        {"Filename": file.name, "Result": "Error", "Confidence": str(e)}
                    )
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Processing complete!")

            # Display results
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # Summary
            st.markdown("### Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(results))
            with col2:
                stego_count = sum(1 for r in results if r["Result"] == "Stego")
                st.metric("Stego Detected", stego_count)
            with col3:
                clean_count = sum(1 for r in results if r["Result"] == "Clean")
                st.metric("Clean Files", clean_count)

            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="steganography_detection_results.csv",
                mime="text/csv",
            )

    with tab3:
        st.markdown(
            '<div class="sub-header">About This Application</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        ### 🎯 Purpose
        This application uses advanced machine learning techniques to detect hidden messages 
        (steganography) in audio files. It analyzes various audio features including:
        
        - **Statistical Features**: Mean, standard deviation, skewness, kurtosis
        - **Spectral Features**: Spectral centroid, bandwidth, rolloff, contrast
        - **MFCC**: Mel-frequency cepstral coefficients
        - **LSB Analysis**: Least Significant Bit patterns and entropy
        - **Temporal Features**: Onset strength, tempo
        
        ### 🔬 Detection Methods
        The system supports multiple machine learning models:
        - **Random Forest**: Ensemble of decision trees
        - **Gradient Boosting**: Sequential ensemble method
        - **SVM**: Support Vector Machine with RBF kernel
        
        ### 📊 How It Works
        1. **Feature Extraction**: Extracts 60+ features from the audio file
        2. **Preprocessing**: Normalizes features using StandardScaler
        3. **Prediction**: Uses trained ML model to classify the audio
        4. **Visualization**: Provides detailed analysis and visualizations
        
        ### 🛠️ Technologies Used
        - **Streamlit**: Web application framework
        - **Librosa**: Audio analysis library
        - **Scikit-learn**: Machine learning models
        - **Plotly**: Interactive visualizations
        
        ### 👨‍💻 Developer
        Created as part of the Audio Steganography Detection Project
        
        ### 📝 License
        MIT License
        """
        )

        st.markdown("---")
        st.markdown("### 📚 References")
        st.markdown(
            """
        - [Librosa Documentation](https://librosa.org/)
        - [Scikit-learn Documentation](https://scikit-learn.org/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        """
        )


if __name__ == "__main__":
    main()
