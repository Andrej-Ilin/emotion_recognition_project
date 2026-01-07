import tempfile

import matplotlib.pyplot as plt
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from emotion_recognition.core.audio_processing import extract_mfcc
from emotion_recognition.inference.prediction import (
    load_audio_model,
    load_label_encoder,
    predict_emotion,
)

# ===== Functions =====


@st.cache_resource
def load_audio_model_cached():
    return load_audio_model()


@st.cache_resource
def load_encoder_cached():
    return load_label_encoder()


def process_audio(audio_bytes: bytes) -> tuple:
    """Process audio bytes and return prediction results."""
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Extract features
    mfcc = extract_mfcc(tmp_path)

    # Load model and encoder
    model = load_audio_model_cached()
    encoder = load_encoder_cached()

    # Make prediction
    pred, probs = predict_emotion(mfcc, model, encoder)

    # Load audio for visualization
    import librosa

    y, sr = librosa.load(tmp_path, sr=22050)

    return pred, probs, y, sr


# ===== Streamlit UI =====
st.title("🎤 Emotion Recognition from Audio")

st.subheader("1. Record Audio")
audio_bytes = audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Process audio
    pred, probs, y, sr = process_audio(audio_bytes)

    # Display results
    st.subheader(f"2. Emotion: **{pred}**")

    st.subheader("3. Waveform")
    if len(y) > 0:
        fig, ax = plt.subplots()
        import librosa.display

        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Audio is too short or empty to display waveform.")

    st.subheader("4. Probability Distribution")
    encoder = load_encoder_cached()
    st.bar_chart(data=dict(zip(encoder.classes_, probs)))
