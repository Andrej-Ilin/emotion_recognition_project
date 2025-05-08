import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pickle
import soundfile as sf
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from audio_recorder_streamlit import audio_recorder

# ===== Функции =====
@st.cache_resource
def load_audio_model():
    return load_model("models/audio_lstm.h5")

@st.cache_resource
def load_encoder():
    with open("data/features.pkl", "rb") as f:
        _, y = pickle.load(f)
    le = LabelEncoder()
    le.fit(y)
    return le

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc, y, sr

def predict_emotion(mfcc, model, encoder):
    mfcc = np.expand_dims(mfcc, axis=0)
    probs = model.predict(mfcc)[0]
    predicted = encoder.inverse_transform([np.argmax(probs)])
    return predicted[0], probs

# ===== Streamlit UI =====
st.title("🎤 Распознавание эмоций по аудио")

st.subheader("1. Запишите аудио")
audio_bytes = audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Обработка
    mfcc, y, sr = extract_mfcc(tmp_path)
    model = load_audio_model()
    encoder = load_encoder()
    pred, probs = predict_emotion(mfcc, model, encoder)

    # Вывод результата
    st.subheader(f"2. Эмоция: **{pred}**")

    st.subheader("3. Волновая форма")
    if len(y) > 0:
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Аудио слишком короткое или пустое для отображения формы сигнала.")
    st.subheader("4. Распределение вероятностей")
    st.bar_chart(data=dict(zip(encoder.classes_, probs)))