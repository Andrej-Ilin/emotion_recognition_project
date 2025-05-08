import numpy as np
import librosa
import sys
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def load_label_encoder():
    # Восстановим энкодер из уже известных меток
    with open('data/features.pkl', 'rb') as f:
        _, y = pickle.load(f)
    le = LabelEncoder()
    le.fit(y)
    return le

def predict(file_path):
    model = load_model('models/audio_lstm.h5')
    le = load_label_encoder()
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # (1, 40, 174)
    prediction = model.predict(mfcc)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    print(f"Предсказанная эмоция: {predicted_label[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python modules/predict_emotion.py path/to/file.wav")
        sys.exit(1)

    predict(sys.argv[1])