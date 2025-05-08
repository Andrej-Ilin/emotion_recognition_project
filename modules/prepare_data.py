import os
import numpy as np
import librosa
import pickle

# Соответствие кода эмоции в названии файла → название эмоции
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    """Извлекает MFCC-признаки из аудиофайла и обрезает/дополняет до одинаковой длины"""
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def load_data(data_dir):
    X, y = [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                emotion_code = file.split("-")[2]
                if emotion_code in EMOTION_LABELS:
                    mfcc = extract_mfcc(filepath)
                    X.append(mfcc)
                    y.append(EMOTION_LABELS[emotion_code])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    data_path = "data/Audio_Speech_Actors_01-24"
    print("Извлекаем признаки из аудиофайлов...")
    X, y = load_data(data_path)
    with open("data/features.pkl", "wb") as f:
        pickle.dump((X, y), f)
    print(f"Готово: сохранено {len(X)} примеров в data/features.pkl")
