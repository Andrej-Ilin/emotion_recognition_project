import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Загрузка признаков и меток
with open('data/features.pkl', 'rb') as f:
    X, y = pickle.load(f)

print(f"Загружено {len(X)} примеров, форма признаков: {X.shape}")

# Подготовка меток
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Построение LSTM-модели
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_onehot.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Создание папки models/
os.makedirs('models', exist_ok=True)

# Обучение модели с сохранением лучшей
checkpoint = ModelCheckpoint('models/audio_lstm.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])