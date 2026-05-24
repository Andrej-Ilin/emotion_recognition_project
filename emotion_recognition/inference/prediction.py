"""Inference and prediction functionality."""

import os
import pickle
from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def load_audio_model(model_path: str = "models/audio_lstm.h5"):
    """Load trained audio emotion recognition model.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded Keras model
    """
    return load_model(model_path)


def load_label_encoder(features_path: str = "data/features.pkl"):
    """Load label encoder from features file.

    Args:
        features_path: Path to features file containing labels

    Returns:
        Fitted LabelEncoder
    """
    with open(features_path, "rb") as f:
        _, y = pickle.load(f)
    le = LabelEncoder()
    le.fit(y)
    return le


def predict_emotion(
    mfcc_features: np.ndarray, model, label_encoder: LabelEncoder
) -> Tuple[str, np.ndarray]:
    """Predict emotion from MFCC features.

    Args:
        mfcc_features: MFCC features array
        model: Trained Keras model
        label_encoder: Label encoder for emotion classes

    Returns:
        Tuple of (predicted_emotion, probabilities)
    """
    # Add batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=0)

    # Get predictions
    probs = model.predict(mfcc_features)[0]
    predicted_idx = np.argmax(probs)
    predicted_emotion = label_encoder.inverse_transform([predicted_idx])[0]

    return predicted_emotion, probs


def predict_from_file(
    audio_file_path: str,
    model_path: str = "models/audio_lstm.h5",
    features_path: str = "data/features.pkl",
) -> Tuple[str, np.ndarray]:
    """Predict emotion directly from audio file.

    Args:
        audio_file_path: Path to audio file
        model_path: Path to saved model
        features_path: Path to features file for label encoder

    Returns:
        Tuple of (predicted_emotion, probabilities)
    """
    from emotion_recognition.core.audio_processing import extract_mfcc

    # Extract features
    mfcc = extract_mfcc(audio_file_path)

    # Load model and encoder
    model = load_audio_model(model_path)
    encoder = load_label_encoder(features_path)

    # Make prediction
    return predict_emotion(mfcc, model, encoder)
