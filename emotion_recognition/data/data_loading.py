"""Data loading and preprocessing functionality."""

import os
import pickle
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def load_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess audio data from RAVDESS dataset.

    Args:
        data_dir: Directory containing audio files

    Returns:
        Tuple of (features, labels)
    """
    from emotion_recognition.core.audio_processing import extract_mfcc

    features = []
    labels = []

    # RAVDESS emotion mapping
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    # Look for Actor directories directly in data directory
    audio_dir = data_dir
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found at {audio_dir}")

    # Load audio files and extract features
    for actor_dir in sorted(os.listdir(audio_dir)):
        actor_path = os.path.join(audio_dir, actor_dir)
        if not os.path.isdir(actor_path) or not actor_dir.startswith("Actor_"):
            continue

        for filename in sorted(os.listdir(actor_path)):
            if filename.endswith(".wav"):
                # Extract emotion from filename (format: 03-01-01-01-01-01-01.wav)
                parts = filename.split("-")
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, "unknown")

                if emotion != "unknown":
                    file_path = os.path.join(actor_path, filename)
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(emotion)

    return np.array(features), np.array(labels)


def prepare_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Prepare dataset for training.

    Args:
        features: Array of MFCC features
        labels: Array of emotion labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, label_encoder)
    """
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_onehot = to_categorical(y_encoded)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_onehot, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, le


def save_features(
    features: np.ndarray, labels: np.ndarray, output_path: str = "data/features.pkl"
) -> None:
    """Save extracted features and labels to file.

    Args:
        features: Array of MFCC features
        labels: Array of emotion labels
        output_path: Path to save features
    """
    with open(output_path, "wb") as f:
        pickle.dump((features, labels), f)


def load_features(
    input_path: str = "data/features.pkl",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels from file.

    Args:
        input_path: Path to features file

    Returns:
        Tuple of (features, labels)
    """
    with open(input_path, "rb") as f:
        features, labels = pickle.load(f)
    return features, labels
