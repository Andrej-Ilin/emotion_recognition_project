"""Emotion Recognition Package

This package provides audio-based emotion recognition using deep learning.
"""

from .core.audio_processing import extract_mfcc
from .data.data_loading import load_data, prepare_dataset
from .inference.prediction import predict_emotion
from .training.model_training import train_model

__version__ = "0.1.0"
__all__ = [
    "extract_mfcc",
    "load_data",
    "prepare_dataset",
    "train_model",
    "predict_emotion",
]
