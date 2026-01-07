"""Audio processing and feature extraction functionality."""

import librosa
import numpy as np


def extract_mfcc(
    file_path: str, n_mfcc: int = 40, max_len: int = 174, sample_rate: int = 22050
) -> np.ndarray:
    """Extract MFCC features from audio file.

    Args:
        file_path: Path to audio file
        n_mfcc: Number of MFCC coefficients to extract
        max_len: Maximum length of MFCC sequence (pad or truncate)
        sample_rate: Sample rate for audio loading

    Returns:
        MFCC features as numpy array
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc
