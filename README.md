# Emotion Recognition Project (Audio-based)

## Overview
This project is focused on detecting emotions from speech using deep learning. It uses a Long Short-Term Memory (LSTM) neural network trained on the RAVDESS dataset to classify emotions from audio recordings. The project supports both uploading .wav files and recording audio directly through a Streamlit web interface.

## Features
- Audio recording via browser using Streamlit
- MFCC-based feature extraction for sequential audio data
- LSTM model trained on the RAVDESS dataset
- Real-time emotion prediction
- Visual waveform and probability distribution

## Limitations
- The model was trained on the RAVDESS dataset, which contains English speech.
- Predictions may be inaccurate for non-English (e.g. Russian) speech due to different phonetic and prosodic characteristics.
- Further training or fine-tuning with Russian-language emotion datasets is recommended for improved multilingual accuracy.

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Setup
1. Download the RAVDESS dataset from Zenodo: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
2. Extract it into `data/Audio_Speech_Actors_01-24/`
3. Run the feature extraction:
   ```bash
   python modules/prepare_data.py
   ```

## Model Training
```bash
python modules/train_model.py
```

## Run Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

## Directory Structure

```
emotion_recognition_project/
├── data/ # Dataset and extracted features
├── models/ # Trained models (e.g. audio_lstm.h5)
├── modules/ # Core modules (training, prediction, feature extraction)
├── dashboard/ # Streamlit app
├── requirements.txt # Project dependencies
└── README.md
```

## What's Next
- Fine-tune or retrain model with Russian-language emotional speech datasets
- Add face-based emotion detection module (via OpenCV + CNN)
- Fuse audio and visual predictions for multi-modal emotion recognition
- Deploy full system via Docker or cloud (e.g., Hugging Face Spaces or Streamlit Cloud)

---

This project is a great foundation for building a multilingual, multimodal emotion recognition system.
