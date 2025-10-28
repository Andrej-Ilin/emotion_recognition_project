# Emotion Recognition Project (Audio-based & Multimodal Extension)

![Roadmap](docs/roadmap_emotion_project.png)

## Overview
This project focuses on detecting emotions from speech using deep learning.  
It currently uses an LSTM model trained on the **RAVDESS** dataset, but is designed to evolve into a **multimodal emotion recognition system** combining **audio + facial expressions**.

---

## Features
- 🎙️ Audio recording via Streamlit interface  
- 🔊 MFCC-based feature extraction  
- 🧠 LSTM neural network trained on RAVDESS  
- ⚡ Real-time emotion prediction  
- 📊 Waveform and probability visualization  

---

## Limitations
- The base model was trained on **English** speech (RAVDESS).  
- For accurate multilingual recognition (e.g., Russian), retraining or fine-tuning on new datasets is required.  

---

## Installation

### With Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### With Docker

```bash
docker build -t emotion-recognition .
docker run -p 8501:8501 emotion-recognition
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Data Setup

1. Download RAVDESS: [Zenodo link](https://zenodo.org/record/1188976)
2. Extract into:

   ```
   data/Audio_Speech_Actors_01-24/
   ```
3. Run preprocessing:

   ```bash
   python modules/prepare_data.py
   ```

---

## Model Training

```bash
python modules/train_model.py
```

---

## Run Streamlit Dashboard

```bash
source venv/bin/activate
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Directory Structure

```
emotion_recognition_project/
├── data/                  # Dataset and extracted features
├── models/                # Trained models (e.g., audio_lstm.h5)
├── modules/               # Core ML modules (training, inference, preprocessing)
├── dashboard/             # Streamlit UI
├── docs/                  # Documentation, roadmap images, etc.
├── requirements.txt
└── README.md
```

---

## 🧩 Project Roadmap

| Phase                               | Weeks | Goals                                                         | Deliverables                     |
| :---------------------------------- | :---- | :------------------------------------------------------------ | :------------------------------- |
| **1. Environment Setup**            | 1–2   | Clean repo, configure venv/Docker, test scripts               | Working reproducible environment |
| **2. Audio Model Upgrade**          | 3–6   | Integrate Wav2Vec2/HuBERT embeddings, compare models          | Improved accuracy, charts        |
| **3. Multimodal Extension**         | 7–10  | Add facial emotion detection (CNN + OpenCV)                   | Combined audio-visual model      |
| **4. Deployment & MCP Integration** | 11–12 | Docker + Streamlit server deployment, remote training via MCP | Hosted demo or container         |
| **5. Experimentation & Analysis**   | 13–14 | Run comparative experiments, visual analysis                  | Results, plots, logs             |
| **6. Documentation & Defense Prep** | 15–16 | Finalize README, presentation, report                         | Full project ready for defense   |

---

## What's Next

* 🗣️ Fine-tune with Russian emotion datasets
* 👁️ Add face-based emotion detection
* 🔗 Fuse audio & visual features
* ☁️ Deploy via Docker or Hugging Face Spaces
* 📊 Publish comparison results for your MSc thesis

---

## License

MIT License
© 2025 Andrey Ilin

```
