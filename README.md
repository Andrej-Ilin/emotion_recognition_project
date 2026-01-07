# Emotion Recognition System

**Audio-based Emotion Recognition using Deep Learning**

This project implements an industrial-grade MLOps pipeline for audio-based emotion recognition using LSTM neural networks. The system is designed for production deployment with comprehensive data management, experiment tracking, model conversion, and inference serving capabilities.

## 🎯 Project Overview

### Problem Statement
Emotion recognition from audio is a challenging task in affective computing with applications in mental health monitoring, customer service analysis, and human-computer interaction. This project provides a robust pipeline for training, deploying, and serving emotion recognition models.

### Solution
- **LSTM-based neural network** trained on the RAVDESS dataset
- **End-to-end MLOps pipeline** with data versioning, experiment tracking, and model serving
- **Production-ready inference** with ONNX and TensorRT optimization
- **Comprehensive monitoring** and logging for model performance

### Key Features
- ✅ Audio feature extraction using MFCC
- ✅ LSTM model for sequential emotion classification
- ✅ Hydra configuration management
- ✅ DVC data versioning and pipeline orchestration
- ✅ MLflow experiment tracking and logging
- ✅ ONNX and TensorRT model conversion
- ✅ Triton Inference Server integration
- ✅ Streamlit web interface for real-time predictions

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (required for numpy 2.1.3 compatibility)
- Poetry (for dependency management)
- DVC (for data versioning)
- MLflow (for experiment tracking)
- Docker (for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/Andrej-Ilin/emotion_recognition_project.git
cd emotion_recognition_project

# Install Poetry if not already installed
sudo apt install python3-poetry

# Install project dependencies using Poetry
poetry install

# Install pre-commit hooks
pre-commit install

# Initialize DVC
dvc init
```

### Important Notes

**Python Version Requirement**: This project requires Python 3.10+ due to numpy 2.1.3 compatibility. The `pyproject.toml` has been updated to reflect this requirement.

**MLflow Tracking**: The training code has been updated to properly configure MLflow tracking URI. No additional setup is needed.

## 📦 Project Structure

```
emotion_recognition_project/
├── emotion_recognition/          # Main Python package
│   ├── core/                     # Core audio processing
│   ├── data/                     # Data loading and preprocessing
│   ├── training/                 # Model training
│   ├── inference/                # Prediction and model conversion
│   ├── configs/                  # Configuration management
│   └── commands.py               # CLI entry point
├── dashboard/                    # Streamlit web interface
├── configs/                      # Hydra configuration files
├── data/                         # Dataset and features (DVC tracked)
├── models/                       # Trained models
├── inference_server_config/      # Inference server configuration
├── .dvc/                         # DVC metadata
├── .gitignore                    # Git ignore rules
├── .pre-commit-config.yaml       # Pre-commit hooks
├── dvc.yaml                      # DVC pipeline definition
├── pyproject.toml                # Poetry project configuration
├── README.md                     # Project documentation
└── requirements.txt              # Legacy requirements
```

## 🎛️ Configuration

The project uses **Hydra** for configuration management. All parameters are defined in YAML files under `configs/`.

### Configuration Files

- `config.yaml` - Main configuration
- `data/default.yaml` - Data loading parameters
- `model/default.yaml` - Model architecture
- `training/default.yaml` - Training parameters
- `inference/default.yaml` - Inference settings
- `conversion/default.yaml` - Model conversion settings
- `mlflow/default.yaml` - MLflow tracking configuration

### Overriding Configuration

```bash
# Override specific parameters
python emotion_recognition/commands.py train model.lstm_units=256 training.epochs=50

# Use different configuration files
python emotion_recognition/commands.py train --config-name=custom
```

## 🔧 Data Pipeline

### Data Preparation

```bash
# Prepare data (extract features from audio)
python emotion_recognition/commands.py prepare_data

# Or use DVC pipeline
dvc repro prepare_data
```

### Dataset Information

- **Source**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Emotions**: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Format**: 24 actors, 2 repetitions, 8 emotions = 384 audio files
- **Features**: 40 MFCC coefficients, 174 timesteps

## 🏋️ Model Training

### Training Process

```bash
# Train the model
python emotion_recognition/commands.py train

# Or use DVC pipeline
dvc repro train
```

### Training Configuration

- **Model Architecture**: LSTM with dropout and dense layers
- **Optimizer**: Adam with configurable learning rate
- **Loss Function**: Categorical cross-entropy
- **Metrics**: Accuracy
- **Callbacks**: Early stopping, model checkpointing

### MLflow Integration

The training process automatically logs to MLflow:

- Experiment parameters and metrics
- Model artifacts
- Training curves
- Git commit information

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# View experiments at http://localhost:5000
```

## 📊 Model Evaluation

Evaluation metrics are automatically logged to MLflow during training:

- Training and validation accuracy
- Training and validation loss
- Confusion matrix
- Classification report

## 🔄 Model Conversion

### ONNX Conversion

```bash
python emotion_recognition/commands.py convert_onnx
```

### TensorRT Conversion

```bash
python emotion_recognition/commands.py convert_tensorrt
```

### Conversion Benefits

- **ONNX**: Cross-platform model format for interoperability
- **TensorRT**: NVIDIA optimization for maximum inference performance
- **Quantization**: Support for FP32, FP16, and INT8 precision

## 🚀 Inference Server

### Triton Inference Server Setup

```bash
# Create inference server configuration
python emotion_recognition/commands.py create_inference_server

# Launch Triton server (requires Docker)
docker run --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/inference_server_config:/models nvcr.io/nvidia/tritonserver:24.04-py3 \
  trtserver --model-store=/models
```

### MLflow Serving

```bash
mlflow models serve -m models/audio_lstm.onnx --port 5001
```

## 🎤 Real-time Prediction

### Streamlit Dashboard

```bash
# Launch the web interface
streamlit run dashboard/app.py

# Access at http://localhost:8501
```

### Programmatic Prediction

```python
from emotion_recognition.inference.prediction import predict_from_file

# Predict emotion from audio file
emotion, probabilities = predict_from_file("sample_audio.wav")
print(f"Predicted emotion: {emotion}")
print(f"Probabilities: {probabilities}")
```

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t emotion-recognition .

# Run container
docker run -p 8501:8501 emotion-recognition
```

## 📈 Experiment Tracking

### MLflow UI

Access the MLflow tracking UI at `http://localhost:5000` to:

- Compare experiment runs
- View training metrics
- Analyze model performance
- Track hyperparameter changes

### Logging Information

- **Parameters**: Model architecture, training settings
- **Metrics**: Accuracy, loss, training time
- **Artifacts**: Model files, training plots
- **Code Version**: Git commit hash

## 🔧 Development Workflow

### Code Quality

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Format code
poetry run black .
poetry run isort .

# Lint code
poetry run flake8 .
```

### Testing

```bash
# Run tests (to be implemented)
pytest
```

### CI/CD Pipeline

The project includes:

- **Pre-commit hooks**: Black, isort, flake8, prettier
- **DVC pipelines**: Data processing, training, conversion
- **MLflow tracking**: Experiment management
- **Docker support**: Containerized deployment

## 📋 Roadmap

### Completed Features
- ✅ MLOps pipeline restructuring
- ✅ Hydra configuration management
- ✅ DVC data versioning
- ✅ MLflow experiment tracking
- ✅ ONNX model conversion
- ✅ TensorRT optimization
- ✅ Triton Inference Server integration
- ✅ Streamlit web interface

### Future Enhancements
- 🔄 Multi-modal emotion recognition (audio + video)
- 🌐 Cloud deployment (AWS/GCP)
- 📊 Advanced monitoring and alerting
- 🤖 Automated model retraining
- 🎯 Fine-tuning with domain-specific datasets

## 📚 Documentation

### API Reference

See the [Python package documentation](emotion_recognition/) for detailed API reference.

### Configuration Guide

See [configs/](configs/) for configuration examples and parameter descriptions.

### Deployment Guide

See [Dockerfile](Dockerfile) and [inference_server_config/](inference_server_config/) for deployment instructions.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Run pre-commit hooks
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guide
- Use type hints
- Write docstrings
- Include tests
- Update documentation

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- RAVDESS dataset providers
- MLOps community for best practices
- Open-source contributors

## 📞 Support

For issues, questions, or suggestions:

- Open a GitHub issue
- Contact the maintainer

## 🔧 Troubleshooting

### Common Issues and Solutions

**MLflow Server Not Starting**:
- **Issue**: Connection refused when accessing `http://127.0.0.1:5000`
- **Solution**: Make sure the MLflow server is running:
  ```bash
  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
  ```

**Poetry Installation Issues**:
- **Issue**: Dependency conflicts or installation failures
- **Solution**: Update Python requirement in `pyproject.toml` to `>=3.10,<4.0` and run:
  ```bash
  poetry install
  ```

**Hydra Command Syntax**:
- **Issue**: Commands like `python emotion_recognition/commands.py train` don't work
- **Solution**: Use the correct Hydra syntax:
  ```bash
  python emotion_recognition/commands.py command=train
  ```

**Data Loading Errors**:
- **Issue**: `FileNotFoundError: Audio directory not found`
- **Solution**: Ensure data is in the correct format (Actor_01, Actor_02, etc. directories)

### Recent Fixes and Updates

1. **Python Version**: Updated from 3.8+ to 3.10+ for numpy compatibility
2. **MLflow Tracking**: Added `mlflow.set_tracking_uri()` in training code
3. **Data Loading**: Fixed path handling for RAVDESS dataset structure
4. **ONNX Conversion**: Updated to use tf2onnx for TensorFlow 2.19.0

---

**Project Status**: 🚀 Production-ready MLOps pipeline for audio emotion recognition
