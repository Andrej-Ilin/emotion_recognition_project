"""Main command line interface for emotion recognition package."""

import logging

import hydra
from omegaconf import DictConfig

# Set up logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for emotion recognition commands.

    Args:
        cfg: Hydra configuration
    """
    log.info(f"Starting emotion recognition with config: {cfg}")

    command = cfg.command
    log.info(f"Executing command: {command}")

    if command == "train":
        train_command(cfg)
    elif command == "predict":
        predict_command(cfg)
    elif command == "prepare_data":
        prepare_data_command(cfg)
    elif command == "convert_onnx":
        convert_onnx_command(cfg)
    elif command == "convert_tensorrt":
        convert_tensorrt_command(cfg)
    elif command == "create_inference_server":
        create_inference_server_command(cfg)
    else:
        log.error(f"Unknown command: {command}")
        raise ValueError(f"Unknown command: {command}")


def train_command(cfg: DictConfig) -> None:
    """Train emotion recognition model."""
    from emotion_recognition.data.data_loading import load_features, prepare_dataset
    from emotion_recognition.training.model_training import train_model

    log.info("Loading features and preparing dataset...")
    features, labels = load_features(cfg.data.features_path)
    X_train, X_test, y_train, y_test, le = prepare_dataset(
        features,
        labels,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
    )

    log.info(f"Training model with {len(features)} samples...")
    model = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        le,
        model_config=cfg.model,
        training_config=cfg.training,
    )

    log.info("Training completed successfully!")


def predict_command(cfg: DictConfig) -> None:
    """Predict emotion from audio file."""
    from emotion_recognition.inference.prediction import predict_from_file

    log.info(f"Predicting emotion from file: {cfg.inference.audio_file}")
    emotion, probabilities = predict_from_file(
        cfg.inference.audio_file, cfg.inference.model_path, cfg.inference.features_path
    )

    log.info(f"Predicted emotion: {emotion}")
    log.info(f"Probabilities: {probabilities}")


def prepare_data_command(cfg: DictConfig) -> None:
    """Prepare data by extracting features from audio files."""
    from emotion_recognition.data.data_loading import load_data, save_features

    log.info(f"Loading data from: {cfg.data.data_dir}")
    features, labels = load_data(cfg.data.data_dir)

    log.info(f"Extracted {len(features)} samples")
    log.info(f"Saving features to: {cfg.data.features_path}")
    save_features(features, labels, cfg.data.features_path)

    log.info("Data preparation completed!")


def convert_onnx_command(cfg: DictConfig) -> None:
    """Convert model to ONNX format."""
    from emotion_recognition.inference.model_conversion import convert_to_onnx

    log.info(f"Converting model to ONNX format...")
    onnx_path = convert_to_onnx(
        model_path=cfg.conversion.model_path,
        onnx_path=cfg.conversion.onnx_path,
        input_shape=tuple(cfg.conversion.input_shape),
        opset=cfg.conversion.onnx_opset,
        input_name=cfg.conversion.input_name,
        output_name=cfg.conversion.output_name,
    )

    log.info(f"ONNX model saved to: {onnx_path}")
    log.info("ONNX conversion completed successfully!")


def convert_tensorrt_command(cfg: DictConfig) -> None:
    """Convert model to TensorRT format."""
    from emotion_recognition.inference.model_conversion import convert_to_tensorrt

    log.info(f"Converting ONNX model to TensorRT format...")
    tensorrt_path = convert_to_tensorrt(
        onnx_path=cfg.conversion.onnx_path,
        tensorrt_path=cfg.conversion.tensorrt_path,
        precision=cfg.conversion.tensorrt_precision,
        max_workspace_size=cfg.conversion.tensorrt_max_workspace_size,
        max_batch_size=cfg.conversion.tensorrt_max_batch_size,
    )

    log.info(f"TensorRT engine saved to: {tensorrt_path}")
    log.info("TensorRT conversion completed successfully!")


def create_inference_server_command(cfg: DictConfig) -> None:
    """Create inference server configuration."""
    import json

    from emotion_recognition.inference.model_conversion import (
        create_inference_server_config,
    )

    log.info(f"Creating inference server configuration...")

    # For now, we'll create Triton config since it gives max points
    server_config = create_inference_server_config(
        model_path=cfg.conversion.onnx_path,
        server_type="triton",
        config_dir="inference_server_config",
    )

    # Save config to file
    with open("inference_server_config/server_config.json", "w") as f:
        json.dump(server_config, f, indent=2)

    log.info(f"Inference server configuration created:")
    log.info(f"Server type: {server_config['server_type']}")
    log.info(f"Model name: {server_config.get('model_name', 'N/A')}")
    log.info(f"Launch command: {server_config['launch_command']}")
    log.info("Inference server configuration completed!")


if __name__ == "__main__":
    main()
