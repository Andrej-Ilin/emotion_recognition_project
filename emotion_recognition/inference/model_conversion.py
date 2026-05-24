"""Model conversion functionality for ONNX and TensorRT."""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import onnx
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

log = logging.getLogger(__name__)


def convert_to_onnx(
    model_path: str,
    onnx_path: str,
    input_shape: Tuple[int, int, int] = (1, 40, 174),
    opset: int = 13,
    input_name: str = "audio_input",
    output_name: str = "emotion_output",
) -> str:
    """Convert Keras model to ONNX format.

    Args:
        model_path: Path to Keras model
        onnx_path: Path to save ONNX model
        input_shape: Input shape for the model
        opset: ONNX opset version
        input_name: Name of input tensor
        output_name: Name of output tensor

    Returns:
        Path to saved ONNX model
    """
    log.info(f"Loading model from {model_path}")
    model = load_model(model_path)

    # Convert Sequential model to Functional model if needed
    if isinstance(model, tf.keras.Sequential):
        log.info(
            "Converting Sequential model to Functional model for ONNX compatibility"
        )
        input_layer = tf.keras.Input(shape=input_shape[1:], name=input_name)
        output_layer = model(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    log.info("Converting to ONNX format...")
    # Use tf2onnx for conversion
    import tf2onnx

    # Prepare input signature
    input_signature = [
        tf.TensorSpec(
            shape=(None,) + input_shape[1:], dtype=tf.float32, name=input_name
        )
    ]

    # Convert model to ONNX
    output_path = tf2onnx.convert.from_keras(
        model, input_signature=input_signature, opset=opset, output_path=onnx_path
    )

    log.info(f"ONNX model saved to {output_path}")
    return output_path


def convert_to_tensorrt(
    onnx_path: str,
    tensorrt_path: str,
    precision: str = "FP32",
    max_workspace_size: int = 1073741824,
    max_batch_size: int = 8,
) -> str:
    """Convert ONNX model to TensorRT format.

    Args:
        onnx_path: Path to ONNX model
        tensorrt_path: Path to save TensorRT engine
        precision: Precision for TensorRT (FP32, FP16, INT8)
        max_workspace_size: Maximum workspace size in bytes
        max_batch_size: Maximum batch size

    Returns:
        Path to saved TensorRT engine
    """
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import tensorrt as trt

        log.info(f"Loading ONNX model from {onnx_path}")
        log.info(f"Converting to TensorRT with precision: {precision}")

        # Initialize TensorRT logger and builder
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    log.error(parser.get_error(error))
                raise RuntimeError("ONNX model parsing failed")

        # Set configuration
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size

        # Set precision
        if precision == "FP16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "INT8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: For INT8, you would need to set calibration parameters

        # Set optimization profile
        profile = builder.create_optimization_profile()
        input_shape = network.get_input(0).shape
        profile.set_shape(
            input_name=network.get_input(0).name,
            min=(1,) + input_shape[1:],
            opt=(max_batch_size,) + input_shape[1:],
            max=(max_batch_size,) + input_shape[1:],
        )
        config.add_optimization_profile(profile)

        log.info(f"Building TensorRT engine...")
        engine = builder.build_engine(network, config)

        log.info(f"Saving TensorRT engine to {tensorrt_path}")
        with open(tensorrt_path, "wb") as f:
            f.write(engine.serialize())

        log.info("TensorRT conversion completed successfully!")
        return tensorrt_path

    except ImportError as e:
        log.error(f"TensorRT conversion failed due to missing dependencies: {e}")
        log.error("Please install tensorrt and pycuda packages")
        raise
    except Exception as e:
        log.error(f"TensorRT conversion failed: {e}")
        raise


def create_inference_server_config(
    model_path: str,
    server_type: str = "triton",
    config_dir: str = "inference_server_config",
) -> dict:
    """Create configuration for inference server.

    Args:
        model_path: Path to model file
        server_type: Type of inference server (triton or mlflow)
        config_dir: Directory to save configuration

    Returns:
        Dictionary with server configuration
    """
    os.makedirs(config_dir, exist_ok=True)

    if server_type == "triton":
        return _create_triton_config(model_path, config_dir)
    elif server_type == "mlflow":
        return _create_mlflow_config(model_path, config_dir)
    else:
        raise ValueError(f"Unknown server type: {server_type}")


def _create_triton_config(model_path: str, config_dir: str) -> dict:
    """Create Triton Inference Server configuration."""
    model_name = "emotion_recognition"
    model_version = "1"

    # Create model repository structure
    model_repo = os.path.join(config_dir, model_name)
    os.makedirs(model_repo, exist_ok=True)

    # Create config.pbtxt
    config_content = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {{
    name: "audio_input"
    data_type: TYPE_FP32
    dims: [ 174, 40 ]
  }}
]
output [
  {{
    name: "emotion_output"
    data_type: TYPE_FP32
    dims: [ 8 ]
  }}
]
"""

    with open(os.path.join(model_repo, "config.pbtxt"), "w") as f:
        f.write(config_content.strip())

    # Copy model file
    model_filename = os.path.basename(model_path)
    model_dest = os.path.join(model_repo, str(model_version), model_filename)
    os.makedirs(os.path.dirname(model_dest), exist_ok=True)

    if os.path.exists(model_path):
        import shutil

        shutil.copy(model_path, model_dest)

    return {
        "server_type": "triton",
        "model_name": model_name,
        "model_repository": config_dir,
        "config_file": os.path.join(model_repo, "config.pbtxt"),
        "launch_command": f"trtserver --model-store={config_dir}",
    }


def _create_mlflow_config(model_path: str, config_dir: str) -> dict:
    """Create MLflow Serving configuration."""
    return {
        "server_type": "mlflow",
        "model_path": model_path,
        "launch_command": f"mlflow models serve -m {model_path} --port 5001",
    }
