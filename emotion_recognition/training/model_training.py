"""Model training functionality."""

import os
from typing import Tuple

import mlflow
import mlflow.keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Set MLflow tracking URI to use SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model_config: dict = None,
    training_config: dict = None,
) -> Sequential:
    """Train LSTM model for emotion recognition.

    Args:
        X_train: Training features
        y_train: Training labels (one-hot encoded)
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        label_encoder: Label encoder for emotion classes
        model_config: Model configuration dictionary
        training_config: Training configuration dictionary

    Returns:
        Trained Keras model
    """
    # Set default configurations
    if model_config is None:
        model_config = {
            "lstm_units": 128,
            "dropout_rate": 0.5,
            "dense_units": 64,
            "learning_rate": 0.001,
        }

    if training_config is None:
        training_config = {
            "epochs": 30,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping_patience": 5,
            "model_checkpoint_monitor": "val_accuracy",
            "model_checkpoint_mode": "max",
        }

    # Start MLflow run
    mlflow.set_experiment("emotion_recognition_training")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_config)
        mlflow.log_params(training_config)

        # Build LSTM model
        model = Sequential()
        model.add(
            LSTM(
                model_config["lstm_units"],
                input_shape=(X_train.shape[1], X_train.shape[2]),
                return_sequences=False,
            )
        )
        model.add(Dropout(model_config["dropout_rate"]))
        model.add(Dense(model_config["dense_units"], activation="relu"))
        model.add(Dense(y_train.shape[1], activation="softmax"))

        # Compile model
        optimizer = Adam(learning_rate=model_config["learning_rate"])
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        # Create callbacks
        os.makedirs("models", exist_ok=True)
        callbacks = [
            ModelCheckpoint(
                "models/audio_lstm.h5",
                monitor=training_config["model_checkpoint_monitor"],
                save_best_only=True,
                mode=training_config["model_checkpoint_mode"],
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=training_config["early_stopping_patience"],
                restore_best_weights=True,
            ),
        ]

        # Train model
        history = model.fit(
            X_train,
            y_train,
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
        )

        # Log metrics
        for epoch, acc in enumerate(history.history["accuracy"]):
            mlflow.log_metric("train_accuracy", acc, step=epoch)
        for epoch, val_acc in enumerate(history.history["val_accuracy"]):
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        for epoch, loss in enumerate(history.history["loss"]):
            mlflow.log_metric("train_loss", loss, step=epoch)
        for epoch, val_loss in enumerate(history.history["val_loss"]):
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Log model
        mlflow.keras.log_model(model, "model")

        # Log label encoder classes
        mlflow.log_param("num_classes", len(label_encoder.classes_))
        mlflow.log_param("classes", list(label_encoder.classes_))

    return model
