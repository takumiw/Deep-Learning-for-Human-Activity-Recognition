"""Functions for training Multi Layer Perceptron (MLP)"""
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from src.utils import plot_learning_history, plot_model
from src.keras_callback import create_callback

tf.random.set_seed(0)


def train_and_predict(
    LOG_DIR: str,
    fold_id: int,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    mlp_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
    """Train MLP
    Args:
        X_train, X_valid, X_test: input signals of shape (num_samples, window_size, num_channels, 1)
        y_train, y_valid, y_test: onehot-encoded labels
    Returns:
        pred_train: train prediction
        pred_valid: train prediction
        pred_test: train prediction
        model: trained best model
    """
    model = build_model(
        input_shape=X_train.shape[1:], output_dim=y_train.shape[1], lr=mlp_params["lr"]
    )
    plot_model(model, path=f"{LOG_DIR}/model.png")

    callbacks = create_callback(
        model=model,
        path_chpt=f"{LOG_DIR}/trained_model_fold{fold_id}.h5",
        verbose=10,
        epochs=mlp_params["epochs"],
    )

    fit = model.fit(
        X_train,
        y_train,
        batch_size=mlp_params["batch_size"],
        epochs=mlp_params["epochs"],
        verbose=mlp_params["verbose"],
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
    )

    plot_learning_history(fit=fit, path=f"{LOG_DIR}/history_fold{fold_id}.png")
    model = keras.models.load_model(f"{LOG_DIR}/trained_model_fold{fold_id}.h5")

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    K.clear_session()
    return pred_train, pred_valid, pred_test, model


def build_model(
    input_shape: Tuple[int, int, int] = (128, 6, 1), output_dim: int = 6, lr: float = 0.001
) -> Model:
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5, seed=0))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5, seed=1))
    model.add(Dense(output_dim))
    model.add(Activation("softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=lr), metrics=["accuracy"]
    )
    return model
