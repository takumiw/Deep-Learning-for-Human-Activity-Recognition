"""Functions for training Convolutional Neural Network (CNN)"""
from logging import getLogger
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

from src.utils import plot_learning_history, plot_model

tf.random.set_seed(0)
logger = getLogger(__name__)


def train_and_predict(
    LOG_DIR: str,
    fold_id: int,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    cnn_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """CNNモデルを学習する
    Args:
        X_train, X_valid, X_test: input signals of shape (num_samples, window_size, num_channels)
        y_train, y_valid, y_test: label-encoded labels
    Returns:
        pred_train: train prediction
        pred_valid: train prediction
        pred_test: train prediction
        model: trained best model
    """
    """X_train = X_train.reshape(*X_train.shape, 1)  # (x, 128, 6) -> (x, 128, 6, 1)
    X_valid = X_valid.reshape(*X_valid.shape, 1)  # (x, 128, 6) -> (x, 128, 6, 1)
    X_test = X_test.reshape(*X_test.shape, 1)  # (x, 128, 6) -> (x, 128, 6, 1)

    y_train = keras.utils.to_categorical(y_train, 6)
    y_valid = keras.utils.to_categorical(y_valid, 6)
    y_test = keras.utils.to_categorical(y_test, 6)
    """
    model = create_baseline(
        input_shape=X_train.shape[1:], output_dim=y_train.shape[1], lr=cnn_params["lr"]
    )
    plot_model(model, path=f"{LOG_DIR}/model.png")

    # 各種callbackの設定
    callbacks = create_callback(model=model, path_chpt=f"{LOG_DIR}/trained_model_fold{fold_id}.h5")

    fit = model.fit(
        X_train,
        y_train,
        batch_size=cnn_params["batch_size"],
        epochs=cnn_params["epochs"],
        verbose=cnn_params["verbose"],
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
    )

    # 学習曲線をプロットする
    plot_learning_history(fit=fit, path=f"{LOG_DIR}/hitsoty_fold{fold_id}.png")

    # Logging training history every 10 epochs
    df = pd.DataFrame(fit.history)
    index = np.arange(0, len(df), 10)
    logger.debug(f"\n{df.iloc[index]}")
    logger.debug(f"Early stopping at {len(df)-1}th epoch")

    # ベストのモデルをロードする
    model = keras.models.load_model(f"{LOG_DIR}/trained_model_fold{fold_id}.h5")

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    K.clear_session()
    return pred_train, pred_valid, pred_test, model


def create_baseline(
    input_shape: Tuple[int, int, int] = (128, 6, 1), output_dim: int = 6, lr: float = 0.001
) -> Any:
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 1), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=(5, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=(5, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=(5, 1)))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(128))
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


def create_callback(model: Any, path_chpt: str) -> List[Any]:
    """callbackの設定
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNNモデル
        path_f1_history (str): f1の経過を出力するパス
    Returns:
        callbacks (List[Any]): Callbackのリスト
    """
    callbacks = []
    callbacks.append(
        EarlyStopping(monitor="val_loss", min_delta=0, patience=30, verbose=1, mode="min")
    )
    callbacks.append(ModelCheckpoint(filepath=path_chpt, save_best_only=True))
    # callbacks.append(ReduceLROnPlateau(factor=0.2, patience=10, verbose=1, min_lr=0.00001))
    return callbacks
