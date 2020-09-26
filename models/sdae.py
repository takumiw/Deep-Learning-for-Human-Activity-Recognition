"""Functions for training Stacked Denoising AutoEncoder (SDAE)"""
from logging import getLogger
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from src.utils import plot_learning_history, plot_model
from src.keras_callback import create_callback

tf.random.set_seed(0)
logger = getLogger(__name__)


class SDAE:
    def __init__(self, LOG_DIR: str, fold_id: int = 0, **params: Dict[str, Any]):
        self.LOG_DIR = LOG_DIR
        self.fold_id = fold_id
        self.pretrain_lr = params["pretrain_lr"]
        self.pretrain_epochs = params["pretrain_epochs"]
        self.pretrain_batch_size = params["pretrain_batch_size"]

        self.finetune_lr = params["finetune_lr"]
        self.finetune_epochs = params["finetune_epochs"]
        self.finetune_batch_size = params["finetune_batch_size"]

        self.verbose = params["verbose"]
        self.freeze_layers = params["freeze_layers"]

    def add_noise(
        self, signal: np.ndarray, noise_type: str = "mask", noise_factor: float = 0.4, seed: int = 0
    ) -> np.ndarray:
        """Add noise to create corrupted signals
        Args:
            signal (np.ndarray): input signal
            noise_type (str): chosen from "mask" (masking noise) or "noise" (additive Gaussian noise)
            noise_factor (float): strength of corruption
            seed (int): seed of normal distribution of noise
        Returns:
            signal (np.ndarray): corrupted signal
        Note:
            noise_factor is preffered to be set 0.4 for mask noise and 0.5 for Gaussian noise.
            This method is not used this time bacause I used dropout layer instead.
        """
        if noise_type == "mask":
            random.seed(seed)
            corrupt_idx = random.sample(range(len(signal)), int(len(signal) * noise_factor))
            signal[corrupt_idx] = 0
        elif noise_type == "noise":
            np.random.seed(seed=seed)
            noise = np.random.normal(loc=0.0, scale=1.0, size=signal.shape)
            signal = signal + noise_factor * noise
            signal = np.clip(signal, 0.0, 1.0)
        return signal

    def train_1st_level(
        self, X_train: np.ndarray, X_valid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """train 1st level DAE
        Note that input must to be scaled betweeen [0, 1] because output layer is governed by sigmoid function.
        Args
            X_train (np.ndarray): has shape (num_samples, window_size * num_channel)
            X_valid (np.ndarray): has shape (num_samples, window_size * num_channel)
        """
        # Build base model
        model = Sequential()
        model.add(Dropout(0.4, seed=10, name="1st_level_dropout"))
        model.add(
            Dense(100, input_dim=X_train.shape[1], activation="sigmoid", name="1st_level_fc")
        )  # encoder
        model.add(Dense(X_train.shape[1], activation="sigmoid"))  # decoder
        model.compile(
            loss="mean_squared_error",
            optimizer=optimizers.Adam(lr=self.pretrain_lr),
            metrics=["mse"],
        )

        callbacks = create_callback(
            model=model,
            path_chpt=f"{self.LOG_DIR}/trained_model_1st_level_fold{self.fold_id}.h5",
            metric="mse",
            verbose=50,
            epochs=self.pretrain_epochs,
        )

        fit = model.fit(
            x=X_train,
            y=X_train,
            batch_size=self.pretrain_batch_size,
            epochs=self.pretrain_epochs,
            verbose=self.verbose,
            validation_data=(X_valid, X_valid),
            callbacks=callbacks,
        )

        plot_learning_history(
            fit=fit, metric="mse", path=f"{self.LOG_DIR}/history_1st_level_fold{self.fold_id}.png"
        )
        plot_model(model, path=f"{self.LOG_DIR}/model_1st_level.png")

        # Load best model
        model = keras.models.load_model(
            f"{self.LOG_DIR}/trained_model_1st_level_fold{self.fold_id}.h5"
        )
        self.model_1st_level = model
        encoder = Model(inputs=model.input, outputs=model.get_layer("1st_level_fc").output)
        pred_train = encoder.predict(X_train)  # predict with clean signal
        pred_valid = encoder.predict(X_valid)  # predict with clean signal
        return pred_train, pred_valid

    def train_2nd_level(
        self, X_train: np.ndarray, X_valid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """train 2nd level DAE
        Note that input must to be scaled betweeen [0, 1] because output layer is governed by sigmoid function.
        Args
            X_train (np.ndarray): has shape (num_samples, output dim of 1st level)
            X_valid (np.ndarray): has shape (num_samples, output dim of 1st level)
        """
        # Build base model
        model = Sequential()
        model.add(Dropout(0.4, seed=20, name="2nd_level_dropout"))
        model.add(
            Dense(30, input_dim=X_train.shape[1], activation="sigmoid", name="2nd_level_fc")
        )  # encoder
        model.add(Dense(X_train.shape[1], activation="sigmoid", name="2nd_level_output"))  # decoder
        model.compile(
            loss="mean_squared_error",
            optimizer=optimizers.Adam(lr=self.pretrain_lr),
            metrics=["mse"],
        )

        # Create callback
        callbacks = create_callback(
            model=model,
            path_chpt=f"{self.LOG_DIR}/trained_model_2nd_level_fold{self.fold_id}.h5",
            metric="mse",
            verbose=50,
            epochs=self.pretrain_epochs,
        )

        fit = model.fit(
            x=X_train,
            y=X_train,
            batch_size=self.pretrain_batch_size,
            epochs=self.pretrain_epochs,
            verbose=self.verbose,
            validation_data=(X_valid, X_valid),
            callbacks=callbacks,
        )

        plot_learning_history(
            fit=fit, metric="mse", path=f"{self.LOG_DIR}/history_2nd_level_fold{self.fold_id}.png"
        )
        plot_model(model, path=f"{self.LOG_DIR}/model_2nd_level.png")

        # Load best model
        model = keras.models.load_model(
            f"{self.LOG_DIR}/trained_model_2nd_level_fold{self.fold_id}.h5"
        )
        self.model_2nd_level = model
        encoder = Model(inputs=model.input, outputs=model.get_layer("2nd_level_fc").output)
        pred_train = encoder.predict(X_train)  # predict with clean signal
        pred_valid = encoder.predict(X_valid)  # predict with clean signal
        return pred_train, pred_valid

    def stack_encoders(
        self,
    ) -> Model:
        """Stack encoders of 1st and 2nd level
        Returns
            Model: stacked model
        """
        model_1st_level = self.model_1st_level
        model_2nd_level = self.model_2nd_level

        model = Sequential()
        model.add(model_1st_level.get_layer("1st_level_fc"))
        model.add(model_2nd_level.get_layer("2nd_level_fc"))
        return model

    def finetune(
        self,
        X_train: np.ndarray,
        X_valid: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
        """Fine-tune stacked model for classification.
        Returns
            Model: stacked model
        """
        model = self.stack_encoders()
        model.add(
            Dense(y_train.shape[1], activation="softmax", name="output")
        )  # Add output layer for classification

        for layer in model.layers:
            if layer in self.freeze_layers:
                model.get_layer([layer]).trainable = False
                logger.debug(f"Freezed {layer=}")

        # Recompile model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.Adam(lr=self.finetune_lr),
            metrics=["accuracy"],
        )

        callbacks = create_callback(
            model=model,
            path_chpt=f"{self.LOG_DIR}/trained_model_finetune_fold{self.fold_id}.h5",
            verbose=50,
            epochs=self.finetune_epochs,
        )

        fit = model.fit(
            X_train,
            y_train,
            batch_size=self.finetune_batch_size,
            epochs=self.finetune_epochs,
            verbose=self.verbose,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
        )

        plot_learning_history(
            fit=fit, path=f"{self.LOG_DIR}/history_finetune_fold{self.fold_id}.png"
        )
        plot_model(model, path=f"{self.LOG_DIR}/model_finetune.png")

        model = keras.models.load_model(
            f"{self.LOG_DIR}/trained_model_finetune_fold{self.fold_id}.h5"
        )

        pred_train = model.predict(X_train)
        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)

        K.clear_session()
        return pred_train, pred_valid, pred_test, model
