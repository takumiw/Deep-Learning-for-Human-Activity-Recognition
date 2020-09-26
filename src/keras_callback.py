from logging import getLogger
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model

from src.utils import round

logger = getLogger(__name__)


class F1Callback(Callback):
    """Plot f1 value of every epoch"""

    def __init__(
        self,
        model: Model,
        path_f1_history: str,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model = model
        self.path_f1_history = path_f1_history
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.tr_fscores: List[float] = []  # train f1 of every epoch
        self.val_fscores: List[float] = []  # valid f1 of every epoch

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        tr_pred = self.model.predict(self.X_tr)
        tr_macro_f1 = f1_score(self.y_tr.argmax(axis=1), tr_pred.argmax(axis=1), average="macro")
        self.tr_fscores.append(tr_macro_f1)
        val_pred = self.model.predict(self.X_val)
        val_macro_f1 = f1_score(self.y_val.argmax(axis=1), val_pred.argmax(axis=1), average="macro")
        self.val_fscores.append(val_macro_f1)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(self.tr_fscores, label="f1_score for training")
        ax.plot(self.val_fscores, label="f1_score for validation")
        ax.set_title("model f1_score")
        ax.set_xlabel("epoch")
        ax.set_ylabel("f1_score")
        ax.legend(loc="upper right")
        fig.savefig(self.path_f1_history)
        plt.close()


class PeriodicLogger(Callback):
    """Logging history every n epochs"""

    def __init__(
        self, metric: str = "accuracy", verbose: int = 1, epochs: Optional[int] = None
    ) -> None:
        self.metric = metric
        self.verbose = verbose
        self.epochs = epochs

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        epoch += 1
        if epoch % self.verbose == 0:
            msg = " - ".join(
                [
                    f"Epoch {epoch}/{self.epochs}",
                    f"loss: {round(logs['loss'], 0.0001)}",
                    f"{self.metric}: {round(logs[self.metric], 0.0001)}",
                    f"val_loss: {round(logs['val_loss'], 0.0001)}",
                    f"val_{self.metric}: {round(logs[f'val_{self.metric}'], 0.0001)}",
                ]
            )
            logger.debug(msg)


def create_callback(
    model: Model, path_chpt: str, patience: int = 30, metric: str = "accuracy", verbose: int = 10, epochs: Optional[int] = None
) -> List[Any]:
    """callback settinngs
    Args:
        model (Model)
        path_chpt (str): path to save checkpoint
    Returns:
        callbacks (List[Any]): List of Callback
    """
    callbacks = []
    callbacks.append(
        EarlyStopping(monitor="val_loss", min_delta=0, patience=patience, verbose=1, mode="min")
    )
    callbacks.append(ModelCheckpoint(filepath=path_chpt, save_best_only=True))
    callbacks.append(PeriodicLogger(metric=metric, verbose=verbose, epochs=epochs))
    return callbacks
