from datetime import datetime
import json
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from src.data_prep.load import load_raw_data
from src.utils import check_class_balance, round
from src.utils import plot_feature_importance, plot_shap_summary, plot_confusion_matrix
from models.deep_conv_lstm import train_and_predict

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

# Logging settings
EXEC_TIME = "deep-conv-lstm-" + datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(CUR_DIR, f"logs/{EXEC_TIME}")
os.makedirs(LOG_DIR, exist_ok=True)  # Create log directory

formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
basicConfig(filename=f"{LOG_DIR}/{EXEC_TIME}.log", level=DEBUG, format=formatter)
mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
mpl_logger.setLevel(WARNING)
# Handle logging to both logging and stdout.
getLogger().addHandler(StreamHandler(sys.stdout))

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")

X_train, X_test, y_train, y_test, label2act, act2label = load_raw_data()
logger.debug(f"{X_train.shape=} {X_test.shape=}")
logger.debug(f"{y_train.shape=} {y_test.shape=}")

check_class_balance(y_train.flatten(), y_test.flatten(), label2act=label2act)

# Split data by preserving the percentage of samples for each class.
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
valid_preds = np.zeros((X_train.shape[0], 6))
test_preds = np.zeros((n_splits, X_test.shape[0], 6))
models = []
scores: Dict[str, Dict[str, List[Any]]] = {
    "logloss": {"train": [], "valid": [], "test": []},
    "accuracy": {"train": [], "valid": [], "test": []},
    "precision": {"train": [], "valid": [], "test": []},
    "recall": {"train": [], "valid": [], "test": []},
    "f1": {"train": [], "valid": [], "test": []},
    "cm": {"train": [], "valid": [], "test": []},
    "per_class_f1": {"train": [], "valid": [], "test": []},
}
# Load hyper-parameters
with open(os.path.join(CUR_DIR, "configs/default.json"), "r") as f:
    dcl_params = json.load(f)["deep_conv_lstm_params"]
    logger.debug(f"{dcl_params=}")

y_test = keras.utils.to_categorical(y_test, 6)

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train[train_index, :]
    X_val = X_train[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]
    
    y_tr = keras.utils.to_categorical(y_tr, 6)
    y_val = keras.utils.to_categorical(y_val, 6)

    logger.debug(f"{X_tr.shape=} {X_val.shape=} {X_test.shape=}")
    logger.debug(f"{y_tr.shape=} {y_val.shape=} {y_test.shape=}")

    pred_tr, pred_val, pred_test, model = train_and_predict(
        LOG_DIR, fold_id, X_tr, X_val, X_test, y_tr, y_val, dcl_params
    )
    models.append(model)

    valid_preds[valid_index] = pred_val
    test_preds[fold_id] = pred_test

    for pred, X, y, mode in zip(
        [pred_tr, pred_val, pred_test], [X_tr, X_val, X_test], [y_tr, y_val, y_test], ["train", "valid", "test"]
    ):
        loss, acc = model.evaluate(X, y, verbose=0)
        pred = pred.argmax(axis=1)
        y = y.argmax(axis=1)
        scores["logloss"][mode].append(loss)
        scores["accuracy"][mode].append(acc)
        scores["precision"][mode].append(precision_score(y, pred, average="macro"))
        scores["recall"][mode].append(recall_score(y, pred, average="macro"))
        scores["f1"][mode].append(f1_score(y, pred, average="macro"))
        scores["cm"][mode].append(confusion_matrix(y, pred, normalize="true"))
        scores["per_class_f1"][mode].append(f1_score(y, pred, average=None))

# Output Cross Validation Scores
logger.debug("---Cross Validation Scores---")
for mode in ["train", "valid", "test"]:
    logger.debug(f"---{mode}---")
    for metric in ["logloss", "accuracy", "precision", "recall", "f1"]:
        logger.debug(f"{metric}={round(np.mean(scores[metric][mode]))}")

    class_f1_mat = scores["per_class_f1"][mode]
    class_f1_result = {}
    for class_id in range(6):
        mean_class_f1 = np.mean([class_f1_mat[i][class_id] for i in range(n_splits)])
        class_f1_result[label2act[class_id]] = mean_class_f1
    logger.debug(f"per-class f1={round(class_f1_result)}")

# Output Final Scores Averaged over Folds
logger.debug("---Final Test Scores Averaged over Folds---")
test_pred = np.mean(test_preds, axis=0).argmax(axis=1)  # average over folds
y_test = y_test.argmax(axis=1)
logger.debug(f"accuracy={accuracy_score(y_test, test_pred)}")
logger.debug(f"precision={precision_score(y_test, test_pred, average='macro')}")
logger.debug(f"recall={recall_score(y_test, test_pred, average='macro')}")
logger.debug(f"f1={f1_score(y_test, test_pred, average='macro')}")
logger.debug(f"per-class f1={f1_score(y_test, test_pred, average=None)}")

# Plot comfusion matrix
plot_confusion_matrix(
    cms=scores["cm"],
    labels=[
        "LAYING",
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
    ],
    path=f"{LOG_DIR}/comfusion_matrix.png",
)

np.save(f"{LOG_DIR}/valid_oof.npy", valid_preds)
np.save(f"{LOG_DIR}/test_oof.npy", np.mean(test_preds, axis=0))  # Averaging
