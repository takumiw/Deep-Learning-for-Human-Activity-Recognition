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

from src.data_prep.load import load_features
from src.utils import check_class_balance, round
from src.utils import plot_feature_importance, plot_shap_summary, plot_confusion_matrix
from models.lgbm import train_and_predict

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

# Logging settings
EXEC_TIME = "lgbm-" + datetime.now().strftime("%Y%m%d-%H%M%S")
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

X_train, X_test, y_train, y_test, label2act, act2label = load_features()
logger.debug(f"{X_train.shape=} {X_test.shape=}")
logger.debug(f"{y_train.shape=} {y_test.shape=}")

check_class_balance(y_train.values.flatten(), y_test.values.flatten(), label2act=label2act)

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
    lgbm_params = json.load(f)["lgbm_params"]
    logger.debug(f"{lgbm_params=}")

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train.loc[train_index]
    y_val = y_train.loc[valid_index]

    logger.debug(f"{X_tr.shape=} {X_val.shape=} {X_test.shape=}")
    logger.debug(f"{y_tr.shape=} {y_val.shape=} {y_test.shape=}")

    pred_tr, pred_val, pred_test, model = train_and_predict(
        X_tr, X_val, X_test, y_tr, y_val, lgbm_params
    )
    models.append(model)

    valid_preds[valid_index] = pred_val
    test_preds[fold_id] = pred_test

    scores["logloss"]["train"].append(model.best_score["training"]["multi_logloss"])
    scores["logloss"]["valid"].append(model.best_score["valid_1"]["multi_logloss"])
    scores["logloss"]["test"].append(log_loss(y_test, pred_test))

    for pred, y, mode in zip(
        [pred_tr, pred_val, pred_test], [y_tr, y_val, y_test], ["train", "valid", "test"]
    ):
        pred = pred.argmax(axis=1)
        scores["accuracy"][mode].append(accuracy_score(y, pred))
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

# Plot feature importance
for importance_type in ["split", "gain"]:
    plot_feature_importance(
        models=models,
        num_features=X_train.shape[1],
        importance_type=importance_type,
        cols=X_train.columns.tolist(),
        path=f"{LOG_DIR}/importance_{importance_type}.png",
        figsize=(16, 20),
        max_display=100,
    )

# Plot shap values over folds
plot_shap_summary(
    models,
    X_train,
    class_names=[
        "LAYING",
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
    ],
    path=f"{LOG_DIR}/shap_summary_plot.png",
    max_display=100,
)

np.save(f"{LOG_DIR}/valid_oof.npy", valid_preds)
np.save(f"{LOG_DIR}/test_oof.npy", np.mean(test_preds, axis=0))  # Averaging
