"""Collection of utility functions"""

from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from logging import getLogger
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from tensorflow import keras

shap.initjs()
logger = getLogger(__name__)


def color_generator(i: int) -> str:
    l = ["#FFAF6D", "#DC4195", "#F1E898", "#6DCBB9", "#3E89C4", "#6F68CF"]
    return l[i]


def round_float(f: float, r: float = 0.000001) -> float:
    return float(Decimal(str(f)).quantize(Decimal(str(r)), rounding=ROUND_HALF_UP))


def round_list(l: List[float], r: float = 0.000001) -> List[float]:
    return [round_float(f, r) for f in l]


def round_dict(d: Dict[Any, Any], r: float = 0.000001) -> Dict[Any, Any]:
    return {key: round(d[key], r) for key in d.keys()}


def round(arg: Any, r: float = 0.000001) -> Any:
    if type(arg) == float or type(arg) == np.float64 or type(arg) == np.float32:
        return round_float(arg, r)
    elif type(arg) == list or type(arg) == np.ndarray:
        return round_list(arg, r)
    elif type(arg) == dict:
        return round_dict(arg, r)
    else:
        logger.error(f"Arg type {type(arg)} is not supported")
        return arg


def check_class_balance(
    y_train: np.ndarray, y_test: np.ndarray, label2act: Dict[int, str], n_class: int = 6
) -> None:
    c_train = Counter(y_train)
    c_test = Counter(y_test)

    for c, mode in zip([c_train, c_test], ["train", "test"]):
        logger.debug(f"{mode} labels")
        len_y = sum(c.values())
        for label_id in range(n_class):
            logger.debug(
                f"{label2act[label_id]} ({label_id}): {c[label_id]} samples ({c[label_id] / len_y * 100:.04} %)"
            )


def plot_feature_importance(
    models: List[Any],
    num_features: int,
    cols: List[str],
    importance_type: str = "gain",
    path: str = "importance.png",
    figsize: Tuple[int, int] = (16, 10),
    max_display: int = -1,
) -> None:
    """
    Args:
        importance_type: chosen from "gain" or "split"
    """
    importances = np.zeros((len(models), num_features))
    for i, model in enumerate(models):
        importances[i] = model.feature_importance(importance_type=importance_type)

    importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({"Feature": cols, "Value": importance})
    importance_df = importance_df.sort_values(by="Value", ascending=False)[:max_display]

    plt.figure(figsize=figsize)
    sns.barplot(x="Value", y="Feature", data=importance_df)
    plt.title("Feature Importance (avg over folds)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_shap_summary(
    models: List[Any],
    X_train: pd.DataFrame,
    class_names: List[str],
    path: str = "shap_summary_plot.png",
    max_display: Optional[int] = None,
) -> None:
    shap_values_list = []
    for model in models:
        explainer = shap.TreeExplainer(
            model,
            num_iteration=model.best_iteration,
            feature_perturbation="tree_path_dependent",
        )
        shap_value_oof = explainer.shap_values(X_train)
        shap_values_list.append(shap_value_oof)

    shap_values = [np.zeros(shap_values_list[0][0].shape) for _ in range(len(class_names))]
    for shap_value_oof in shap_values_list:
        for i in range(len(class_names)):
            shap_values[i] += shap_value_oof[i]

    for i in range(len(class_names)):
        shap_values[i] /= len(models)

    shap.summary_plot(
        shap_values,
        X_train,
        max_display=max_display,
        class_names=class_names,
        color=color_generator,
        show=False,
    )
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cms: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    path: str = "confusion_matrix.png",
) -> None:
    """Plot confusion matrix"""
    cms = [np.mean(cms[mode], axis=0) for mode in ["train", "valid", "test"]]

    fig, ax = plt.subplots(ncols=3, figsize=(20, 7))
    for i, (cm, mode) in enumerate(zip(cms, ["train", "valid", "test"])):
        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            square=True,
            vmin=0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax[i],
        )
        ax[i].set_xlabel("Predicted label")
        ax[i].set_ylabel("True label")
        ax[i].set_title(f"Normalized confusion matrix - {mode}")

    plt.tight_layout()
    fig.savefig(path)
    plt.close()


def plot_model(model: Any, path: str) -> None:
    if not os.path.isfile(path):
        keras.utils.plot_model(model, to_file=path, show_shapes=True)


def plot_learning_history(fit: Any, metric: str = "accuracy", path: str = "history.png") -> None:
    """Plot learning curve
    Args:
        fit (Any): History object
        path (str, default="history.png")
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    axL.plot(fit.history["loss"], label="train")
    axL.plot(fit.history["val_loss"], label="validation")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(fit.history[metric], label="train")
    axR.plot(fit.history[f"val_{metric}"], label="validation")
    axR.set_title(metric.capitalize())
    axR.set_xlabel("epoch")
    axR.set_ylabel(metric)
    axR.legend(loc="upper right")

    fig.savefig(path)
    plt.close()
