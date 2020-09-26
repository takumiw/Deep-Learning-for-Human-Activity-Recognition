"""Load dataset"""
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_prep.preprocess_raw_data import preprocess_raw_data

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
DATA_DIR = os.path.join(CUR_DIR, "../../data")


def load_features() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, str],
    Dict[str, int],
]:
    """Load created features.
    The following six classes are included in this experiment.
        - WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
    The following transition classes are excluded.
        - STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, and LIE_TO_STAND
    Returns:
        X_train (pd.DataFrame): Explanatory variable in train data
        X_test (pd.DataFrame): Explanatory variable in test data
        y_train (pd.DataFrame): Teacher data in train data
        y_test (pd.DataFrame): Teacher data in test data
        label2act (Dict[int, str]): Dict of label_id to title_of_class
        act2label (Dict[str, int]): Dict of title_of_class to label_id
    """
    X_train = pd.read_pickle(os.path.join(DATA_DIR, "my_dataset/X_train.pickle"))
    y_train = pd.DataFrame(np.load(os.path.join(DATA_DIR, "my_dataset/y_train.npy")))
    subject_id_train = pd.read_table(
        os.path.join(DATA_DIR, "hapt_data_set/Train/subject_id_train.txt"), sep=" ", header=None
    )

    X_test = pd.read_pickle(os.path.join(DATA_DIR, "my_dataset/X_test.pickle"))
    y_test = pd.DataFrame(np.load(os.path.join(DATA_DIR, "my_dataset/y_test.npy")))
    subject_id_test = pd.read_table(
        os.path.join(DATA_DIR, "hapt_data_set/Test/subject_id_test.txt"), sep=" ", header=None
    )

    activity_labels = pd.read_table(
        os.path.join(DATA_DIR, "hapt_data_set/activity_labels.txt"), header=None
    ).values.flatten()
    activity_labels = np.array([label.rstrip().split() for label in activity_labels])
    label2act, act2label = {}, {}
    for label, activity in activity_labels:
        label2act[int(label)] = activity
        act2label[activity] = int(label)

    class_names_inc = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    class_ids_inc = [act2label[c] for c in class_names_inc]

    idx_train = y_train[y_train[0].isin(class_ids_inc)].index
    X_train = X_train.iloc[idx_train].reset_index(drop=True)
    y_train = y_train.iloc[idx_train].reset_index(drop=True)
    # subject_id_train = subject_id_train.iloc[idx_train].reset_index(drop=True)

    idx_test = y_test[y_test[0].isin(class_ids_inc)].index
    X_test = X_test.iloc[idx_test].reset_index(drop=True)
    y_test = y_test.iloc[idx_test].reset_index(drop=True)
    # subject_id_test = subject_id_test.iloc[idx_test].reset_index(drop=True)

    # Replace 6 to 0
    rep_activity = label2act[6]
    label2act[0] = rep_activity
    label2act.pop(6)
    act2label[rep_activity] = 0

    y_train = y_train.replace(6, 0)
    y_test = y_test.replace(6, 0)

    return X_train, X_test, y_train, y_test, label2act, act2label


def load_raw_data(
    scaler: str = "normalize",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[str, int]]:
    """Load raw dataset.
    The following six classes are included in this experiment.
        - WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
    The following transition classes are excluded.
        - STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, and LIE_TO_STAND
    Args:
        scaler (str): scaler for raw signals, chosen from normalize or minmax
    Returns:
        X_train (pd.DataFrame):
        X_test (pd.DataFrame):
        y_train (pd.DataFrame):
        y_test (pd.DataFrame):
        label2act (Dict[int, str]): Dict of label_id to title_of_class
        act2label (Dict[str, int]): Dict of title_of_class to label_id
    """
    X_train, X_test = preprocess_raw_data(scaler=scaler)
    # X_train = np.load(os.path.join(DATA_DIR, "my_dataset/Raw_X_train.npy"))
    # X_test = np.load(os.path.join(DATA_DIR, "my_dataset/Raw_X_test.npy"))
    y_train = pd.read_table(os.path.join(DATA_DIR, "my_dataset/y_train.txt"), sep=" ", header=None)
    y_test = pd.read_table(os.path.join(DATA_DIR, "my_dataset/y_test.txt"), sep=" ", header=None)

    activity_labels = pd.read_table(
        os.path.join(DATA_DIR, "hapt_data_set/activity_labels.txt"), header=None
    ).values.flatten()
    activity_labels = np.array([label.rstrip().split() for label in activity_labels])
    label2act, act2label = {}, {}
    for label, activity in activity_labels:
        label2act[int(label)] = activity
        act2label[activity] = int(label)

    # Replace 6 to 0
    rep_activity = label2act[6]
    label2act[0] = rep_activity
    label2act.pop(6)
    act2label[rep_activity] = 0

    y_train = y_train.replace(6, 0)
    y_test = y_test.replace(6, 0)

    return X_train, X_test, y_train.values, y_test.values, label2act, act2label
