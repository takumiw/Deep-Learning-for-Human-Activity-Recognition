"""Preprocess raw data for creating input for DL tasks."""
import glob
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data_prep.preprocessing import Preprocess

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
DATA_DIR = os.path.join(CUR_DIR, "../../data/")  # Path to dataset directory
TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]


def preprocess_signal(signal: pd.DataFrame) -> pd.DataFrame:
    _signal = signal.copy()
    of = Preprocess()
    _signal = of.apply_filter(_signal, filter="median")
    _signal = of.apply_filter(_signal, filter="butterworth")
    _signal = of.segment_signal(_signal)
    return _signal


def scale(
    signal: pd.DataFrame, scaler="normalize", minmax_range: Optional[Tuple[int, int]] = (0, 1)
) -> pd.DataFrame:
    if scaler == "normalize":
        signal = StandardScaler().fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])


def preprocess_raw_data(scaler):
    acc_files = sorted(glob.glob(os.path.join(DATA_DIR, "hapt_data_set/RawData/acc*.txt")))
    gyro_files = sorted(glob.glob(os.path.join(DATA_DIR, "hapt_data_set/RawData/gyro*.txt")))
    label_info = pd.read_table(
        os.path.join(DATA_DIR, "hapt_data_set/RawData/labels.txt"),
        sep=" ",
        header=None,
        names=["ExpID", "UserID", "ActID", "ActStart", "ActEnd"],
    )

    X_train = np.array([])
    X_test = np.array([])

    for acc_file, gyro_file in zip(acc_files, gyro_files):
        exp_id = int(acc_file.split("exp")[1][:2])
        user_id = int(acc_file.split("user")[1][:2])

        temp_label_info = label_info[
            (label_info.ExpID == exp_id)
            & (label_info.UserID == user_id)
            & (label_info.ActID.isin([1, 2, 3, 4, 5, 6]))
        ]

        acc_raw = pd.read_table(acc_file, sep=" ", header=None, names=["x", "y", "z"])
        gyro_raw = pd.read_table(gyro_file, sep=" ", header=None, names=["x", "y", "z"])

        acc_raw = scale(acc_raw, scaler=scaler)
        gyro_raw = scale(gyro_raw, scaler=scaler)

        for _, _, act_id, act_start, act_end in temp_label_info.values:
            temp_acc_raw = acc_raw.iloc[act_start : act_end + 1]
            temp_gyro_raw = gyro_raw.iloc[act_start : act_end + 1]
            tAccXYZ = preprocess_signal(temp_acc_raw)
            tBodyGyroXYZ = preprocess_signal(temp_gyro_raw)
            features = np.zeros((len(tAccXYZ), 128, 6))

            for i in range(len(tAccXYZ)):
                feature = pd.DataFrame(
                    np.concatenate((tAccXYZ[i], tBodyGyroXYZ[i]), 1),
                    columns=["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"],
                )
                features[i] = feature

            if user_id in TRAIN_SUBJECTS:
                if len(X_train) == 0:
                    X_train = features
                else:
                    X_train = np.vstack((X_train, features))
            else:
                if len(X_test) == 0:
                    X_test = features
                else:
                    X_test = np.vstack((X_test, features))

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 6, 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 6, 1)

    return X_train, X_test
