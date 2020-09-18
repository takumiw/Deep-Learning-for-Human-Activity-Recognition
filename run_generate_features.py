from datetime import datetime
import glob
from logging import basicConfig, getLogger, StreamHandler, DEBUG
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_prep.create_features import create_features, get_feature_names

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
DATA_DIR = os.path.join(CUR_DIR, "data/")  # Path to dataset directory
ATC_IDS = [1, 2, 3, 4, 5, 6]  # Target activity ids
TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]

# Logging settings
EXEC_TIME = "generate-features-" + datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(CUR_DIR, f"logs/{EXEC_TIME}")
os.makedirs(LOG_DIR, exist_ok=True)  # Create log directory

formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
basicConfig(filename=f"{LOG_DIR}/{EXEC_TIME}.log", level=DEBUG, format=formatter)
# Handle logging to both logging and stdout.
getLogger().addHandler(StreamHandler(sys.stdout))

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")


def main() -> None:
    """Create features from raw acceleration and gyroscope sensor data, and save features/labels to pickle/npy files"""
    logger.debug("Start creating features...")

    # Get file names of raw dataset
    acc_files = sorted(glob.glob(os.path.join(DATA_DIR, "hapt_data_set/RawData/acc*.txt")))
    gyro_files = sorted(glob.glob(os.path.join(DATA_DIR, "hapt_data_set/RawData/gyro*.txt")))
    label_info = pd.read_table(
        os.path.join(DATA_DIR, "hapt_data_set/RawData/labels.txt"),
        sep=" ",
        header=None,
        names=["ExpID", "UserID", "ActID", "ActStart", "ActEnd"],
    )

    X_train, X_test = [], []
    y_train, y_test = np.array([]), np.array([])

    # Create features from each row data file
    for i in tqdm(range(len(acc_files))):
        acc_file, gyro_file = acc_files[i], gyro_files[i]
        exp_id = int(acc_file.split("exp")[1][:2])
        user_id = int(acc_file.split("user")[1][:2])

        temp_label_info = label_info[
            (label_info.ExpID == exp_id)
            & (label_info.UserID == user_id)
            & (label_info.ActID.isin(ATC_IDS))
        ]
        acc_raw = pd.read_table(acc_file, sep=" ", header=None, names=["x", "y", "z"])
        gyro_raw = pd.read_table(gyro_file, sep=" ", header=None, names=["x", "y", "z"])

        for _, _, act_id, act_start, act_end in temp_label_info.values:
            temp_acc_raw = acc_raw.iloc[act_start : act_end + 1]
            temp_gyro_raw = gyro_raw.iloc[act_start : act_end + 1]
            features = create_features(temp_acc_raw, temp_gyro_raw)  # Create features
            labels = [act_id] * len(features)

            if user_id in TRAIN_SUBJECTS:
                X_train.append(features)
                y_train = np.hstack((y_train, labels))
            else:
                X_test.append(features)
                y_test = np.hstack((y_test, labels))

    columns = get_feature_names()
    X_train = pd.DataFrame(np.vstack(X_train), columns=columns)
    X_test = pd.DataFrame(np.vstack(X_test), columns=columns)

    logger.debug(f"{X_train.shape=}, {X_test.shape=}")
    logger.debug(f"{y_train.shape=}, {y_test.shape=}")

    # Save features/labels to pickle/npy files
    X_train.to_pickle(os.path.join(DATA_DIR, "my_dataset/X_train.pickle"))
    X_test.to_pickle(os.path.join(DATA_DIR, "my_dataset/X_test.pickle"))
    np.save(os.path.join(DATA_DIR, "my_dataset/y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "my_dataset/y_test.npy"), y_test)


if __name__ == "__main__":
    main()
