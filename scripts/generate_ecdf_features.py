# -*- coding:utf-8 -*-
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import glob
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from statsmodels.distributions.empirical_distribution import ECDF
from obtain_features import ObtainFeatures
ATC_IDS = [1,2,3,4,5,6]  # Target activity ids


# Create log file
EXEC_TIME = 'generate-ecdf-features-' + datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=f'../logs/{EXEC_TIME}.log', level=logging.DEBUG)
logging.debug(f'../logs/{EXEC_TIME}.log')


def create_features(acc_raw, gyro_raw):
    of = ObtainFeatures(acc_raw, gyro_raw)
    
    # Remove noises by median filter & Butterworth filter
    acc_raw = of.apply_median_filter(acc_raw)
    acc_raw = of.apply_butterworth_filter(acc_raw)
    gyro_raw = of.apply_median_filter(gyro_raw)
    gyro_raw = of.apply_butterworth_filter(gyro_raw)
    
    # Sample signals in fixed-width sliding windows
    tAccXYZ, tBodyGyroXYZ = of.segment_signal(acc_raw, gyro_raw)
    
    # Separate acceleration signal into body and gravity acceleration signal
    tBodyAccXYZ, tGravityAccXYZ = [], []
    for acc in tAccXYZ:
        body_acc, grav_acc = of.remove_gravity(acc)
        tBodyAccXYZ.append(body_acc)
        tGravityAccXYZ.append(grav_acc)

    features = np.empty((0, 60))  # Create 10 ecdf feature values for each triaxial acceleration and gyro signals.
    idx = np.linspace(0, tBodyAccXYZ[0].shape[0]-1, 10)  # Take 10 linspace percentile.
    idx = [int(Decimal(str(ix)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) for ix in idx]
    
    for i in range(len(tBodyAccXYZ)):
        feature_vector = np.array([])
        for axis in ['x', 'y', 'z']:
            ecdf = ECDF(tBodyAccXYZ[i][axis].values)  # fit
            x = ecdf.x[1:]  # Remove -inf
            feat = x[idx]
            feature_vector = np.hstack([feature_vector, feat])

        for axis in ['x', 'y', 'z']:
            ecdf = ECDF(tBodyGyroXYZ[i][axis].values)  # fit
            x = ecdf.x[1:]  # Remove -inf
            feat = x[idx]
            feature_vector = np.hstack([feature_vector, feat])
        features = np.vstack([features, feature_vector])
    
    return features


def main():
    logging.debug('Start creating ECDF features...')
    train_subjects = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    test_subjects = [2, 4, 9, 10, 12, 13, 18, 20, 24]
    
    root = '../data/hapt_data_set/RawData/'
    acc_files = sorted(glob.glob(root + 'acc*.txt'))
    gyro_files = sorted(glob.glob(root + 'gyro*.txt'))
    label_info = pd.read_table(root + 'labels.txt', sep=' ', header=None, names=['ExpID', 'UserID', 'ActID', 'ActStart', 'ActEnd'])
    
    X_train, X_test = np.empty((0, 60)), np.empty((0, 60))
    
    for acc_file, gyro_file in zip(acc_files, gyro_files):
        exp_id = int(acc_file.split('exp')[1][:2])
        user_id = int(acc_file.split('user')[1][:2])
        logging.debug(f'Exp ID: {exp_id},  User ID: {user_id}')
        temp_label_info = label_info[(label_info.ExpID==exp_id)&(label_info.UserID==user_id)&(label_info.ActID.isin(ATC_IDS))]
    
        acc_raw = pd.read_table(acc_file, sep=' ', header=None, names=['x', 'y', 'z'])
        gyro_raw = pd.read_table(gyro_file, sep=' ', header=None, names=['x', 'y', 'z'])
        
        for _, _, act_id, act_start, act_end in temp_label_info.values:
            temp_acc_raw = acc_raw.iloc[act_start:act_end+1]
            temp_gyro_raw = gyro_raw.iloc[act_start:act_end+1]
            features = create_features(temp_acc_raw, temp_gyro_raw)
            if user_id in train_subjects:
                X_train = np.vstack((X_train, features))
            else:
                X_test = np.vstack((X_test, features))
    
    columns = [f'tBody{sensor}ECDF-{axis}{i}' for sensor in ['Acc', 'Gyro'] for axis in ['X', 'Y', 'Z'] for i in range(10)]
    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)

    logging.debug(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    
    pd.to_pickle('../data/my_dataset/ECDF_X_train.pickle', X_train)
    pd.to_pickle('../data/my_dataset/ECDF_X_test.pickle', X_test)


if __name__ == '__main__':
    main()