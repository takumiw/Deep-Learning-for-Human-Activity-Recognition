# -*- coding:utf-8 -*-
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import glob
from create_features import create_features


# Create log file
EXEC_TIME = 'generate-features-of-hapt-' + datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=f'../logs/{EXEC_TIME}.log', level=logging.DEBUG)
logging.debug(f'../logs/{EXEC_TIME}.log')


def create_features_all():
    logging.debug('Start creating features')
    
    train_subjects = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    test_subjects = [2, 4, 9, 10, 12, 13, 18, 20, 24]
    
    root = '../data/hapt_data_set/RawData/'
    acc_files = sorted(glob.glob(root + 'acc*.txt'))
    gyro_files = sorted(glob.glob(root + 'gyro*.txt'))
    
    label_info = pd.read_table(root + 'labels.txt', sep=' ', header=None, names=['ExpID', 'UserID', 'ActID', 'ActStart', 'ActEnd'])
    
    X_train = np.array([])
    X_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])
    
    for acc_file, gyro_file in zip(acc_files, gyro_files):
        exp_id = int(acc_file.split('exp')[1][:2])
        user_id = int(acc_file.split('user')[1][:2])
        
        print(f'User ID: {user_id}')
        
        temp_label_info = label_info[(label_info.ExpID==exp_id)&(label_info.UserID==user_id)&(label_info.ActID.isin([1,2,3,4,5,6]))]
    
        acc_raw = pd.read_table(acc_file, sep=' ', header=None, names=['x', 'y', 'z'])
        gyro_raw = pd.read_table(gyro_file, sep=' ', header=None, names=['x', 'y', 'z'])
        
        for _, _, act_id, act_start, act_end in temp_label_info.values:
            temp_acc_raw = acc_raw.iloc[act_start:act_end+1]
            temp_gyro_raw = gyro_raw.iloc[act_start:act_end+1]
            features = create_features(temp_acc_raw, temp_gyro_raw)
            labels = [act_id] * len(features)
            
            if user_id in train_subjects:
                if len(X_train) == 0:
                    X_train = features
                else:
                    X_train = np.vstack((X_train, features))
                if len(y_train) == 0:
                    y_train = labels
                else:
                    y_train = np.hstack((y_train, labels))
            else:
                if len(X_test) == 0:
                    X_test = features
                else:
                    X_test = np.vstack((X_test, features))
                if len(y_test) == 0:
                    y_test = labels
                else:
                    y_test = np.hstack((y_test, labels))
    
    logging.debug(f'X_train {X_train.shape}')
    logging.debug(f'X_test {X_test.shape}')
    logging.debug(f'y_train {y_train.shape}')
    logging.debug(f'y_test {y_test.shape}')
    
    np.savetxt('../data/my_dataset/X_train.txt', X_train)
    np.savetxt('../data/my_dataset/X_test.txt', X_test)
    np.savetxt('../data/my_dataset/y_train.txt', y_train)
    np.savetxt('../data/my_dataset/y_test.txt', y_test)


if __name__ == '__main__':
    create_features_all()