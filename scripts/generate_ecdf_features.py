# -*- coding:utf-8 -*-
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import glob
from ecdf import ECDF
# from statsmodels.distributions.empirical_distribution import ECDF
from obtain_features import ObtainFeatures


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

    features = []
    
    # print('tBodyAccXYZ', len(tBodyAccXYZ))
    for i in range(len(tBodyAccXYZ)):
        # print(tBodyAccXYZ[i]['x'].values.reshape(-1, 1).shape)

        e = ECDF()
        accxyz = [tBodyAccXYZ[i]['x'].values, tBodyAccXYZ[i]['y'].values, tBodyAccXYZ[i]['z'].values]
        xyz_acc_features = e._do(accxyz)
        xyz_acc_features = xyz_acc_features.flatten()

        gyroxyz = [tBodyGyroXYZ[i]['x'].values, tBodyGyroXYZ[i]['y'].values, tBodyGyroXYZ[i]['z'].values]
        xyz_gyro_features = e._do(gyroxyz)
        xyz_gyro_features = xyz_gyro_features.flatten()

        feature_vector = np.hstack((xyz_acc_features, xyz_gyro_features))
        """
        ecdf = ECDF(tBodyAccXYZ[i]['x'].values)  # fit
        xnew = np.linspace(ecdf.x[1], ecdf.x[-2], 20)
        x_acc_features = ecdf(xnew)

        ecdf = ECDF(tBodyAccXYZ[i]['y'].values)  # fit
        xnew = np.linspace(ecdf.x[1], ecdf.x[-2], 20)
        y_acc_features = ecdf(xnew)

        ecdf = ECDF(tBodyAccXYZ[i]['z'].values)  # fit
        xnew = np.linspace(ecdf.x[1], ecdf.x[-2], 20)
        z_acc_features = ecdf(xnew)

        ecdf = ECDF(tBodyGyroXYZ[i]['x'].values)  # fit
        xnew = np.linspace(ecdf.x[1], ecdf.x[-2], 20)
        x_gyro_features = ecdf(xnew)

        ecdf = ECDF(tBodyGyroXYZ[i]['y'].values)  # fit
        xnew = np.linspace(ecdf.x[1], ecdf.x[-2], 20)
        y_gyro_features = ecdf(xnew)
        
        ecdf = ECDF(tBodyGyroXYZ[i]['z'].values)  # fit
        xnew = np.linspace(ecdf.x[1], ecdf.x[-2], 20)
        z_gyro_features = ecdf(xnew)
        """
        # feature_vector = np.hstack((x_acc_features, y_acc_features, z_acc_features, x_gyro_features, y_gyro_features, z_gyro_features))
        features.append(feature_vector)
    
    features = np.array(features)
    return features

def create_features_all():
    logging.debug('Start creating ECDF features')
    
    train_subjects = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    test_subjects = [2, 4, 9, 10, 12, 13, 18, 20, 24]
    
    root = '../data/hapt_data_set/RawData/'
    acc_files = sorted(glob.glob(root + 'acc*.txt'))
    gyro_files = sorted(glob.glob(root + 'gyro*.txt'))
    
    label_info = pd.read_table(root + 'labels.txt', sep=' ', header=None, names=['ExpID', 'UserID', 'ActID', 'ActStart', 'ActEnd'])
    
    X_train = np.array([])
    X_test = np.array([])
    # y_train = np.array([])
    # y_test = np.array([])
    
    # for acc_file, gyro_file in zip(acc_files, gyro_files):
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
            """
            
            # labels = [act_id] * len(features)
            """
            if user_id in train_subjects:
                if len(X_train) == 0:
                    X_train = features
                else:
                    X_train = np.vstack((X_train, features))
                """if len(y_train) == 0:
                    y_train = labels
                else:
                    y_train = np.hstack((y_train, labels))"""
            else:
                if len(X_test) == 0:
                    X_test = features
                else:
                    X_test = np.vstack((X_test, features))
                """if len(y_test) == 0:
                    y_test = labels
                else:
                    y_test = np.hstack((y_test, labels))"""
    
    logging.debug(f'X_train {X_train.shape}')
    logging.debug(f'X_test {X_test.shape}')
    # logging.debug(f'y_train {y_train.shape}')
    # logging.debug(f'y_test {y_test.shape}')
    
    np.savetxt('../data/my_dataset/ECDF_Iwasawa_X_train.txt', X_train)
    np.savetxt('../data/my_dataset/ECDF_Iwasawa_X_test.txt', X_test)
    # np.savetxt('../data/my_dataset/ECDF_y_train.txt', y_train)
    # np.savetxt('../data/my_dataset/y_test.txt', y_test)


if __name__ == '__main__':
    create_features_all()