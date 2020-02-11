# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from lgbm_optuna_stepwise import OptunaStepwise


def load_data():
    """
    Load dataset.
    The following six classes are included in this experiment.
        - WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
    The following transition classes are excluded.
        - STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, and LIE_TO_STAND
    Args:
    Returns:
        X_tin (DataFrame): 
        X_test (DataFrame):
        y_train (DataFrame): 
        y_test (DataFrame): 
        subject_id_train (array): subject_id for each record
        subject_id_test (array): subject_id for each record
        label2activity_dict (dict): key:label_id, value: title_of_class
        activity2label_dict (dict): key:title_of_class, value: label_id
    """
    root = '../../data/hapt_data_set/'
    features = pd.read_table(root + 'features.txt', header=None).values.flatten()
    features = np.array([feature.rstrip() for feature in features])

    X_train = pd.read_table(root + 'Train/X_train.txt', sep=' ', header=None, names=features)
    y_train = pd.read_table(root + 'Train/y_train.txt', sep=' ', header=None)
    subject_id_train = pd.read_table(root + 'Train/subject_id_train.txt', sep=' ', header=None)

    X_test = pd.read_table(root + 'Test/X_test.txt', sep=' ', header=None, names=features)
    y_test = pd.read_table(root + 'Test/y_test.txt', sep=' ', header=None)
    subject_id_test = pd.read_table(root + 'Test/subject_id_test.txt', sep=' ', header=None)

    activity_labels = pd.read_table(root + 'activity_labels.txt', header=None).values.flatten()
    activity_labels = np.array([label.rstrip().split() for label in activity_labels])
    label2activity_dict, activity2label_dict = {}, {}
    for label, activity in activity_labels:
        label2activity_dict[int(label)] = activity
        activity2label_dict[activity] = int(label)

    class_names_inc = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    class_ids_inc = [activity2label_dict[c] for c in class_names_inc]

    idx_train = y_train[y_train[0].isin(class_ids_inc)].index
    X_train = X_train.iloc[idx_train].reset_index(drop=True)
    y_train = y_train.iloc[idx_train].reset_index(drop=True)
    subject_id_train = subject_id_train.iloc[idx_train].reset_index(drop=True)

    idx_test = y_test[y_test[0].isin(class_ids_inc)].index
    X_test = X_test.iloc[idx_test].reset_index(drop=True)
    y_test = y_test.iloc[idx_test].reset_index(drop=True)
    subject_id_test = subject_id_test.iloc[idx_test].reset_index(drop=True)

    # Replace 6 to 0
    rep_activity = label2activity_dict[6]
    label2activity_dict[0] = rep_activity
    label2activity_dict.pop(6)
    activity2label_dict[rep_activity] = 0

    y_train = y_train.replace(6, 0)
    y_test = y_test.replace(6, 0)

    return X_train, X_test, y_train, y_test, subject_id_train, subject_id_test, label2activity_dict, activity2label_dict


X_train, X_test, y_train, y_test, subject_id_train, subject_id_test, label2activity_dict, activity2label_dict = load_data()
stepwise = OptunaStepwise(
    X_train, y_train, 
    objective='multiclass',
    num_class=6,
    model_selection='StratifiedKFold', 
    metric='log_loss', metric_lgbm='multi_logloss'
    )
stepwise.start_tuning()