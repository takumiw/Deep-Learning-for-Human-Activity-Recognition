# -*- coding:utf-8 -*-
import os, logging, sys
from datetime import datetime
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import lightgbm as lgb

sys.path.append('../')
from logs.logger import log_evaluation
EXEC_TIME = 'ecdf-lgbm-' + datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=f'../logs/{EXEC_TIME}.log', level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.debug(f'../logs/{EXEC_TIME}.log')


with open('../configs/default.json', 'r') as f:
    d = json.load(f)
    params = d['lgbm_params']

logging.debug(f'params: {params}')


def load_data():
    """
    Load dataset.
    The following six classes are included in this experiment.
        - WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
    The following transition classes are excluded.
        - STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, and LIE_TO_STAND
    Args:
    Returns:
        X_train (DataFrame): 
        X_test (DataFrame):
        y_train (DataFrame): 
        y_test (DataFrame): 
        subject_id_train (array): subject_id for each record
        subject_id_test (array): subject_id for each record
        label2activity_dict (dict): key:label_id, value: title_of_class
        activity2label_dict (dict): key:title_of_class, value: label_id
    """
    root = '../data/my_dataset/'
    # features = pd.read_table(root + 'features.txt', header=None).values.flatten()
    # features = np.array([feature.rstrip() for feature in features])

    X_train = pd.read_table(root + 'ECDF_Iwasawa_X_train.txt', sep=' ', header=None)
    y_train = pd.read_table(root + 'y_train.txt', sep=' ', header=None)
    # subject_id_train = pd.read_table(root + 'Train/subject_id_train.txt', sep=' ', header=None)

    X_test = pd.read_table(root + 'ECDF_Iwasawa_X_test.txt', sep=' ', header=None)
    y_test = pd.read_table(root + 'y_test.txt', sep=' ', header=None)
    # subject_id_test = pd.read_table(root + 'Test/subject_id_test.txt', sep=' ', header=None)

    activity_labels = pd.read_table('../data/hapt_data_set/activity_labels.txt', header=None).values.flatten()
    activity_labels = np.array([label.rstrip().split() for label in activity_labels])
    label2activity_dict, activity2label_dict = {}, {}
    for label, activity in activity_labels:
        label2activity_dict[int(label)] = activity
        activity2label_dict[activity] = int(label)

    class_names_inc = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    class_ids_inc = [activity2label_dict[c] for c in class_names_inc]

    # idx_train = y_train[y_train[0].isin(class_ids_inc)].index
    # X_train = X_train.iloc[idx_train].reset_index(drop=True)
    # y_train = y_train.iloc[idx_train].reset_index(drop=True)
    # subject_id_train = subject_id_train.iloc[idx_train].reset_index(drop=True)

    # idx_test = y_test[y_test[0].isin(class_ids_inc)].index
    # X_test = X_test.iloc[idx_test].reset_index(drop=True)
    # y_test = y_test.iloc[idx_test].reset_index(drop=True)
    # subject_id_test = subject_id_test.iloc[idx_test].reset_index(drop=True)

    # Replace 6 to 0
    rep_activity = label2activity_dict[6]
    label2activity_dict[0] = rep_activity
    label2activity_dict.pop(6)
    activity2label_dict[rep_activity] = 0

    y_train = y_train.replace(6, 0)
    y_test = y_test.replace(6, 0)

    # return X_train, X_test, y_train, y_test, subject_id_train, subject_id_test, label2activity_dict, activity2label_dict
    return X_train, X_test, y_train, y_test, label2activity_dict, activity2label_dict


# X_train, X_test, y_train, y_test, subject_id_train, subject_id_test, label2activity_dict, activity2label_dict = load_data()
X_train, X_test, y_train, y_test, label2activity_dict, activity2label_dict = load_data()

print(f'X_train:{X_train.shape} X_test:{X_test.shape}')
print(f'y_train:{y_train.shape} y_test:{y_test.shape}')
logging.debug(f'X_train:{X_train.shape} X_test:{X_test.shape}')
logging.debug(f'y_train:{y_train.shape} y_test:{y_test.shape}')

for c, mode in zip([Counter(y_train.values.flatten()), Counter(y_test.values.flatten())], ['train', 'test']):
    print(f'{mode} samples')
    logging.debug(f'{mode} samples')
    for label_id in range(6):
        print(f'{label2activity_dict[label_id]} ({label_id}): {c[label_id]} ({c[label_id]/len(c):.04})', end='\t')
        logging.debug(f'{label2activity_dict[label_id]} ({label_id}): {c[label_id]} ({c[label_id]/len(c):.04})')
    print()

# Split data by preserving the percentage of samples for each class.
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
valid_preds = np.zeros((X_train.shape[0], 6))
test_preds = np.zeros((n_splits, X_test.shape[0], 6))
importances = np.zeros((n_splits, X_train.shape[1]))
score_list = {
    'logloss': {'train': [], 'valid': [], 'test': []},
    'accuracy': {'train': [], 'valid': [], 'test': []},
    'precision': {'train': [], 'valid': [], 'test': []},
    'recall': {'train': [], 'valid': [], 'test': []},
    'f1': {'train': [], 'valid': [], 'test': []},
    }

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train.loc[train_index]
    y_val = y_train.loc[valid_index]

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=50)]
    
    model = lgb.train(
            params, lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            verbose_eval=50,
            num_boost_round=50000,
            early_stopping_rounds=50,
            callbacks=callbacks
        )
    
    logging.debug(f'best iteration: {model.best_iteration}')
    logging.debug(f'training best score: {model.best_score["training"].items()}')
    logging.debug(f'valid_1 best score: {model.best_score["valid_1"].items()}')

    pred_tr = model.predict(X_tr, num_iteration=model.best_iteration)
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    valid_preds[valid_index] = pred_val
    test_preds[fold_id] = pred_test
    importances[fold_id] = model.feature_importance()

    score_list['logloss']['train'].append(model.best_score['training']['multi_logloss'])
    score_list['logloss']['valid'].append(model.best_score['valid_1']['multi_logloss'])
    score_list['logloss']['test'].append(log_loss(y_test, pred_test))

    for pred, y, mode in zip([pred_tr, pred_val, pred_test], [y_tr, y_val, y_test], ['train', 'valid', 'test']):
        pred = pred.argmax(axis=1)
        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, average='macro')
        recall = recall_score(y, pred, average='macro')
        f1 = f1_score(y, pred, average='macro')

        score_list['accuracy'][mode].append(acc)
        score_list['precision'][mode].append(prec)
        score_list['recall'][mode].append(recall)
        score_list['f1'][mode].append(f1)

        print(f'{mode} confusion matrix\n{np.array2string(confusion_matrix(y, pred))}')
        logging.debug(f'{mode} confusion matrix\n{np.array2string(confusion_matrix(y, pred))}')
    

print('---Cross Validation Scores---')
for mode in ['train', 'valid', 'test']:
    print(f'---{mode}---')
    logging.debug(f'---{mode}---')
    for metric in ['logloss', 'accuracy', 'precision', 'recall', 'f1']:
        print(f'{metric}\t{np.mean(score_list[metric][mode])}\t{score_list[metric][mode]}')
        logging.debug(f'{metric}\t{np.mean(score_list[metric][mode])}\t{score_list[metric][mode]}')


# Plot feature importance
importance = np.mean(importances, axis=0)
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Value': importance})

plt.figure(figsize=(16, 10))
sns.barplot(x="Value", y="Feature", data=importance_df.sort_values(by="Value", ascending=False)[:30])
plt.title('LightGBM Top 30 Feature Importance (avg over folds)')
plt.tight_layout()
plt.savefig(f'../features/importances/{EXEC_TIME}.png')

os.makedirs(f'../data/pred/{EXEC_TIME}', exist_ok=True)
np.save(f'../data/pred/{EXEC_TIME}/valid_oof.npy', valid_preds)

test_preds = np.mean(test_preds, axis=0)  # Averaging
np.save(f'../data/pred/{EXEC_TIME}/test_oof.npy', test_preds)