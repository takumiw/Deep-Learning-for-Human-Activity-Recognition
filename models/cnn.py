# -*- coding:utf-8 -*-
import os, logging, sys
from datetime import datetime
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping


EXEC_TIME = 'cnn-' + datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=f'../logs/{EXEC_TIME}.log', level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.debug(f'../logs/{EXEC_TIME}.log')


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
    X_train = np.load(root + 'Raw_X_train.npy')
    y_train = pd.read_table(root + 'y_train.txt', sep=' ', header=None)
    
    # subject_id_train = pd.read_table(root + 'Train/subject_id_train.txt', sep=' ', header=None)

    X_test = np.load(root + 'Raw_X_test.npy')
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

    y_train = np_utils.to_categorical(y_train.values.reshape(len(y_train)), 6)
    y_test = np_utils.to_categorical(y_test.values.reshape(len(y_test)), 6)

    # return X_train, X_test, y_train, y_test, subject_id_train, subject_id_test, label2activity_dict, activity2label_dict
    return X_train, X_test, y_train, y_test, label2activity_dict, activity2label_dict


# X_train, X_test, y_train, y_test, subject_id_train, subject_id_test, label2activity_dict, activity2label_dict = load_data()
X_train, X_test, y_train, y_test, label2activity_dict, activity2label_dict = load_data()


for c, mode in zip([Counter(y_train.argmax(axis=1)), Counter(y_test.argmax(axis=1))], ['train', 'test']):
    print(f'{mode} samples')
    logging.debug(f'{mode} samples')
    for label_id in range(6):
        print(f'{label2activity_dict[label_id]} ({label_id}): {c[label_id]} ({c[label_id]/len(c):.04})', end='\t')
        logging.debug(f'{label2activity_dict[label_id]} ({label_id}): {c[label_id]} ({c[label_id]/len(c):.04})')
    print()


def define_model(window_size:int=128, output_dim:int=6) -> Model:
    print('--define model--')

    model = Sequential()
    model.add(Conv2D(70, kernel_size=(7, 1), input_shape=(window_size, 6, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.2, seed=0))
    model.add(Conv2D(70, kernel_size=(7, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3, seed=1))
    model.add(Conv2D(30, kernel_size=(5, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3, seed=2))
    model.add(Conv2D(30, kernel_size=(3, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1))) 
    model.add(Dropout(0.4, seed=3))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(
        loss=keras.losses.categorical_crossentropy, 
        # optimizer=keras.optimizers.Adam(lr=0.0001),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['accuracy']
        )
    return model


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0, shuffle=True, stratify=y_train)

print(f'X_train:{X_tr.shape} X_valid:{X_val.shape} X_test:{X_test.shape}')
print(f'y_train:{y_train.shape} y_valid:{y_val.shape} y_test:{y_test.shape}')
logging.debug(f'X_train:{X_tr.shape} X_valid:{X_val.shape} X_test:{X_test.shape}')
logging.debug(f'y_train:{y_train.shape} y_valid:{y_val.shape} y_test:{y_test.shape}')

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='min'))

model = define_model()
# モデルの構造のプロット図を保存
plot_model(model, to_file=f'{EXEC_TIME}_model.png', show_shapes=True)
fit = model.fit(
    X_tr, y_tr, 
    batch_size=32, 
    epochs=2000, 
    verbose=1, 
    validation_data = (X_val, y_val),
    callbacks=callbacks)

# 学習曲線をプロットする
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
axL.plot(fit.history['loss'], label="loss for training")
axL.plot(fit.history['val_loss'], label="loss for validation")
axL.set_title('model loss')
axL.set_xlabel('epoch')
axL.set_ylabel('loss')
axL.legend(loc='upper right')

axR.plot(fit.history['acc'], label="accuracy for training")
axR.plot(fit.history['val_acc'], label="accuracy for validation")
axR.set_title('model accuracy')
axR.set_xlabel('epoch')
axR.set_ylabel('accuracy')
axR.legend(loc='upper right')

fig.savefig(f'{EXEC_TIME}_hitsoty.png')
plt.close()


for X, y, mode in zip([X_tr, X_val, X_test], [y_tr, y_val, y_test], ['train', 'valid', 'test']):
    logloss, acc = model.evaluate(X, y)
    pred = model.predict(X).argmax(axis=1)
    y = y.argmax(axis=1)
    prec = precision_score(y, pred, average='macro')
    recall = recall_score(y, pred, average='macro')
    f1 = f1_score(y, pred, average='macro')

    print(mode)
    print(f'logloss {logloss}')
    print(f'accuracy {acc}')
    print(f'precision {prec}')
    print(f'recall {recall}')
    print(f'f1 {f1}')

    logging.debug(mode)
    logging.debug(f'logloss {logloss}')
    logging.debug(f'accuracy {acc}')
    logging.debug(f'precision {prec}')
    logging.debug(f'recall {recall}')
    logging.debug(f'f1 {f1}')

    print(f'confusion matrix\n{np.array2string(confusion_matrix(y, pred))}')
    logging.debug(f'confusion matrix\n{np.array2string(confusion_matrix(y, pred))}')
    