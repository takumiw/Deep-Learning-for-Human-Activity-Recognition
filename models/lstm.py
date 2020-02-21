# -*- coding:utf-8 -*-
import os, logging, sys
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Suppress warnings 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

EXEC_TIME = 'lstm-' + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f'../logs/{EXEC_TIME}-history', exist_ok=True)

logging.basicConfig(filename=f'../logs/{EXEC_TIME}-history/{EXEC_TIME}.log', level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.debug(f'../logs/{EXEC_TIME}-history/{EXEC_TIME}.log')


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
        label2activity_dict (dict): key:label_id, value: title_of_class
        activity2label_dict (dict): key:title_of_class, value: label_id
    """
    root = '../data/my_dataset/'
    X_train = np.load(root + 'Raw_X_train.npy')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    y_train = pd.read_table(root + 'y_train.txt', sep=' ', header=None)

    X_test = np.load(root + 'Raw_X_test.npy')
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    y_test = pd.read_table(root + 'y_test.txt', sep=' ', header=None)

    activity_labels = pd.read_table('../data/hapt_data_set/activity_labels.txt', header=None).values.flatten()
    activity_labels = np.array([label.rstrip().split() for label in activity_labels])
    label2activity_dict, activity2label_dict = {}, {}
    for label, activity in activity_labels:
        label2activity_dict[int(label)] = activity
        activity2label_dict[activity] = int(label)

    class_names_inc = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    class_ids_inc = [activity2label_dict[c] for c in class_names_inc]

    # Replace 6 to 0
    rep_activity = label2activity_dict[6]
    label2activity_dict[0] = rep_activity
    label2activity_dict.pop(6)
    activity2label_dict[rep_activity] = 0

    y_train = y_train.replace(6, 0)
    y_test = y_test.replace(6, 0)

    y_train = np_utils.to_categorical(y_train.values.reshape(len(y_train)), 6)
    y_test = np_utils.to_categorical(y_test.values.reshape(len(y_test)), 6)

    return X_train, X_test, y_train, y_test, label2activity_dict, activity2label_dict


X_train, X_test, y_train, y_test, label2activity_dict, activity2label_dict = load_data()

for c, mode in zip([Counter(y_train.argmax(axis=1)), Counter(y_test.argmax(axis=1))], ['train', 'test']):
    print(f'{mode} samples')
    logging.debug(f'{mode} samples')
    for label_id in range(6):
        print(f'{label2activity_dict[label_id]} ({label_id}): {c[label_id]} ({c[label_id]/len(c):.04})', end='\t')
        logging.debug(f'{label2activity_dict[label_id]} ({label_id}): {c[label_id]} ({c[label_id]/len(c):.04})')
    print()


def create_model(window_size:int=128, output_dim:int=6) -> Model:
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 6)))  # 64次元のベクトルのsequenceを出力する
    # model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(window_size, 6)))
    model.add(Dropout(0.3, seed=0))
    model.add(LSTM(64, return_sequences=True)) # 32次元のベクトルのsequenceを出力する
    model.add(Dropout(0.4, seed=1))
    model.add(LSTM(32))  # 32次元のベクトルを一つ出力する
    model.add(Dropout(0.5, seed=2))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, 
                # optimizer=keras.optimizers.RMSprop(lr=0.001),  # default lr=0.001
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])

    return model


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=71, shuffle=True, stratify=y_train)

print(f'X_train:{X_tr.shape} X_valid:{X_val.shape} X_test:{X_test.shape}')
print(f'y_train:{y_tr.shape} y_valid:{y_val.shape} y_test:{y_test.shape}')
logging.debug(f'X_train:{X_tr.shape} X_valid:{X_val.shape} X_test:{X_test.shape}')
logging.debug(f'y_train:{y_tr.shape} y_valid:{y_val.shape} y_test:{y_test.shape}')

# callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='min')
fn = f'../logs/{EXEC_TIME}-history/trained_model.h5'
model_checkpoint = ModelCheckpoint(filepath=fn, save_best_only=True)
reduce_lr_on_plateau = ReduceLROnPlateau(factor=0.2, patience=5, verbose=1, min_lr=0.000001)
callbacks = [early_stopping, model_checkpoint, reduce_lr_on_plateau]

model = create_model()

# Save plot of model architecture
plot_model(model, to_file=f'../logs/{EXEC_TIME}-history/model.png', show_shapes=True)

fit = model.fit(
    X_tr, y_tr, 
    batch_size=32, 
    epochs=2000, 
    verbose=2, 
    validation_data=(X_val, y_val),
    callbacks=callbacks)

# Plot learning curve
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

fig.savefig(f'../logs/{EXEC_TIME}-history/history.png')
plt.close()

# Logging training history of every 10 epochs
df = pd.DataFrame(fit.history)
index = np.arange(0, len(df), 10)
logging.debug(f'\n{df.iloc[index]}')
logging.debug(f'Early stopping at {len(df)-1}th epoch')

# ベストのモデルをロードする
model = load_model(fn)


for X, y, mode in zip([X_tr, X_val, X_test], [y_tr, y_val, y_test], ['train', 'valid', 'test']):
    logloss, acc = model.evaluate(X, y)
    pred = model.predict(X).argmax(axis=1)
    y = y.argmax(axis=1)
    prec = precision_score(y, pred, average='macro')
    recall = recall_score(y, pred, average='macro')
    f1 = f1_score(y, pred, average='macro')

    logging.debug(mode)
    logging.debug(f'logloss {logloss}')
    logging.debug(f'accuracy {acc}')
    logging.debug(f'precision {prec}')
    logging.debug(f'recall {recall}')
    logging.debug(f'f1 {f1}')

    logging.debug(f'confusion matrix\n{np.array2string(confusion_matrix(y, pred))}')
    