import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, MaxPool1D, MaxPool2D, MaxPool3D
import time
import constants

from tensorflow.python.keras.backend import constant


def load_img_data(path, indx=None, filename='masked.pickle'):
    bacs = []
    dirs = os.listdir(path)
    if indx != None:
        dirs = np.array(dirs)[indx]
    for i in dirs:
        dirpath = os.path.join(path, i)
        files = os.listdir(dirpath)
        for j in files:
            if j == filename:
                pickle_in = open(os.path.join(dirpath, j), "rb")
                bacs.append(pickle.load(pickle_in))
    return bacs


def create_dataset_labels(bacs):
    dataset = []
    for ind, val in enumerate(bacs):
        dataset.append([val, ind])
    return dataset


def concat_lines_on_datasets(x, y, bac, cl: int):
    if (x.shape[0] > 0):
        x = np.concatenate((x, bac))
    else:
        x = bac

    y = np.append(y, [cl]*len(bac))
    return (x, y)


def create_one_hot_encoding(dataset, nbacs):
    y = np.zeros((dataset.shape[0], nbacs))
    for i in range(len(dataset)):
        y[i, int(dataset[i])] = 1
    return y


def shuffle_dataset(x, y):
    seed = np.random.randint(0, 1000*1000)
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return (x, y)


def split_xy(dataset):
    x = np.array([])
    y = np.array([], dtype=np.uint8)
    for f, l in dataset:
        x, y = concat_lines_on_datasets(x, y, f, l)
    return (x, y)
    # return (x, create_one_hot_encoding(y, N_BACS))


def split_xy(dataset, percentage):
    x1 = np.array([])
    y1 = np.array([], dtype=np.uint8)
    x2 = np.array([])
    y2 = np.array([], dtype=np.uint8)
    for bac, index in dataset:
        half = int(bac.shape[0]*percentage)
        x1, y1 = concat_lines_on_datasets(x1, y1, bac[:half, :], index)
        x2, y2 = concat_lines_on_datasets(x2, y2, bac[half:, :], index)
    return (x1, y1, x2, y2)

def predict_on_bac(model, bac_mat):
    ypred = model.predict(bac_mat, batch_size=1800)
    return np.argmax(ypred, axis=1)


def get_callbacks(tensorDir=None, name=None):
    cc = []
    cc.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20))
    if tensorDir is not None:
        cc.append(TensorBoard(log_dir='{}/{}'.format(tensorDir, name)))
    return cc


def mlp_compile_and_fit(samples: tuple, layers: tuple, act='relu', tensorBoardDir=None, savePath=None, showSummary=False):
    assert samples is not None and layers is not None
    xtrain, ytrain, xtest, ytest = samples

    n_features = xtrain.shape[1]
    model_name = 'mlp'
    mlp_model = Sequential([
        Flatten(input_shape=(n_features, 1)),
    ])
    for i in layers:
        model_name += '-{}d-{}'.format(i, act)
        mlp_model.add(Dense(i, activation=act))

    NAME = '{}-{}'.format(model_name,
                          int(time.time()))

    mlp_model.add(Dense(constants.N_BACS, activation='softmax'))

    mlp_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(name='scc')])
    if showSummary:
        mlp_model.summary()

    mlp_history = mlp_model.fit(xtrain, ytrain, batch_size=1500, epochs=200, validation_data=(
        xtest, ytest), shuffle=True, callbacks=[get_callbacks(tensorBoardDir, NAME)])

    mlp_model.custom_name = model_name
    if savePath is not None:
        mlp_model.save(os.path.join(savePath, model_name))
    return (mlp_model, mlp_history)


def cnn_compile_and_fit(samples: tuple, conv_layers: tuple, dense_layers: tuple, act_dense='relu',
                        between_layers: tuple = None, tensorDir=None, savePath=None, showSummary=False):

    assert samples is not None and conv_layers is not None and dense_layers is not None

    xtrain, ytrain, xtest, ytest = samples
    n_features = xtrain.shape[1]

    model_name = 'cnn'
    cnn_model = Sequential()
    for i in range(len(conv_layers)):
        fs, ks, ps = conv_layers[i]
        model_name += '-{}x{}x{}'.format(fs, ks, ps)
        if i == 0:
            cnn_model.add(Conv1D(filters=fs, kernel_size=ks, activation='relu',
                                 input_shape=(n_features, 1)))
        else:
            cnn_model.add(
                Conv1D(filters=fs, kernel_size=ks, activation='relu'))

        if ps != -1:
            cnn_model.add(MaxPool1D(pool_size=ps))

        if between_layers is not None:
            for k in between_layers:
                cnn_model.add(k)

    cnn_model.add(Flatten())
    for i in dense_layers:
        model_name += '-{}d-{}'.format(i, act_dense)
        cnn_model.add(Dense(i, activation=act_dense))

    cnn_model.add(Dense(constants.N_BACS, activation='softmax'))

    print(n_features, model_name)

    NAME = '{}-{}'.format(model_name,
                          int(time.time()))

    cnn_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy(name='scc')])
    if showSummary:
        cnn_model.summary()

    cnn_history = cnn_model.fit(xtrain, ytrain, batch_size=1800, epochs=500, validation_data=(
        xtest, ytest), shuffle=True, callbacks=[get_callbacks(tensorDir, NAME)])

    cnn_model.custom_name = model_name
    if savePath is not None:
        cnn_model.save(os.path.join(savePath, model_name))
    return (cnn_model, cnn_history)
