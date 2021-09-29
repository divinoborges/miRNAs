import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
# from keras.models import Model
from keras.layers import Input, Concatenate, add
from keras import optimizers

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def train_cnn(model, x_training, y_training, x_validation, y_validation):
    batch_size = 16
    model_filepath = "Results/teste_1_best.hdf5"
    if not os.path.exists('Results'):
        os.makedirs('Results')
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=0, save_best_only=True, mode='min')

    hist = model.fit(x_training, y_training, batch_size=batch_size, epochs=50, verbose=1, callbacks=[checkpointer],
                     validation_data=(x_validation, y_validation))


def build_cnn(x_training):
    inputs = Input(shape=(x_training.shape[1], x_training.shape[2], 1))

    conv1 = Conv2D(8, 3, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(16, 3, padding='same', activation='relu')(conv1)
    pool1 = GlobalAveragePooling2D()(conv2)
    dense1 = Dense(64, activation='relu')(pool1)
    drop1 = Dropout(0.1)(dense1)

    outputs = Dense(1, activation='relu')(drop1)

    model = tf.keras.Model(inputs, outputs)

    # print (model.summary())

    # custom_adam = optimizers.Adam(lr=0.0001)  # , decay=0.00003, amsgrad=True)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['acc', 'mae'])

    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    return model


def splitting_data(arr_3d, classes):
    x_training, x_validation, y_training, y_validation = train_test_split(arr_3d,
                                                                          classes,
                                                                          test_size=0.1,
                                                                          random_state=5)
    x_training = np.expand_dims(x_training, axis=3)
    x_validation = np.expand_dims(x_validation, axis=3)
    return x_training, x_validation, y_training, y_validation


def reshaping_dataframe(dataframe):
    x = dataframe[['FILE', 'MIRNA', 'READ', 'READS']].values
    y = dataframe['classe'].values

    arr_3d = x.reshape((552, 1881, 4))
    classe_red = y.reshape((552, 1881))
    classes = []
    for i in classe_red:
        classes.append(i[0])
    classes = np.array(classes)

    return arr_3d, classes


def build_dataframe():
    dataframe = pd.read_csv('mirnas.csv', sep=',', thousands=',', engine='python')
    print(dataframe.head())
    dataframe['classe'] = pd.to_numeric(dataframe['classe'])
    return dataframe


def main():
    dataframe = build_dataframe()
    arr_3d, classes = reshaping_dataframe(dataframe)

    splitting_data(arr_3d, classes)
    x_training, x_validation, y_training, y_validation = splitting_data(arr_3d, classes)

    model = build_cnn(x_training)
    train_cnn(model, x_training, y_training, x_validation, y_validation)

    accuracy = model.evaluate(x_validation, y_validation)
    print("Acuracia", accuracy)

    print("Gerando predicao para 3 exemplos")
    predictions = model.predict(x_validation[:3])
    print("shape:", predictions.shape)


if __name__ == '__main__':
    main()
