#fix the error unable to download dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import shap

def Cifar10CNN(Optimizerr, Loss_function, Learning_rate, Batch_size, Regularization):
    # load cifar10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #CNN model
    #fix number of epochs = 10 for best accuracy tested
    epochs = 10
    #build model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Flatten())
    if Regularization == 'none':
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(2048, activation='relu'))
    elif Regularization == 'l1':
        model.add(Dense(2048, kernel_regularizer=tf.keras.regularizers.l1(0.2), activation='relu'))
        model.add(Dense(2048, kernel_regularizer=tf.keras.regularizers.l1(0.2), activation='relu'))
    else:
        model.add(Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu'))
        model.add(Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    if Optimizerr == 'Adam':
        model.compile(loss=Loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_rate),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    elif Optimizerr == 'SGD':
        model.compile(loss=Loss_function, optimizer=tf.keras.optimizers.SGD(learning_rate=Learning_rate),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    else:
        model.compile(loss=Loss_function, optimizer=tf.keras.optimizers.RMSprop(learning_rate=Learning_rate),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()

    cifar10_train = model.fit(x_train, y_train, batch_size=Batch_size,epochs=epochs,validation_data=(x_test, y_test))
    test_eval = model.evaluate(x_test, y_test, verbose=0)
   
    return test_eval


if __name__ == "__main__":
    #The hyperparameters that produces best accuracy
    Optimizerr = 'Adam'
    Loss_function = 'sparse_categorical_crossentropy'
    Learning_rate = 0.001
    Batch_size = 256
    Regularization = 'none'
    #file1 = open("Result1.txt","a")
    #L = [sys.argv[2], ' ', sys.argv[3], ' ', sys.argv[4], ' ', sys.argv[5], ' ', sys.argv[1], '\n'] 
    #file1.writelines(L) 
    #print(Optimizerr, Loss_function, Learning_rate, Batch_size)
    res = Cifar10CNN(Optimizerr, Loss_function, Learning_rate, Batch_size, Regularization)
    #L = ['Test loss:', str(res[0]),'\n']
    #file1.writelines(L)
    #L = ['Test accuracy:', str(res[1]),'\n']
    #file1.writelines(L)
    #file1.writelines('\n')
    print('Test loss:', res[0])
    print('Test accuracy:', res[1])
    #file1.close()
      