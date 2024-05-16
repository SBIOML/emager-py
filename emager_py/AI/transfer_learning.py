import numpy as np
from matplotlib import pyplot as plt
from getData import getData_EMG
from sklearn.utils import shuffle
import csv

# Machine learning imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
# from tf.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
# Parameters
model_filename = 'model_felix_with_transfer'
nb_class = 5

# model parameters
nb_epoch = 10
batch_size = 80
model_name = 'model_felix_full.h5'
nb_rep = 3

# load data (6 x 3 x 64 x 5000)
data = getData_EMG("000","007",nb_repetition=nb_rep,transfer_learn=True)
bracelet_width = 4
bracelet_length = 16
nb_channels = bracelet_width * bracelet_length

# Takes out the mean (remove DC) + remove 60 Hz
data_shape = data.shape
print("data shape1:", data_shape)
data = data - np.reshape(np.repeat(np.mean(data,axis=3),data_shape[3],axis=2),data_shape)
# rectify the data (all positive)
data = np.absolute(data)

# moving average
window = 20
mask = (1 / window) * np.ones(window, dtype='float')
for gest in range(6):
    for rep in range(nb_rep):
        data_seq = data[gest, rep, :, :]
        for col in range(len(data_seq)):
            data_seq[col,:] = np.convolve(data_seq[col,:], mask, mode='same')
        data[gest, rep, :, :] = data_seq

# Delete the first #window who have bad averages and reshapes
data = data[:, :, :, 100:]
data = np.swapaxes(data,2,3)
print("data shape2:", data.shape)
# Create labels for gesture 0 to 5
labels = np.zeros(data.shape)
for gest in range(6):
    labels[gest, :, :, :] = gest
labels = labels[:, :, :, 0]
print("labels shape1:", labels.shape)
# Creates the training dataset
if nb_class == 4:
    X_train = np.concatenate((np.reshape(data[0,:,:,:],(1,10,4900,64)),data[3:,:,:,:]),axis=0)
    y_train = np.concatenate((np.reshape(labels[0,:,:],(1,10,4900)),labels[3:,:,:]),axis=0)
    y_train[y_train == 3] = 1
    y_train[y_train == 4] = 2
    y_train[y_train == 5] = 3
    print(X_train.shape)
    print(y_train.shape)
elif nb_class == 5:
    X_train = np.concatenate((data[0:2,:,:,:],data[3:,:,:,:]),axis=0)
    y_train = np.concatenate((labels[0:2,:,:],labels[3:,:,:]),axis=0)
    y_train[y_train == 3] = 2
    y_train[y_train == 4] = 3
    y_train[y_train == 5] = 4
    print(X_train.shape)
    print(y_train.shape)
else:
    X_train = data
    y_train = labels

# Reshape arrays: data --> (dataPts x 64)  labels (dataPts x 1)
X_train = np.reshape(X_train, (nb_class*nb_rep*len(X_train[0,0]), nb_channels))
y_train = (np.reshape(y_train, (nb_class*nb_rep*len(y_train[0,0]), 1))).astype(int)
print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)
# Create OneHot vectors for labels
# Train
matrix_one_hot = np.zeros((len(y_train), nb_class))
for i in range(len(y_train)):
    matrix_one_hot[i, y_train[i]] = 1
y_train = matrix_one_hot
print("y_train.shape", y_train.shape)
# Reshape the data to put into the neural network and Shuffle the values in a random order
X_train, y_train = shuffle(X_train, y_train)
print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], bracelet_width, 16, 1))
print(X_train.shape)

# Define the keras model
model = tf.keras.models.load_model(model_name)

# Freezes all layers except the last if Transfer Learning
for layer in model.layers:
    layer.trainable=False
model.layers[-1].trainable=True

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size) #, validation_split=0.1)
print("-----------------------------------------------------------------------")
print("Model done fitting")
print("-----------------------------------------------------------------------")

# evaluate the model
scores = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save the model
model.save(model_filename + ".h5")
#model.save(model_filename, save_format="tf")
print("Saved model to disk")
