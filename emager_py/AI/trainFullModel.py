import numpy as np
from matplotlib import pyplot as plt
from getData import getData_EMG
from sklearn.utils import shuffle
from collections import Counter
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
model_filename = 'model_felix_full'
nb_class = 5

# model parameters
nb_epoch = 10
batch_size = 80
trans_learn = False
model_name = 'model_full.h5'
if trans_learn:
    nb_rep = 3
else:
    nb_rep = 50

# load data (6 x 10 x 64 x 5000)
data = np.zeros((6,50,64,5000))
#data = np.zeros((6,10,64,5000))
data[:,0:10,:,:] = getData_EMG("000","000")
data[:,10:20,:,:] = getData_EMG("000","001")
data[:,20:30,:,:] = getData_EMG("000","002")
data[:,30:40,:,:] = getData_EMG("000","003")
data[:,40:50,:,:] = getData_EMG("000","004")
bracelet_width = 4
bracelet_length = 16
nb_channels = bracelet_width * bracelet_length

print("Max is!!!! ", np.max(np.absolute(data)))
plt.bar()
quit()
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
if trans_learn:
    model = tf.keras.models.load_model(model_name)
else:
    model = Sequential()
    # model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), input_shape=(4, 16, 1), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))  # Selu activation function
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5, input_shape=(1024, 1)))
    model.add(Dense(256, input_dim=1024, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))

# Freezes all layers except the last if Transfer Learning
if trans_learn:
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
model.save(model_filename, save_format='tf')
print("Saved model to disk")

# Save the min and max for real time application
# csv_filename = model_filename + ".csv"
# with open(csv_filename, 'w', newline='') as f:
#     # create the csv writer
#     writer = csv.writer(f)
#     # write rows to the csv file
#     writer.writerows(min_and_max)
#     # close the file
#     f.close()

# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.gray_r, savefile=False, filepath=None):
#     """
#         *** Taken from source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py ***
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     # classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     #     print("Normalized confusion matrix")
#     # else:
#     #     print('Confusion matrix, without normalization')
#     #
#     # print(cm)
#
#     if savefile:
#         np.savetxt(filepath, cm, delimiter=',', fmt='%s')
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax