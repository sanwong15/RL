# Behavioural Cloning :
# User drive on their own, while the system start recording the behaviour and sensor data
# The car will then start to learn from the recorded data

import os
import csv
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda

from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint

from keras.models import model_from_json

import tensorflow as tf


def run():
    # Step 1: Sample Gathering (define empty sample array)
    samples = []

    # Step 2: Add sample from data
    samples = add_to_samples('data-udacity/driving_log.csv', samples)
    samples = add_to_samples('data-recovery-annie/driving_log.csv', samples)
    samples = samples[1:]

    print("Length of samples: ", len(samples))

    # Step 3: Split the samples into training and validation
    train_samples, validation_samples = train_test_split(samples, test_size = 0.1)

    # Step 4: Get the X and Y from samples with Generator
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # Step 5: Create the model
    model = behaviouralCloneModel()

    # Step 6: Start the training
    trainModel(model, train_generator, validation_generator)











def add_to_samples(csv_filepath, samples):
    # Take in csv log data
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


def generator(samples, batch_size = 32):
    num_samples = len(samples)

    while 1: # Loop forever, Generator never end
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # Getting batch sample of batch size
            batch_samples = samples[offset: offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # TODO: name has to be updated
                name = './data-udacity/' + batch_sample[0]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train) #yield return a generator


# Data preprocessing functions
def resize_comma(image):
    return tf.image.resize_images(image, 40, 160)

def behaviouralCloneModel():
    # Construct a model, Return model
    model = Sequential()

    # INPUT LAYER 0
    # (0.1) Crop 70 pixels from the top of the image and 25 from the bottom
    model.add(Cropping2D(cropping = ((70, 25), (0, 0)), dim_ordering='tf', input_shaoe=(160, 320, 3)))

    # (0.2) Resize the data
    model.add(Lambda(resize_comma))

    # (0.3) Normalise the data
    model.add(Lambda(lambda  x: (x/255.0) - 0.5))

    # CONV LAYER 1
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())

    # CONV LAYER 2
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # CONV LAYER 3
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))

    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())


    # FULLY CONNECTED LAYER 1
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())

    # FULLY CONNECTED LAYER 2
    model.add(Dense(50))
    model.add(ELU())

    model.add(Dense(1))

    adam = Adam(lr=0.0001)

    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

    print("Model summary:\n", model.summary())

    return model


def trainModel(model, train_generator, validation_generator):
    nb_epoch = 20

    # Create checkpoint to save model weights after each epoch
    checkpointer = ModelCheckpoint(filepath="./tmp/v2-weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1,
                                   save_best_only=False)


    # Train Model using Generator
    model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=nb_epoch,
                        callbacks=[checkpointer])

    # Save model
    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print("Saved model to disk")