import cv2
import numpy as np
import sklearn
import tensorflow as tf
import os
import csv
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Convolution2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Model
import random

# Use generator to load data
samples = []
with open('../Drive_Manual/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

random.shuffle(samples) # shuffle the images after read them all, and before the test_train split, m

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../Drive_Manual/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) # convert the img to RGB to be matched with drive.py
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
#                 images.append(np.fliplr(center_image)) #flip the image, because it is not suit for the same croping methode, commented out, m
#                 angles.append(-center_angle) # the steering angle reverse, because it is not suit for the same croping methode, commented out, m
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Build up the layers (Cropping the images, )
ch, row, col = 3, 80, 320  # Trimmed image format
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # same meaning with x/127.5-1
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) #70,25
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,\
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=2, verbose=1)

model.save('model.h5')

# produce the visualization

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=2, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()