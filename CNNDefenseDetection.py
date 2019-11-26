from numpy import array
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, TimeDistributed, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
import pickle

from imageGenerator import ImageDataGenerator




print("Keras version = {}".format(keras.__version__))


layer = Convolution2D(64, (3, 3), activation='relu')

model = Sequential()

input = TimeDistributed(layer, input_shape = (150, 224, 224, 1))
# input, with 64 convolutions for 5 images
# that have (224, 224, 3) shape
model.add(input)


model.add(
    TimeDistributed( 
        Convolution2D(64, (3,3), activation='relu')
    )
)
model.add(
    TimeDistributed(
        GlobalAveragePooling2D()
    )
)

model.add(
    LSTM(1024, activation='relu', return_sequences=False)
)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(.5))
# For example, for 3 outputs classes 
model.add(Dense(3, activation='sigmoid'))
model.compile('adam', loss='categorical_crossentropy')


import os
directoryPath = "C:\\Users\\Nico Rode\\Desktop\\NFLAnalysis"


datagen = ImageDataGenerator()

train_generator = datagen.flow_from_directory(
        os.path.join(directoryPath, "NFLPreSnapVideos", "Extractions", "Train"),
        target_size=(224, 224),
        batch_size=32,
        frames_per_step=150,
        shuffle=False)

validation_generator = datagen.flow_from_directory(
        os.path.join(directoryPath, "NFLPreSnapVideos", "Extractions", "Test"),
        target_size=(224, 224),
        batch_size=32,
        frames_per_step=150,
        shuffle=False)


model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=250)

