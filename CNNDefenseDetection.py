from numpy import array
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, TimeDistributed, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Activation
import pickle
import os
from datetime import datetime


logdir = "logs/scalars/" + "nr112619"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)





print("Keras version = {}".format(keras.__version__))



def build_time_distributed_model():
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


    return model


def build_naive_model(X):

    model = Sequential()

    model.add(Convolution2D(224, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(224, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.save("nflDefenseDetectionModel.h5")
    return model



def train_model(model, X, y):
    model.fit(X, y, batch_size=64, epochs=2, validation_split=0.3, callbacks=[tensorboard_callback])
    return model


def predict_model(model, X):
   return model.predict_classes(X) 



