from numpy import array
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, TimeDistributed, Dropout, GRU
from keras.layers import GlobalMaxPooling2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Activation, BatchNormalization, MaxPooling1D
from keras.layers import Convolution2D as Conv2D

import pickle
import os
from datetime import datetime



print("Keras version = {}".format(keras.__version__))



def build_convnet(shape=(224, 224, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(64, (3,3), input_shape=shape, padding='same', activation='relu'), )
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPooling2D())
    
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPooling2D())
    
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalAveragePooling2D())
    return model


def action_model(shape=(5, 112, 112, 3), nbout=5):
    # Create our convnet with (112, 112, 3) input shape
    print(shape[1:])
    convnet = build_convnet(shape[1:])

    print("layer shape")
    print(convnet.output_shape)
    
    # then create our final model
    model = keras.Sequential()
    print("Adding timedistributed")
    model.add(TimeDistributed(convnet, input_shape=shape))

    print("adding LSTM")
    model.add(GRU(64))
# add the convnet with (5, 112, 112, 3) shape

    print("after LSTM")
# and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    print("returning model")
    return model


def build_time_distributed_model(NBFRAME, SIZE, CHANNELS, classes, train, valid, EPOCHS, callbacks):
    INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
    print(INSHAPE)
    model = action_model(INSHAPE, len(classes))
    print("got model")
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['acc'])
    print("compiled model")

    print("got compiled model")
    model.fit_generator(
        train,
        validation_data=valid,
        verbose=1,
        callbacks = callbacks,
        epochs=EPOCHS
    )
    print("trying to train")


    
    return model



def build_medium_model(shape=(5,224,224,3)):
    momentum = .9
    model = Sequential()
    # after having Conv2D...
    model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu'), input_shape=(5, 224, 224, 3)))

    model.add(TimeDistributed(Conv2D(64, (3,3), input_shape=shape, padding='same', activation='relu'), ))
    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(BatchNormalization(momentum=momentum)))
    
    model.add(TimeDistributed(MaxPooling2D()))
    
    model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(BatchNormalization(momentum=momentum)))
    
    model.add(TimeDistributed(MaxPooling2D()))
    
    model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(BatchNormalization(momentum=momentum)))
    
    model.add(TimeDistributed(MaxPooling2D()))
    
    model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(BatchNormalization(momentum=momentum)))
    
    # flatten...
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    
    
    # previous layer gives 5 outputs, Keras will make the job
    # to configure LSTM inputs shape (5, ...)
    model.add(
        LSTM(1024, activation='relu', return_sequences=False)
    )
    # and then, common Dense layers... Dropout...
    # up to you
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.3))
    # For example, for 3 outputs classes 
    model.add(Dense(3, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(optimizer, 'categorical_crossentropy', metrics=['acc'])

    return model



