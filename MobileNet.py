import keras
#from keras import models.mobilenet.MobileNet
from keras.layers import Dense, LSTM, Input, TimeDistributed, Dropout, GRU
from extractFeaturesVectorizedVideos import create_training_data, test_testing_data
import os
from data import video_gen
directoryPath = os.path.join("/","Users", "nicholasrode", "Desktop", "ProjectCanary")
PCDirectoryPath = os.path.join("c:/", "Users", "Nico Rode", "Desktop", "NFLAnalysis")


def build_mobilenet(shape=(224, 224, 3), nbout=3):
    model = keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=shape,
        weights='imagenet')
    trainable = 9
    for layer in model.layers[:-trainable]:
        layer.trainable = False
    for layer in model.layers[-trainable:]:
        layer.trainable = True
    output = keras.layers.GlobalMaxPooling2D()
    return keras.Sequential([model, output])




def action_model(shape=(5, 112, 112, 3), nbout=3):


    # Create our convnet with (112, 112, 3) input shape
    convnet = build_mobilenet(shape[1:])

    model = keras.Sequential()

    model.add(TimeDistributed(convnet, input_shape=shape))

    print("adding LSTM")
    model.add(GRU(64))
# add the convnet with (5, 112, 112, 3) shape

    
    # then create our final model
    
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


# Set size to 224, 224
SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 5
BS = 8


train, valid, classes = video_gen(PCDirectoryPath, SIZE, CHANNELS, NBFRAME)

print(len(classes))

EPOCHS=15
# create a "chkp" directory before to run that
# because ModelCheckpoint will write models inside
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'logs/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]


INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 224, 224, 3)
model = action_model(INSHAPE, 3)
optimizer = keras.optimizers.SGD()
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)


print("got compiled model")
model.fit_generator(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS
)


