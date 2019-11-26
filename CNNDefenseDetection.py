from numpy import array
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, TimeDistributed, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
import pickle




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



pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

model.fit(X,y, batch_size=32, epochs=3, validation_split=.3)



