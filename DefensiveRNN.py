import os
import os.path
from tqdm import tqdm
import numpy as np
from collections import deque
import pickle
from cv2 import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def get_data(directoryPath, num_frames, num_classes, input_length):
    """Get the data from our saved predictions or pooled features."""


    CATEGORIES = ["43Formation"]
    trainingPath = os.path.join(directoryPath, "NFLPreSnapVideos", "Extractions", "Train")

        # Local vars.
    X = []
    y = []
    temp_list = deque()

    for category in CATEGORIES:
        categoricalPath = os.path.join(trainingPath, category)
        for img in tqdm(os.listdir(categoricalPath)):  # iterate over each image per dogs and cats
            try:
                features = cv2.imread(os.path.join(trainingPath, category, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                actual = CATEGORIES.index(category)
               
            except Exception as e:  # in the interest in keeping the output clean...
                pass

            if len(temp_list) == len(tqdm(os.listdir(categoricalPath))) - 1:
                temp_list.append(features)
                flat = list(temp_list)
                X.append(np.array(flat))
                y.append(actual)
                temp_list = temp_list.popleft()
            else:
                temp_list.append(features)
                continue

    print("Total dataset size: %d" % len(X))

    # Numpy.
    X = np.array(X)
    y = np.array(y)

    # Reshape.
    X = X.reshape(-1, num_frames, input_length)

    # One-hot encoded categoricals.
    y = to_categorical(y, num_classes)

    # Split into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test


def define_data(directoryPath, SEQ_LEN):

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = []  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    CATEGORIES = ["43Formation", "46Formation", "52Formation"]
    trainingPath = os.path.join(directoryPath, "NFLPreSnapVideos", "Extractions", "Train")

    for category in CATEGORIES:
        categoricalPath = os.path.join(trainingPath, category)
        for img in tqdm(os.listdir(categoricalPath)):  # iterate over each image per dogs and cats
            try:
                features = cv2.imread(os.path.join(trainingPath, category, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                actual = CATEGORIES.index(category)
               
            except Exception as e:  # in the interest in keeping the output clean...
                pass

            # prev_days.append(features)  # store all but the target
            # if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            #     sequential_data.append([np.array(prev_days), actual])  # append those bad boys!
            #     prev_days = []
            sequential_data.append([features, category])
    

    print(len(sequential_data))

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    print(len(X[0][0]))
    X = np.array(X).reshape(150,224,224)
    y = np.array(y)
    np.save(os.path.join(directoryPath,"savedTrainingData"), X)
    np.save(os.path.join(directoryPath,"savedTrainingActual"), y)



def save_data(X,y):
    pickle_out = open("X2.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y2.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

