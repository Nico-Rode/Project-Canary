training_data = []
import os
import os.path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
from keras_preprocessing.image import load_img, img_to_array
from CNNDefenseDetection import build_naive_model, train_model, predict_model

import pickle



def create_training_data(directoryPath):
    CATEGORIES = ["43Formation", "46Formation", "52Formation"]
    trainingPath = os.path.join(directoryPath, "NFLPreSnapVideos", "Extractions", "Train")
    video_data = []

    for category in CATEGORIES:  
        path = os.path.join(trainingPath,category)  # create path to defensive formations
        class_num = CATEGORIES.index(category)  # get the classification  (0, 1, 2) -- 0 = 43Formation, 1 = 46Formation, 2 = 52Formation

        for index, img in enumerate(tqdm(os.listdir(path))):  # iterate over each image per dogs and cats
            try:
                # frames.append(os.path.join(path,img))
                # if index%150 == 0:
                #     training_data.append(build_image_sequence(frames))
                #     frames = []


                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                video_data.append([img_array, class_num])    
               
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


    X = []
    y = []
    for features,label in video_data:
        X.append(features)
        y.append(label)

    print(np.array(X).shape)
    #print(X[0].reshape(-1, 224, 224, 1))

    '''
    Reshaping data so that we have individual images stack on top on one another. 
    Current Naive approach, doesn't take temporal dimension into account
    '''
    X = np.array(X).reshape(-1, 224, 224, 1)

    save_data(X,y)

def test_testing_data(directoryPath):
    CATEGORIES = ["43Formation", "46Formation", "52Formation"]
    trainingPath = os.path.join(directoryPath, "NFLPreSnapVideos", "Extractions", "Test")
    video_data = []
    training_data =[]

    print(trainingPath)
    for category in CATEGORIES:  
        path = os.path.join(trainingPath,category)  # create path to defensive formations
        print(path)
        for index, img in enumerate(tqdm(os.listdir(path))):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                #img_array = img_array.reshape(-1, 224, 224, 1)
                video_data.append(img_array)    
               
            except Exception as e:  # in the interest in keeping the output clean...
                print(e)
                pass


    '''
    Reshaping data so that we have individual images stack on top on one another. 
    Current Naive approach, doesn't take temporal dimension into account
    '''
    X = np.array(video_data).reshape(-1,224,224,1)
    model = load_model("nflDefenseDetectionModel.h5")
    print(X.shape)
    y_prob = model.predict(X) 
    y_classes = y_prob.argmax(axis=-1)



"""
Saves the data to a pickle file -- will only run if first-time bool is set
"""

def save_data(X,y):
    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()



def build_image_sequence(frames):
    """Given a set of frames (filenames), build our sequence."""
    return [process_image(x, (224, 224, 3)) for x in frames]


def process_image(image, targetShape):
    h, w, _ = targetShape
    image = load_img(image, target_size=(h, w))
    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 224.).astype(np.float32)

    return x











