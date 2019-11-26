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



def create_training_data(directoryPath):
    CATEGORIES = ["43Formation", "46Formation", "52Formation"]
    trainingPath = os.path.join(directoryPath, "NFLPreSnapVideos", "Extractions", "Train")
    video_data = []
    training_data =[]

    for category in CATEGORIES:  
        path = os.path.join(trainingPath,category)  # create path to defensive formations
        class_num = CATEGORIES.index(category)  # get the classification  (0, 1, 2) -- 0 = 43Formation, 1 = 46Formation, 2 = 52Formation
        for index, img in enumerate(tqdm(os.listdir(path))):  # iterate over each image per dogs and cats
            try:
                if (index%150 == 0 and index != 0) or index+1 == len(tqdm(os.listdir(path))):
                    training_data.append([video_data, class_num])  # add this to our training_data
                    video_data = []

                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                video_data.append(img_array)    
               
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
    training_data = np.asarray(training_data, dtype=np.float32)


    print("PRINTING LENGTHS: {}".format(len(training_data)))

    
    X = []
    y = []
    for features,label in training_data:
        X.append(features)
        y.append(label)

    print(X[0].reshape(5, 224, 224, 1))

    X = np.array(X).reshape(5, 224, 224, 1)

    import pickle

    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()













