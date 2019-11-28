from readFiles import get_list_of_raw_videos, trim_videos_in_list
from vectorizeFiles import extract_frames_from_trimmed_videos 
from extractFeaturesVectorizedVideos import create_training_data, test_testing_data
from CNNDefenseDetection import build_naive_model, train_model, predict_model
import os
import numpy as np
import pickle
from keras.models import load_model
from cv2 import cv2

def main():
    print("starting program")    
    PCDirectoryPath = "C:\\Users\\Nico Rode\\Desktop\\NFLAnalysis"
    macDirectoryPath = os.path.join("/","Users", "nicholasrode", "Desktop", "ProjectCanary")
    firstTimeBool = False

    prep_videos(macDirectoryPath, firstTimeBool)

    



def prep_videos(directoryPath, firstTimeBool):



    if firstTimeBool:
        listOfRawVideos = get_list_of_raw_videos("{}\\{}".format(directoryPath, "NFLRawVideos"))
        trim_videos_in_list(listOfRawVideos, directoryPath)
        extract_frames_from_trimmed_videos(directoryPath)
        create_training_data(directoryPath)

        pickle_in = open("X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb")
        y = pickle.load(pickle_in)

        X = X/255.0

        model = build_naive_model(X)

        model = train_model(model, X, y)



    test_testing_data(directoryPath)


if __name__ =="__main__":
    main()









