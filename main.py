from readFiles import get_list_of_raw_videos, trim_videos_in_list
from vectorizeFiles import extract_frames_from_trimmed_videos 
from extractFeaturesVectorizedVideos import create_training_data, test_testing_data

from CNNDefenseDetection import build_time_distributed_model, build_medium_model
from DefensiveRNN import get_data, define_data
from data import video_gen
import os
import numpy as np
import pickle

from keras.models import load_model
from cv2 import cv2
import keras

def main():
    print("starting program")    
    #PCDirectoryPath = "C:\\Users\\Nico Rode\\Desktop\\NFLAnalysis"
    PCDirectoryPath = os.path.join("c:/", "Users", "Nico Rode", "Desktop", "NFLAnalysis")
    
    macDirectoryPath = os.path.join("/","Users", "nicholasrode", "Desktop", "ProjectCanary")
    firstTimeBool = False

    prep_videos(PCDirectoryPath, firstTimeBool)



def prep_videos(directoryPath, firstTimeBool):
    if firstTimeBool:
        listOfVideso = get_list_of_raw_videos(os.path.join(directoryPath, "NFLRawVideos"))
        print(listOfVideso)
        trim_videos_in_list(listOfVideso, directoryPath)
        # extract_frames_from_trimmed_videos(directoryPath)
        # create_training_data(directoryPath)

    SIZE = (224, 224)
    CHANNELS = 1
    NBFRAME = 5
    train, valid, classes = video_gen(directoryPath, SIZE, CHANNELS, NBFRAME)
    EPOCHS=50
    # create a "chkp" directory before to run that
    # because ModelCheckpoint will write models inside

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(verbose=1),
        keras.callbacks.ModelCheckpoint(
            'logs/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1),
    ]

    model = build_time_distributed_model(NBFRAME, SIZE, CHANNELS, classes, train, valid, EPOCHS, callbacks)

    model.save("mediumModel.h5")
    




if __name__ =="__main__":
    main()









