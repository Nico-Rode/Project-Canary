from readFiles import get_list_of_raw_videos, trim_videos_in_list
from vectorizeFiles import extract_frames_from_trimmed_videos 
from extractFeaturesVectorizedVideos import create_training_data, test_testing_data

from CNNDefenseDetection import build_time_distributed_model
from DefensiveRNN import get_data, define_data
from data import video_gen
import os
import numpy as np
import pickle

from keras.models import load_model
from cv2 import cv2
import keras

import datetime



def main():
    print("starting program")    
    #PCDirectoryPath = "C:\\Users\\Nico Rode\\Desktop\\NFLAnalysis"
    PCDirectoryPath = os.path.join("c:/", "Users", "Nico Rode", "Desktop", "NFLAnalysis")
    
    macDirectoryPath = os.path.join("/","Users", "nicholasrode", "Desktop", "ProjectCanary")

    motionClasspath = os.path.join("Motion", "snippedVideos")
    motionClass = "Motion"
    firstTimeBool = True

    prep_videos(PCDirectoryPath, motionClasspath, motionClass, firstTimeBool)
    generate_NN(PCDirectoryPath, motionClasspath)



def prep_videos(directoryPath, classPath, model, firstTimeBool):
    if firstTimeBool:
        listOfVideso = get_list_of_raw_videos(os.path.join(directoryPath, "NFLRawVideos"))
        print(listOfVideso)
        #trim_videos_in_list(listOfVideso, directoryPath)
        #extract_frames_from_trimmed_videos(directoryPath, classPath, model)
        #create_training_data(directoryPath)

def generate_NN(directoryPath, classPath):
    SIZE = (112, 112)
    CHANNELS = 1
    NBFRAME = 4
    train, valid, classes = video_gen(directoryPath, classPath, SIZE, CHANNELS, NBFRAME)
    EPOCHS=100
    # create a "chkp" directory before to run that
    # because ModelCheckpoint will write models inside

    print("DATE")
    print(datetime.date.today())

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(verbose=1),
        keras.callbacks.ModelCheckpoint(
            'logs/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1),
        keras.callbacks.TensorBoard(log_dir='logs/runs/size-({}, {})--NBFrame-{}--channels{}--date-{}'.format(str(SIZE[0]), str(SIZE[1]), NBFRAME, CHANNELS, str(datetime.date.today())))
    ]

    model = build_time_distributed_model(NBFRAME, SIZE, CHANNELS, classes, train, valid, EPOCHS, callbacks)

    model.save("mediumModel.h5")
    




if __name__ =="__main__":
    main()









