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


        test_testing_data(directoryPath)

    # define_data(directoryPath, 150)
    # X = np.load(os.path.join(directoryPath,"savedTrainingData.npy"))
    # print(X.shape)
    # y = np.load(os.path.join(directoryPath,"savedTrainingActual.npy"))

    # model = build_time_distributed_model()
    # model = train_model(model, X, y)

    SIZE = (224, 224)
    CHANNELS = 3
    NBFRAME = 5

    train, valid, classes = video_gen(directoryPath, SIZE, CHANNELS, NBFRAME)

    print(len(classes))

    EPOCHS=50
    # create a "chkp" directory before to run that
    # because ModelCheckpoint will write models inside
    # callbacks = [
    #     keras.callbacks.ReduceLROnPlateau(verbose=1),
    #     keras.callbacks.ModelCheckpoint(
    #         'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    #         verbose=1),
    # ]

    model = build_time_distributed_model(NBFRAME, SIZE, CHANNELS, classes, train, valid, EPOCHS)

    #model = build_medium_model()
    # print("got compiled model")
    # model.fit_generator(
    #     train,
    #     validation_data=valid,
    #     verbose=1,
    #     epochs=EPOCHS
    # )
    # print("trained model")
    model.save("mediumModel.h5")
    

    #Figure out what the issue is with global pooling vs mac pooling






if __name__ =="__main__":
    main()









