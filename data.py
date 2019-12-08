import os
import glob
import keras
from keras_video.generator import VideoFrameGenerator
from keras.preprocessing.image import ImageDataGenerator
#from keras_video.sliding import SlidingFrameGenerator


def video_gen(directoryPath, SIZE, CHANNELS, NBFRAME):
    classPath = os.path.join(directoryPath, "NFLPreSnapVideos", "Train")
    print(classPath)

    print("Listing Dirs")
    print(os.listdir(classPath))


    classes = [i for i in os.listdir(classPath)]
    print(classes)



    classes.sort()
    # some global params
    BS = 8
    # pattern to get videos and classes
    glob_pattern = os.path.join(classPath, '{classname}', "*.mp4")
    #glob_pattern=classPath+'/{classname}/*.mp4'
    print(glob_pattern)
    # for data augmentation
    data_aug = ImageDataGenerator(
        zoom_range=.1,
        horizontal_flip=True,
        rotation_range=8,
        width_shift_range=.2,
        height_shift_range=.2)
    # Create video frame generator
    train = VideoFrameGenerator(
        classes=classes, 
        glob_pattern=glob_pattern,
        nb_frames=NBFRAME,
        split=.3, 
        shuffle=True,
        batch_size=BS,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)
    
    valid = train.get_validation_generator()
    import keras_video.utils

    return train, valid, classes
    

