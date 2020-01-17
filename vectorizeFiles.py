import os
from moviepy.editor import VideoFileClip
from moviepy.video.tools import drawing
from PIL import Image
import csv
import os.path
from subprocess import call

import numpy as np
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def extract_frames_from_trimmed_videos(directoryPath, classPath, model):
    dataFile = []
    folders = ['train']

    for folder in folders:
        classFolders = glob.glob(os.path.join("{}\\NFLPreSnapVideos\\{}\\{}".format(directoryPath,classPath,folder), '*'))
        print(classFolders)
        for outcome in classFolders:
            print(outcome)
            defensiveFormationVideos = glob.glob(os.path.join(outcome, '*.mp4'))
            print(defensiveFormationVideos)
            for video in defensiveFormationVideos:
                trainOrTest = "train"
                className = outcome.split("\\")[-1]  
                fileName = video.split("\\")[-1]
                fileNameNoExtension = fileName.split(".mp4")[0]

                #print("train or test: {} ---- className: {} ------ filename: {} ----- filenameNoExtension: {}".format(trainOrTest, className, fileName, fileNameNoExtension))

                if not check_already_extracted(directoryPath, trainOrTest, className, model, fileName, fileNameNoExtension):
                    print("got here")
                    src = os.path.join(directoryPath, "NFLPreSnapVideos", classPath, trainOrTest, className, fileName)
                    print("source: {}".format(src))
                    dest = os.path.join(directoryPath, "NFLPreSnapVideos", model, "Extractions", trainOrTest, className, fileNameNoExtension + '-%04d.jpg')
                    print("Command: ffmpeg -i  \"{}\" -vf scale=224:224  \"{}\"".format(src, dest))
                    #call(["ffmpeg", "-i", "\"{}\"".format(src), "\"{}\"".format(dest)])
                    call("ffmpeg -i  \"{}\" -vf scale=224:224 \"{}\"".format(src, dest), shell=True)

                    nbFrames = get_nb_frames_for_video(directoryPath, trainOrTest, className, fileNameNoExtension)

                    dataFile.append([trainOrTest, className, fileNameNoExtension, nbFrames])

                    print("Generated %d frames for %s" % (nbFrames, fileNameNoExtension))
                else:
                    print("Already extracted!")

                
        if dataFile:
            with open('{}//logs//data_file.csv'.format(directoryPath), 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerows(dataFile)

        print("Extracted and wrote %d video files." % (len(dataFile)))






def get_nb_frames_for_video(directoryPath, trainOrTest, className, fileNameNoExtension):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    generated_files = glob.glob(os.path.join(directoryPath, "NFLPreSnapVideos\\Extractions", trainOrTest, className, fileNameNoExtension + '*.jpg'))
    return len(generated_files)




def check_already_extracted(directoryPath, trainOrTest, className, model, fileName, fileNameNoExtension):
    """Check to see if we created the -0001 frame of this file."""
    return bool(os.path.exists(os.path.join(directoryPath, "NFLPreSnapVideos", model, "Extractions", trainOrTest, className, fileNameNoExtension + '-0001.jpg')))
