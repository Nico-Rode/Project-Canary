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

# function to grab the list of downloaded coach-view mp4 files. 
# will then be passed to the trim videos function to get relevant parts of video for training
def get_list_of_raw_videos(directoryPath):

    print("grabbing files")
    listOfRawVideos = []
    for file in os.listdir(directoryPath):
        if file.endswith(".mp4"):
            listOfRawVideos.append(file)
        else:
            continue

    return listOfRawVideos


# currently just grabbing the fisrt few seconds of the videos for presnap look of offense and defense
def trim_videos_in_list(listOfRawVideos, folderPath):
    for fileName in listOfRawVideos:
        ffmpeg_extract_subclip("{}\\NFLRawVideos\\{}".format(folderPath, fileName), 3.5, 9, targetname="{}\\NFLPreSnapVideos\\Motion\\snippedVideos\\{}".format(folderPath, fileName))
       








