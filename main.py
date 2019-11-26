from readFiles import get_list_of_raw_videos, trim_videos_in_list
from vectorizeFiles import extract_frames_from_trimmed_videos 
from extractFeaturesVectorizedVideos import create_training_data

def main():
    print("starting program")
    directoryPath = "C:\\Users\\Nico Rode\\Desktop\\NFLAnalysis"
    firstTimeBool = False

    prep_videos(directoryPath, firstTimeBool)




def prep_videos(directoryPath, firstTimeBool):
    if firstTimeBool:
        listOfRawVideos = get_list_of_raw_videos("{}\\{}".format(directoryPath, "NFLRawVideos"))
        trim_videos_in_list(listOfRawVideos, directoryPath)
        extract_frames_from_trimmed_videos(directoryPath)
    create_training_data(directoryPath)





if __name__ =="__main__":
    main()









