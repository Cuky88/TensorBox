# First Arg: path to directory with video files
# Second Arg: output path
# python imgToVideo.py ~/Devel/data_link/FZI_crossing/beginning/ ~/Devel/data_link/FZI_crossing/FZI_Video.avi
from os import listdir
from os.path import isfile, join
import cv2
import sys
import os

def imgToVideo(inputPath, outputPath):
    print("Start Conversion")
    # Read all images
    imgfiles = [f for f in listdir(inputPath) if isfile(join(inputPath, f)) and not f.endswith('txt')]
    
    images = sorted(imgfiles, key=lambda x: float(x.split(".png")[0].split("T")[1]))
    #print(images)
    #return 

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputPath, fourcc, 20, (960, 540))
    print("##### Total images: %s\n"%len(imgfiles))
    print("##### Start video creation!")

    j = 1
    for i in images:
        #print(i)
        img = cv2.imread(inputPath+i, -1)
        #print(img.shape)
        frame = cv2.resize(img, (960, 540))
	print("%d: %s"%(j, i))
        # write the frame
        out.write(frame)
	j += 1

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    print("##### Finished!")

# MAIN
inputPath = sys.argv[1]
outputPath = sys.argv[2]

imgToVideo(inputPath, outputPath)
