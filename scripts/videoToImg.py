# First Arg: path to directory with video files
# Second Arg: output path
from os import listdir
from os.path import isfile, join
import cv2
import sys
import os

def videoToImg(fileName, inputPath, outputPath):
    filePath = join(inputPath, fileName)
    print(filePath)
    cap = cv2.VideoCapture(filePath)
    if not cap.isOpened():
        cap.open(filePath)
    retries = 0
    frameNumber = 0
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not os.path.isfile(join(outputPath, fileName + "_frame_" + str(int(pos_frame)) + '.png')):
                resized = cv2.resize(frame, (864,480))
                cv2.imwrite(join(outputPath, fileName + "_frame_" + str(int(pos_frame)) + '.png'), resized)
            print("[Info] File: " + str(fileName) + ", Frame: "+ str(pos_frame) +  "/" + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            # The next frame is not ready, so we try to read it again
            print("Frame is " + str(frameNumber) + " pos_frame " + str(pos_frame - 1))
            if frameNumber != pos_frame-1:
                retries = 1
                frameNumber = pos_frame - 1
            else:
                retries += 1

            if retries < 5:
                print("[Info] Going 1 frame back. " + str(retries) + " Retries")
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                cv2.waitKey(1000)
            else:
                # skip frame
                cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame+1)
                print("[Error] File: " + fileName + ", Frame " + str(pos_frame) + " not saved. (total frames: " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + ")")
                retries = 0
                if cap.get(cv2.CAP_PROP_FRAME_COUNT) - pos_frame < 5:
                    break;
            # It is better to wait for a while for the next frame to be ready

        # if cv2.waitKey(10) == 27:
        #     break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break


# MAIN
inputPath = sys.argv[1]
outputPath = sys.argv[2]

onlyfiles = [f for f in listdir(inputPath) if isfile(join(inputPath, f)) and not f.endswith('txt')]

for f in onlyfiles:
    videoToImg(f, inputPath, outputPath)
