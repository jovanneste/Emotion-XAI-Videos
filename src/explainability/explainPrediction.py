from quantiseVideo import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
sys.path.append('../')
from evaluateModel import *
from tensorflow import keras
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

# does not work - we will use LIME instead
def maskPixels(key_frame, lower_frame, upper_frame, frameSize, fps, box_size=100):
    print("Original video prediction:")
    exciting_label = predict(load_sample("../../data/videos/test_videos/2496.mp4"), model)[0]

    frames = getSortedFrames()
    differences = {}

    for vertical_counter in range(frameSize[1]//box_size):
        for horizontal_counter in range(frameSize[0]//box_size):
            frame_index = (horizontal_counter, vertical_counter)
            print('frame index')
            print(frame_index)
            print("Removing masked frames from last video...")
            files = glob.glob('../../data/maskedFrames/*')
            for f in files:
                os.remove(f)

            print("Masking " + str(upper_frame-lower_frame) + " frames...")

            for i in range(lower_frame, upper_frame):
                frame = cv2.imread('../../data/frames/frame' + str(i) + '.jpg')

                for k in range(vertical_counter*box_size,(vertical_counter+1)*box_size):
                    for j in range(horizontal_counter*box_size,(horizontal_counter+1)*box_size):
                        frame[k][j] = (0,0,0)

                cv2.imwrite('../../data/maskedFrames/frame'+str(i)+'.jpg', frame)

            print("Building masked video...")
            video = str(frame_index) + '.mp4'
            out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frameSize)

            for f in frames:
                if lower_frame <= f < upper_frame:
                    img = cv2.imread('../../data/maskedFrames/frame' +str(f)+ '.jpg')
                    out.write(img)
                else:
                    img = cv2.imread('../../data/frames/frame' +str(f)+ '.jpg')
                    out.write(img)

            out.release()

            data = load_sample(video)
            result = predict(data, model)
            print("Masked video prediction")
            print(video)
            print(result)
            differences.update({frame_index:result[0]-exciting_label})

            #os.remove(video)
    print("\nFinished\n")
    print(sortDict(differences, False))
