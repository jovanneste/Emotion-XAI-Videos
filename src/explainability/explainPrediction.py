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


def plot(x, y, ranges):
    lowers, uppers = [], []
    for r in ranges:
        for point in x:
            if r[0]==point:
                lowers.append(np.abs(r[0] - r[1]))
                uppers.append(np.abs(r[0] - r[2]))
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=[lowers, uppers], fmt='-o')
    ax.set_xlabel("Frames")
    ax.set_ylabel("Frame importance")
    plt.show()

def getSortedFrames():
    frames = []
    for filename in glob.glob('../../data/frames/*.jpg'):
        filename = filename.split('/')[4].split('.')[0]
        frames.append(int(filename[5:]))
    return sorted(frames)

def sortDict(d, key):
    if key:
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}
    else:
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

def maskFrames(video_path, n):
    print("Quantising video...")
    ranges, fps, frameSize = getFrames(video_path, n)
    data = load_sample(video_path)
    result = predict(data, model)
    exciting_label = result[0]
    print("Original video result", result)
    print(ranges)

    differences = {}
    frames = getSortedFrames()

    for r in ranges:
        keyFrame = r[0]
        print("Building new video without", keyFrame)
        lowerBound = r[1]
        upperBound = r[2]
        out = cv2.VideoWriter(str(keyFrame) + '.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), fps, frameSize)
        for frame in frames:
            if lowerBound <= frame <= upperBound:
                continue
            else:
                img = cv2.imread('../../data/frames/frame' +str(frame)+ '.jpg')
                out.write(img)

        out.release()

        data = load_sample(str(keyFrame) + '.mp4')
        result = predict(data, model)
        print(keyFrame)
        print(result)
        differences.update({keyFrame:result[0]-exciting_label})


    differences = sortDict(differences, True)
    x = np.asarray(list(differences.keys()))
    y = np.asarray(list(differences.values()))*-1

    plot(x, y, ranges)

    prime_frame = list(differences.keys())[0]
    print("prime frame", prime_frame)

    for r in ranges:
        if r[0]==prime_frame:
            lower_frame, upper_frame = r[1], r[2]
            break

    return prime_frame, lower_frame, upper_frame, frameSize, fps


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


if __name__ == "__main__":
    print("Loading model...")
    global model
    model = keras.models.load_model('../../data/models/predict_model')
    video_path = "../../data/videos/train_videos/31.mp4"
    prime_frame, lower_frame, upper_frame, frameSize, fps = maskFrames(video_path, 15)
    print('\n\n\n')
    print(prime_frame, lower_frame, upper_frame, frameSize, fps)
    print('\n\n\n')
    maskPixels(prime_frame, lower_frame, upper_frame, frameSize, fps)
