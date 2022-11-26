from quantiseVideo import *
import sys
sys.path.append('../')
from evaluateModel import *
from tensorflow import keras
import cv2
import glob


def maskVideos(video_path):
    print("Quantising video...")
    ranges, fps, frameSize = getFrames(video_path, 10)

    data = load_sample(video_path)
    result = predict(data, model)
    print("Original video result", result)

    frames = []

    for filename in glob.glob('../../data/frames/*.jpg'):
        filename = filename.split('/')[4].split('.')[0]
        frames.append(int(filename[5:]))
    frames = sorted(frames)

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


if __name__ == "__main__":
    print("Loading model...")
    model = keras.models.load_model('../../data/models/predict_model')
    video_path = "../../data/videos/test_videos/2496.mp4"
    print(video_path[3:])
    maskVideos(video_path)
