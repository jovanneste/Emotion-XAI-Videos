from quantiseVideo import *
import sys
sys.path.append('../')
from evaluateModel import *
#from tensorflow import keras
import cv2
import glob

#model = keras.models.load_model('../../data/models/predict_model')

video_path = "../../data/videos/test_videos/2496.mp4"
print(video_path[3:])
print("Quantising video...")
#ranges, fps, frameSize = getFrames(video_path, 10)
# for 2496
ranges = [[40, 0, 49], [320, 272, 323], [350, 337, 353], [810, 809, 812],
[260, 231, 268], [850, 822, 854], [860, 853, 866], [430, 382, 431], [380, 371, 382],
[750, 749, 752]]

# data = load_sample(video_path)
# result = predict(data, model)

result = [0.95431006, 0.00025563448]
frames = []


fps = 30.0
frameSize = (600, 480)

for filename in glob.glob('../../data/frames/*.jpg'):
    filename = filename.split('/')[4].split('.')[0]
    frames.append(int(filename[5:]))
frames = sorted(frames)

print('Frame size', frameSize)
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
    print("Video finished... EXITING")
