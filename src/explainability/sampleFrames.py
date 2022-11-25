from quantiseVideo import *
import sys
sys.path.append('../')
from evaluateModel import *
from tensorflow import keras

model = keras.models.load_model('../../data/models/predict_model')

video_path = "../../data/videos/test_videos/2496.mp4"
print(video_path[3:])
# ranges = getFrames(video_path, 10)
# for 2496
ranges = [[40, 1, 49], [320, 272, 323], [350, 337, 353], [810, 809, 812],
[260, 231, 268], [850, 822, 854], [860, 853, 866], [430, 382, 431], [380, 371, 382],
[750, 749, 752]]

data = load_sample(video_path)
result = predict(data, model)

print(result)
