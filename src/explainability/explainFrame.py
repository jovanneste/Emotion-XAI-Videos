import lime
from lime import lime_image
import os
import sys
sys.path.append('../')
from evaluateModel import *
import cv2
from tensorflow import keras
import keras.utils as image
import numpy as np

global model
model = keras.models.load_model('../../data/models/predict_model')

print(model.summary())

def transform_img_fn(img_path):
    print('transform_img_fn')
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return np.squeeze(x)

explainer = lime_image.LimeImageExplainer()
# print("Frame data")
# frame = transform_img_fn('../../data/frames/frame300.jpg')
# print(frame.shape)
#

data = load_sample('../../data/videos/train_videos/43.mp4')
print("data=", data.shape)
y = np.squeeze(data)
print("y=", y.shape)
x = np.expand_dims(data, axis=0)
print("x=", x.shape)
# create auxilary local model
print('Creating explaination')
explanation  = explainer.explain_instance(y.astype('double'), model.predict, top_labels=2, hide_color=None, num_samples=1000)

print(explanation)
