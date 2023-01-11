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
# from keras.applications import inception_v3 as inc_net
#
# global model
# model = keras.models.load_model('../../data/models/predict_model')

def transform_img_fn(img_path):
    print('transform_img_fn')
    out = []
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inc_net.preprocess_input(x)
    out.append(x)
    return np.vstack(out)

explainer = lime_image.LimeImageExplainer()
#frame = transform_img_fn('../../data/frames/frame300.jpg')
print("Video data")
print(load_sample('../../data/videos/train_videos/1.mp4').shape)

print("Frame data")
data = load_sample('../../data/frames/frame300.jpg')
print(data.shape)


sys.exit()

# create auxilary local model
print('creating explaination')
explanation  = explainer.explain_instance(frame.astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)

print(explanation)
