import shap
import tensorflow as tf
# tf 2.0 is not compatible with SHAP deepExplain
tf.compat.v1.disable_v2_behavior()
import sys
sys.path.append('../')
from evaluateModel import *
import numpy as np
import glob
import random

BACKGROUND_SET_NUM = 1
FEATURES = 1

model = tf.keras.models.load_model('predict_model')

print("Creating background set for expectations...")
background = []
files = random.sample(glob.glob('../../data/videos/train_videos/*'),BACKGROUND_SET_NUM)
for f in files:
    background.append(load_sample(f))


to_explain = load_sample('../../data/videos/train_videos/1955.mp4')
video_result = predict(to_explain, model)
explainer = shap.DeepExplainer(model, background)
shap_values = np.squeeze(np.asarray(explainer.shap_values(to_explain)))

shap_result = predict(shap_values, model)
