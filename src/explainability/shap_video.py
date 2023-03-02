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
import scipy

BACKGROUND_SET_NUM = 1
FEATURES = 5

model = tf.keras.models.load_model('predict_model')
to_explain = load_sample('vivo_minions_ad.mp4')

print("Creating background set for expectations...")
background = []
files = random.sample(glob.glob('../../data/videos/train_videos/*'),BACKGROUND_SET_NUM)
for f in files:
    background.append(load_sample(f))

video_result = predict(to_explain, model)

# creating SHAP explainer
explainer = shap.DeepExplainer(model, background)
shap_values = np.squeeze(np.asarray(explainer.shap_values(to_explain)))

shap_result = predict(shap_values, model)
assert video_result!=shap_result, "SHAP values not correctly calculated."
# fidelity
# print(scipy.stats.pearsonr(video_result, shap_result))
print("Relevance:")
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(video_result, shap_result)
print(r_value**2)
#
# relevance
print("Excitement fidelity: " + str(video_result[0]-shap_result[0]))
print("Funny fidelity: " + str(video_result[1]-shap_result[1]))
