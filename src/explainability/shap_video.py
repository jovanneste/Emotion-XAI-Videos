import shap
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import sys
sys.path.append('../')
from evaluateModel import *
# import cv2
#
model = tf.keras.models.load_model('predict_model')
# cvframe = cv2.imread('../../data/frames/frame16.jpg')[:,:,::-1]
#
data = load_sample('../../data/videos/train_videos/1940.mp4')
#
#
# # explainer = shap.Explainer(model,max_evals=3841)
# # explainer = shap.explainers.Permutation(model, max_evals = 3841)
explainer = shap.DeepExplainer(model, data)
print("Explainer:")
print(explainer)
# #
#
#
#
shap_values = explainer.shap_values(data)
print(shap_values)
print(len(shap_values))
print(shap_values[0].shape)
