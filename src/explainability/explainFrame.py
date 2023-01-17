import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import numpy as np
import keras.utils as image
from tensorflow import keras
import cv2
from evaluateModel import *
import lime
from lime import lime_image
import os
import sys
sys.path.append('../')


# global model
# model = keras.models.load_model('../../data/models/predict_model')

# print(model.summary())

image = img_as_float(io.imread('../../data/frames/frame4.jpg'))

for numSegments in (200):
	segments = slic(image, n_segments=numSegments, sigma=5, start_label=1)
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	print("Number:", numSegments)
	print("Segments:", segments)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")

plt.show()



# explainer = lime_image.LimeImageExplainer()
#
# # create auxilary local model
# print('Creating explaination')
# explanation  = explainer.explain_instance(y.astype('double'), model.predict, top_labels=2, hide_color=None, num_samples=1000)
#
# print(explanation)
