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
import random

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt


# global model
# model = keras.models.load_model('../../data/models/predict_model')

#print(model.summary())

def getSuperPixels(image_path, n=300):
    image = img_as_float(io.imread(image_path))
    segments = slic(image, n_segments=n, sigma=5, start_label=1)
    fig = plt.figure("Superpixels -- %d segments" % (n))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
    plt.show()
    return segments


def createNeighbourhoodSet(image_path, blocks, perturbed_num):
    segments = getSuperPixels(image_path)
    for i in range(perturbed_num):
        nums = random.sample(range(1,np.max(segments)), blocks)
        print(nums)


path = '../../data/frames/frame4.jpg'
createNeighbourhoodSet(path, 20, 1)

# explainer = lime_image.LimeImageExplainer()
#
# # create auxilary local model
# print('Creating explaination')
# explanation  = explainer.explain_instance(y.astype('double'), model.predict, top_labels=2, hide_color=None, num_samples=1000)
#
# print(explanation)
