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
import copy
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import glob

# global model
# model = keras.models.load_model('../../data/models/predict_model')

#print(model.summary())

def remove():
    files = glob.glob('../../data/LIMEset/*')
    for f in files:
        if f.endswith('jpg'):
            os.remove(f)


def visualiseSuperPixels(segments, image):
    for (i, segVal) in enumerate(np.unique(segments)):
        print("Inspecting segment %d" % (i))
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        cv2.imshow("Mask", mask)
        cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
        key = cv2.waitKey(0)


def createNeighbourhoodSet(image_path, blocks, perturbed_num, pixel_segments=500, visualise=False):
    image = img_as_float(io.imread(image_path))
    segments = slic(image, n_segments=pixel_segments, sigma=5, start_label=1)

    if visualise:
        visualiseSuperPixels(segments, image)
        # visualise super pixel regions
        fig = plt.figure("Superpixels -- %d segments" % (pixel_segments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        plt.show()
    else:
        print("\nTo see super pixel segmentation set visualise=True\n")

    perturbed_pixels = []
    for i in range(perturbed_num):
        frame = cv2.imread(image_path)
        indexes = []
        nums = random.sample(range(1,np.max(segments)), blocks)
        for n in nums:
            pos = np.transpose(np.where(segments==n))
            for p in pos:
                indexes.append(p)
        perturbed_pixels.append(np.asarray(indexes))

    return perturbed_pixels


def maskPixels(pixels, i, j):
    frame = cv2.imread('../../data/frames/frame'+str(i)+'.jpg')
    for p in pixels:
        frame[p[0], p[1]] = (0,0,0)
    cv2.imwrite("../../data/LIMEset/"+ str(i) +".jpg", frame)


def createMaskedVideos(prime_frame, lower_frame, upper_frame):
    j=1
    path = '../../data/frames/frame' + str(prime_frame) + ".jpg"
    perturbed_pixels = createNeighbourhoodSet(path, 10, 10)

    for pixels in perturbed_pixels:
        remove()
        for i in range(lower_frame, upper_frame):
            maskPixels(pixels, i, j)

        video = '../../data/LIMEset/' + str(j) + '.mp4'
        out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('m','p','4','v'), 59.94, (600,480))

        files = glob.glob('../../data/LIMEset/*')
        for f in files:
            img = cv2.imread(f)
            out.write(img)
        out.release()
        j+=1
    remove()
    print("Finished")


if __name__ == '__main__':
    createMaskedVideos(20, 1, 28)



# explainer = lime_image.LimeImageExplainer()
#
# # create auxilary local model
# print('Creating explaination')
# explanation  = explainer.explain_instance(y.astype('double'), model.predict, top_labels=2, hide_color=None, num_samples=1000)
#
# print(explanation)
