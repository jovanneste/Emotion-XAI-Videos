import os
import sys
sys.path.append('../')
from evaluateModel import *
from quantise_video import *
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
import pickle

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


def plot(x, y, ranges):
    lowers, uppers = [], []
    for r in ranges:
        for point in x:
            if r[0]==point:
                lowers.append(np.abs(r[0] - r[1]))
                uppers.append(np.abs(r[0] - r[2]))
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=[lowers, uppers], fmt='-o')
    ax.set_xlabel("Frames")
    ax.set_ylabel("Frame importance")
    plt.show()


def getSortedFrames():
    frames = []
    for filename in glob.glob('../../data/frames/*.jpg'):
        filename = filename.split('/')[4].split('.')[0]
        frames.append(int(filename[5:]))
    return sorted(frames)


def sortDict(d, key):
    if key:
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}
    else:
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def maskFrames(video_path, n, model, verbose, label):
    ranges, fps, frameSize = getFrames(video_path, n, verbose)
    data = load_sample(video_path)
    result = predict(data, model)
    label_result = result[label]
    differences = {}
    frames = getSortedFrames()

    for r in ranges:
        keyFrame = r[0]
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

        data = load_sample(str(keyFrame) + '.mp4')
        result = predict(data, model)
        differences.update({keyFrame:result[label]-label_result})


    differences = sortDict(differences, False)
    x = np.asarray(list(differences.keys()))
    y = np.asarray(list(differences.values()))*-1

    prime_frame = list(differences.keys())[0]

    for r in ranges:
        if r[0]==prime_frame:
            lower_frame, upper_frame = r[1], r[2]
            break

    if verbose:
        print("Ranges used: ", ranges)
        print("Order: ", differences)
        print("Prime frame", prime_frame)
        plot(x, y, ranges)

    return prime_frame, lower_frame, upper_frame, frameSize, fps


def createNeighbourhoodSet(image_path, blocks, perturbed_num, prime_frame, pixel_segments, visualise):
    image = img_as_float(io.imread(image_path))
    segments = slic(image, n_segments=pixel_segments, sigma=5, start_label=1)
    plt.imshow(mark_boundaries(image, segments))
    segments_and_prime_frame = [segments, prime_frame]
    file = open('segments_and_prime_frame', 'wb')
    pickle.dump(segments_and_prime_frame, file)
    file.close()

    print(perturbed_num)

    # if visualise:
    #     visualiseSuperPixels(segments, image)
    #     # visualise super pixel regions
    #     fig = plt.figure("Superpixels -- %d segments" % (pixel_segments))
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.imshow(mark_boundaries(image, segments))
    #     plt.axis("off")
    #     plt.show()


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
    if j!=0:
        for p in pixels:
            frame[p[0], p[1]] = (0,0,0)
    cv2.imwrite("../../data/LIMEset/"+ str(i) +".jpg", frame)


def createMaskedVideos(prime_frame, lower_frame, upper_frame, fps, frameSize, n, num_segments, verbose):
    j=0
    path = '../../data/frames/frame' + str(prime_frame) + ".jpg"
    perturbed_pixels = createNeighbourhoodSet(path, 10, n, prime_frame, num_segments, verbose)

    for pixels in perturbed_pixels:
        remove()
        for i in range(int(lower_frame), int(upper_frame)):
            maskPixels(pixels, i, j)

        video = '../../data/LIMEset/' + str(j) + '.mp4'
        out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('m','p','4','v'), fps, frameSize)

        files = glob.glob('../../data/LIMEset/*')
        for f in files:
            img = cv2.imread(f)
            out.write(img)
        out.release()
        j+=1
    remove()
    if verbose:
        print("Masked videos created")
