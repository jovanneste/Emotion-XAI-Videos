from tensorflow import keras
import tensorflow as tf
from PIL import Image
import math
import torch
import clip
import cv2
import numpy as np
import pandas as pd
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

global pretrained_model
pretrained_model = keras.models.load_model('../data/models/predict_model')

SAMPLE_FRAMES = 10
FrameSize = 216

# autopep8 -i to fix indentation errors

def load_sample(video_path):
    # read video
    video = cv2.VideoCapture(video_path)
    # if video loading fails, exit the program
    if not video.isOpened():
        print('Video open failed, please check the video file.', video_path)
        sys.exit()
    # CLIP setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # read video informations and calculate sample rate
    FrameNumber = video.get(7)
    FrameNumber = int(FrameNumber)
    rate = math.floor((FrameNumber-1)/SAMPLE_FRAMES)
    if rate <= 0:
        rate = 1

    # initailize
    video_samples = []

    # Sampling
    for i in range(SAMPLE_FRAMES):
        samp_loc = i*rate
        if samp_loc >= FrameNumber:
            samp_loc = FrameNumber-1

        # set video reading location to sample point
        video.set(cv2.CAP_PROP_POS_FRAMES, samp_loc)
        rval, frame = video.read()
        if not rval:
            print('Video read Failed, please check the video file.')

        # preprocessing frame for clip
        frame = cv2.resize(frame, (FrameSize, FrameSize),
                           fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        img_pred = preprocess(frame_pil).unsqueeze(0).to(device)

        # add sample to sample list
        encoded_image = model.encode_image(img_pred)
        encoded_image_array = encoded_image.detach().cpu().numpy()
        video_samples.append(encoded_image_array)

    # resize the samples to fit model
    video_samples = np.array(video_samples, dtype='float')
    video_samples = np.squeeze(video_samples, axis=-2)
    video_samples = np.expand_dims(video_samples, 0)
    return video_samples


def round(n):
    if n < 0.5:
        return 0
    else:
        return 1


def predict(data,model):
    y_score = pretrained_model.predict(data)
    return [y_score[0][0], y_score[0][1]]


def evaluateModel(model):
    print('Evaluating Model', str(model))
    df = pd.read_csv('../data/annotatedVideos.csv', delim_whitespace=True)
    nums = df.shape[0]
    funny_accuracy, exciting_accuracy = 0, 0
    for index, row in df.iterrows():
        print(index)
        idn = row['id']
        funny_label = row['Funny']
        exciting_label = row['Exciting']

        video_path = "../data/videos/test_videos/"+str(idn)+".mp4"

        try:
            data = load_sample(video_path)
            result = predict(data,model)
            print(result)
        except:
            print("Video Failed")
            break

        # might want F1 or precision too
        if round(result[0]) == exciting_label:
            exciting_accuracy += 1
        if round(result[1]) == funny_label:
            funny_accuracy += 1

    return [exciting_accuracy/nums, funny_accuracy/nums]



def crossValidation(k=5):
    df = pd.read_csv('../data/annotatedVideos.csv', delim_whitespace=True)
    videos = []

    for index, row in df.iterrows():
        videos.append("../data/videos/test_videos/"+str(row['id'])+".mp4")

    videos = np.split(np.array(videos), k)

    test_fold = videos[0]
    train_fold = np.concatenate(videos[1:])

    train_values = []
    train_labels = []

    test_values = []
    test_labels = []

    for index, row in df.iterrows():
        video_path = "../data/videos/test_videos/"+str(row['id'])+".mp4"
        if video_path in train_fold:
            print('Loading for training...', video_path)
            train_values.append(load_sample(video_path))
            train_labels.append([row['Exciting'], row['Funny']])
        elif video_path in test_fold:
            print('Loading for testing...', video_path)
            test_values.append(load_sample(video_path))
            test_labels.append([row['Exciting'], row['Funny']])


    train_values = np.asarray(train_values).astype('float32').reshape(len(train_values), 10, 512)
    train_labels = np.asarray(train_labels)
    print("Updating model on fold 1...")
    pretrained_model.fit(
        train_values,
        train_labels,
        batch_size=4,
        epochs=10,
    )
    print("Saving model")
    pretrained_model.save('../../data/predict_model')

    exciting_accuracy, funny_accuracy, nums = 0, 0, 0

    for data in test_values:
        print("Testing", nums)
        result = predict(data)
        if round(result[0]) == test_labels[nums][0]:
            exciting_accuracy += 1
        if round(result[1]) == test_labels[nums][1]:
            funny_accuracy += 1
        nums += 1
    exciting_accuracy = exciting_accuracy/nums
    funny_accuracy = funny_accuracy/nums
    average = (exciting_accurac+funny_accuracy)/2

    return [average, exciting_accuracy, funny_accuracy]



if __name__ == '__main__':
    print(evaluateModel(pretrained_model))
