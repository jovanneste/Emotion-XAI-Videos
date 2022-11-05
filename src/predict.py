import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import cv2
import clip
import torch
import math
from PIL import Image
import tensorflow as tf
from tensorflow import keras

global pretrained_model
pretrained_model = keras.models.load_model('../data/predict_model')

SAMPLE_FRAMES = 10
FrameSize = 216

df = pd.read_csv('../data/annotatedVideos.csv', delim_whitespace=True)

def load_sample(video_path):
    # read video
    video = cv2.VideoCapture(video_path)

    # if video loading fails, exit the program
    if not video.isOpened():
        print('Video open failed, please check the video file.')
        sys.exit(0)

    # CLIP setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # read video informations and calculate sample rate
    FrameNumber = video.get(7)
    FrameNumber = int(FrameNumber)
    rate = math.floor((FrameNumber-1)/SAMPLE_FRAMES)
    if rate<=0:
        rate = 1

    # initailize
    video_samples = []

    # Sampling
    for i in range(SAMPLE_FRAMES):
        samp_loc = i*rate
        if samp_loc >= FrameNumber:
            samp_loc = FrameNumber-1

        # set video reading location to sample point
        video.set(cv2.CAP_PROP_POS_FRAMES,samp_loc)
        rval, frame = video.read()
        if not rval:
            print('Video read Failed, please check the video file.')

        # preprocessing frame for clip
        frame = cv2.resize(frame,(FrameSize,FrameSize),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        img_pred = preprocess(frame_pil).unsqueeze(0).to(device)

        # add sample to sample list
        encoded_image = model.encode_image(img_pred)
        encoded_image_array = encoded_image.detach().cpu().numpy()
        video_samples.append(encoded_image_array)

    # resize the samples to fit model
    video_samples = np.array(video_samples,dtype='float')
    video_samples = np.squeeze(video_samples,axis=-2)
    video_samples = np.expand_dims(video_samples,0)

    return video_samples


def round(n):
    if n < 0.5:
        return 0
    else:
        return 1

def predict(data):
    y_score = pretrained_model.predict(data)
    return y_score[0][0], y_score[0][1]


i=1
funny_accuracy = 0
exciting_accuracy = 0
for index, row in df.iterrows():
	print(index)
	idn = row['id']
	funny_label  = row['Funny']
	exciting_label = row['Exciting']

	video_path = "../data/videos/"+str(idn)+".mp4"
	data = load_sample(video_path)
	exciting_pred, funny_pred = predict(data)


	if round(exciting_pred) == exciting_label:
		exciting_accuracy += 1
	if round(funny_pred)  == funny_label:
		funny_accuracy += 1

	if i == 60:
		break
	i+=1


print(i)
print(funny_accuracy/i)
print(exciting_accuracy/i)