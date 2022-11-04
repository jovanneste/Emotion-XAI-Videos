import math
import cv2
import pandas as pd
from pathlib import Path
import pickle
import io
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers,Model
from PIL import Image
import pytesseract
import clip
from keras_preprocessing import sequence

### This file is for get encoded clip data for all videos ###

# read video list

folder_path = "../../data/basicModelVideos/"
data_path = "../../data/basicDatas/"

pd_reader = pd.read_csv("../../data/basicModelVideos/video_list.csv",header=None)
pd_reader_exciting = pd.read_json("../../data/video_Exciting_clean.json",orient='index')
pd_reader_funny = pd.read_json("../../data/video_Funny_clean.json",orient='index')
SamplesForEachVideo = 100
FrameSize = 216
SampleType = 'CLIP'
SampleRate = 1
SampleLength = 0

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

Y_list = []
samples = []
sample_sum = 0

print(pd_reader.shape[0])

for x in range(680):
    print(x)
    data_file_path = data_path+str(x)+'.pickle'
    if os.path.exists(data_file_path):
        print("Exists")
        video_tail = pd_reader.loc[x][0]
        video_tail = str(video_tail).replace("'",'')
  
        exciting_value = pd_reader_exciting.loc[video_tail][0]
        funny_value = pd_reader_funny.loc[video_tail][0]
    
        if exciting_value > 0.5:
            exciting_value = 1
        else:
            exciting_value = 0
        if funny_value > 0.5:
            funny_value = 1
        else:
            funny_value = 0
        Y_list.append([exciting_value,funny_value])
        continue

    video_tail = pd_reader.loc[x][0]
    video_tail = str(video_tail).replace("'",'')
    video_path = folder_path+video_tail+".mp4"
    video = cv2.VideoCapture(video_path)

    exciting_value = pd_reader_exciting.loc[video_tail][0]
    funny_value = pd_reader_funny.loc[video_tail][0]
    
    if exciting_value > 0.5:
        exciting_value = 1
    else:
        exciting_value = 0
    if funny_value > 0.5:
        funny_value = 1
    else:
        funny_value = 0
    Y_list.append([exciting_value,funny_value])

    if video.isOpened():
        rval, frame = video.read()
    else:
        rval = False

    FrameNumber = video.get(7)
    count = 0
    Samplecount = 0

    video_frames = []
    IsSampling = False
    sampletime = []
    raw_frames = np.zeros((1,512))

    n = math.floor(FrameNumber/SampleRate)+1

    for k in range(n):
        sampletime.append(k*SampleRate)

    while rval:
        if (count in sampletime):
            IsSampling = True
        if IsSampling:
            frame = cv2.resize(frame,(FrameSize,FrameSize),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame)
            img_pred = preprocess(frame_pil).unsqueeze(0).to(device)
            encoded_image = model.encode_image(img_pred)
            encoded_image_array = encoded_image.detach().cpu().numpy()
            raw_frames = raw_frames+encoded_image_array
            Samplecount += 1
        if Samplecount>SampleLength*2:
            Samplecount = 0
            IsSampling = False
            averaged_image_array = raw_frames/(SampleLength*2+1)
            video_frames.append(averaged_image_array)
            raw_frames = np.zeros((1,512))

        count+=1
        rval, frame = video.read()

    with open(data_file_path,'wb') as handle:
        pickle.dump(video_frames,handle)
    print("video No:" + str(x) +" Completed.")
    video.release()

#samples = sequence.pad_sequences(samples,padding='post',dtype='float')
#samples = np.array(samples,dtype=object)
Y_list = np.array(Y_list)


#print(samples[0:1])
print(Y_list[0:10])

#print(samples.shape)
print(Y_list.shape)

#with open('../../data/train_data_encoded_CLIP_ALL.pickle','wb') as handle:
#    pickle.dump(samples,handle)

with open('../../data/train_value_CLIP_ALL.pickle','wb') as handle:
    pickle.dump(Y_list,handle)

print('Encoded_sample saved.')



