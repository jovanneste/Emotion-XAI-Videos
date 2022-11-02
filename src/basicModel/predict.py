import sys
from tkinter import filedialog
import torch
from PIL import Image
import clip
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
import math



### This file is the source code of Emotion Prediction Program ###

SAMPLE_FRAMES = 50
FrameSize = 216
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():
    print('Welcome using Emotion Prediction. Please select the file you want to predict. The file should be .mp4.')
    video_path = select_video()
    print('Video sampling, please wait:')
    data = load_sample(video_path)
    print('Sample successed. Predicting Video labels.')
    result = predict(data)

    # preparing output lines
    if result[0]:
        line1 = 'This video is exciting. '
    else:
        line1 = 'This video is not exciting. '

    if result[1]:
        line2 = 'This video is funny.'
    else:
        line2 = 'This video is not funny.'
    # CLIP
    print(line1+line2)
    print('Thanks for using emotion prediction.')
    sys.exit(0)

# use filefialog to get file path
def select_video():
    my_filetypes = [('video files', '.mp4')]
    answer = filedialog.askopenfilename(initialdir=os.getcwd(),
                                        title="Please select a .mp4 file:",
                                        filetypes=my_filetypes)
    return answer

# load sample from the video
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

# predict emotion result using pre-trainde model
def predict(data):
    # read pre-trained model from local dict
    model = keras.models.load_model('../../data/predict_model')
    print('loading model')
    y_score = model.predict(data)
    result = [True,True]

    # set result. y_score[0] is the prediction value for exciting and y_score[1] is for funny.
    if y_score[0][0] > 0.5:
        result[0] = True
    else:
        result[0] = False
    if y_score[0][1] > 0.5:
        result[1] = True
    else:
        result[1] = False

    return result

if __name__ == "__main__":
    main()