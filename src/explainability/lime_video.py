import numpy as np
import pickle
import lime
from lime import lime_base
from PIL import Image
import cv2
from matplotlib import cm
from functools import partial
import sklearn
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
import glob
import sys
sys.path.append('../')
from evaluateModel import *

from transformers import AutoImageProcessor, VideoMAEForPreTraining
import torch
import clip
import tqdm
from img2vec_pytorch import Img2Vec
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

class VideoExplanation(object):
    def __init__(self, video, segments):
        self.video = video
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}


    def get_image_and_mask(self, label, prime_frame, num_features, positive_only=True, negative_only=False, min_weight=-10):
        segments = self.segments
        video = self.video
        values = list(self.local_exp.keys())
        exp = self.local_exp[values[0]]
        mask = np.zeros(segments.shape, segments.dtype)
        image = cv2.imread('../../data/frames/frame'+str(prime_frame)+'.jpg')
        temp = image.copy()

        if positive_only:
            try:
                fs = [x[0] for x in exp if x[1] > -5 and x[1] > min_weight][:num_features]
            except:
                # no sampled instances made any impact
                print("Using 1 feature only")
                fs = [x[0] for x in exp if x[1] > -5 and x[1] > min_weight][:1]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask


class LimeVideoExplainer(object):
    def __init__(self, l):
        self.l = l
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        kernel_fn = partial(kernel)
        self.random_state = check_random_state(None)
        self.base = lime_base.LimeBase(kernel_fn, True, random_state=self.random_state)

    def explain_instances(self, video, classifier_fn, segments):
        data, labels, order = self.data_labels(classifier_fn)

        distances = []
        vidcap = cv2.VideoCapture(video)
        success, original_image = vidcap.read()
        for f in order:
            distances.append(self.distance(f, original_image[:,:,0]))
        distances = np.asarray(distances)

        ret_exp = VideoExplanation(video, segments)

        top = np.argsort(labels)[-1:]
        ret_exp.top_labels = list(top)
        ret_exp.top_labels.reverse()

        for label in top:
            (ret_exp.intercept[label[0]],
             ret_exp.local_exp[label[0]],
             ret_exp.score[label[0]],
             ret_exp.local_pred[label[0]]) = self.explain_instance_with_data(data, labels, distances, label, segments)
        return ret_exp

    def data_labels(self, classifier_fn, scale=0.4):
        data, labels, order = [], [], []
        files = glob.glob('../../data/LIMEset/*')
        for f in files:
            frames = []
            order.append(f)
            d = load_sample(f)
            label = classifier_fn(d)
            labels.append(label)
            capture = cv2.VideoCapture(f)
            while(True):
                success, frame = capture.read()
                if success:
                    frames.append(cv2.resize(frame, (int(scale*frame.shape[1]), int(scale*frame.shape[0]))))
                else:
                    break
            capture.release()
            data.append(frames)
        return np.array(data), np.squeeze(np.array(labels)), order


    def distance(self, video, original_image):
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        image = image[:,:,0]

        if success:
            return sklearn.metrics.pairwise_distances(image, original_image, metric='cosine')
        else:
            print("Distance calculation failed")


    def explain_instance_with_data(self, neighbourhood_data, neighbourhood_labels, distances, label, segments):
        used_features = [i for i in range(np.max(segments))]
        features = self.feature_extraction(neighbourhood_data, clip_model=True)

        model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)

        model_regressor.fit(features[:, used_features], neighbourhood_labels)
        prediction_score = model_regressor.score(features[:, used_features], neighbourhood_labels)
        local_pred = model_regressor.predict(features[0,used_features].reshape(1, -1))


        return (model_regressor.intercept_,
                sorted(zip(used_features, model_regressor.coef_[self.l]),
                key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)


    def feature_extraction(self, neighbourhood_data, clip_model):
        video_features = []
        if clip_model:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            m, preprocess = clip.load("ViT-B/32", device=device)
            for video in neighbourhood_data:
                frame_features = []
                for frame in video:
                    frame_pil = Image.fromarray(frame)
                    img_pred = preprocess(frame_pil).unsqueeze(0).to(device)
                    encoded_image = m.encode_image(img_pred)
                    encoded_image_array = encoded_image.detach().cpu().numpy()
                    frame_features.append(np.squeeze(encoded_image_array))
                video_features.append(sum(frame_features)/len(frame_features))
        else:
            #weights=ResNet18_Weights.DEFAULT
            img2vec = Img2Vec(cuda=False, model='resnet18')
            for video in neighbourhood_data:
                frame_features = []
                for frame in video:
                    img = Image.fromarray(frame)
                    vec = img2vec.get_vec(img, tensor=True)
                    frame_features.append(np.squeeze(np.asarray(vec).ravel()))
                video_features.append(sum(frame_features)/len(frame_features))

        return np.asarray(video_features)
