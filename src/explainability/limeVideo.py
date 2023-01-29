import numpy as np
import pickle
import lime
from lime import lime_base
from PIL import Image
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
from img2vec_pytorch import Img2Vec


class VideoExplanation(object):
    def __init__(self, video):
        self.video = video
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}


class LimeVideoExplainer(object):
    def __init__(self):
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        kernel_fn = partial(kernel)
        self.random_state = check_random_state(None)
        self.base = lime_base.LimeBase(kernel_fn, True, random_state=self.random_state)

    def explain_instances(self, video, classifier_fn, segments):
        print(segments)
        print(segments.shape)
        data, labels, order = self.data_labels(classifier_fn)

        distances = []
        vidcap = cv2.VideoCapture(video)
        success, original_image = vidcap.read()
        for f in order:
            distances.append(self.distance(f, original_image[:,:,0]))
        distances = np.asarray(distances)

        ret_exp = VideoExplanation(video)

        top = np.argsort(labels)[-1:]
        ret_exp.top_labels = list(top)
        ret_exp.top_labels.reverse()

        for label in top:
            (ret_exp.intercept[label[0]],
             ret_exp.local_exp[label[0]],
             ret_exp.score[label[0]],
             ret_exp.local_pred[label[0]]) = self.explain_instance_with_data(data, labels, distances, label, segments)
        print("Done")
        return ret_exp

    def data_labels(self, classifier_fn, scale=0.4):
        data, labels, order = [], [], []
        # might have to sort this
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
            return sklearn.metrics.pairwise_distances(image, original_image, metric = 'cosine')
        else:
            print("Distance calculation failed")


    def explain_instance_with_data(self, neighbourhood_data, neighbourhood_labels, distances, label, segments, num_features=1000):
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        kernel_fn = partial(kernel)

        weights = kernel_fn(distances).ravel()
        labels_column = np.squeeze(neighbourhood_labels[:, label])
        used_features = self.base.feature_selection(neighbourhood_data,
                                                    labels_column,
                                                    weights,
                                                    num_features,
                                                    method='none')

        used_features = [i for i in range(np.max(segments))]

        features = self.embedding(neighbourhood_data)

        model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        easy_model = model_regressor

        easy_model.fit(features[:, used_features], neighbourhood_labels)
        prediction_score = easy_model.score(features[:, used_features], neighbourhood_labels)

        local_pred = easy_model.predict(features[0,used_features].reshape(1, -1))
        print(easy_model.intercept_)
        print(easy_model.coef_[0])
        print(sorted(zip(used_features, easy_model.coef_[0]),key=lambda x: np.abs(x[1]), reverse=True))
        print(prediction_score)
        print(local_pred)

        return (easy_model.intercept_,
                easy_model.coef_,
                prediction_score, local_pred)


    def embedding(self, neighbourhood_data):
        #weights=ResNet18_Weights.DEFAULT
        img2vec = Img2Vec(cuda=False, model='resnet18')
        video_features = []
        for video in neighbourhood_data:
            frame_features = []
            for frame in video:
                img = Image.fromarray(frame)
                vec = img2vec.get_vec(img, tensor=True)
                frame_features.append(np.squeeze(np.asarray(vec).ravel()))
            video_features.append(sum(frame_features)/len(frame_features))
        return np.asarray(video_features)




if __name__ == '__main__':
    file = open('segments', 'rb')
    segments = pickle.load(file)
    file.close()
    global model
    model = keras.models.load_model('../../data/models/predict_model')
    originl_video = '../../data/LIMEset/0.mp4'
    explainer = LimeVideoExplainer()
    print("\nExplainer created")
    explanation = explainer.explain_instances(originl_video, model.predict, segments)
