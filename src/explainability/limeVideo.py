import numpy as np
import lime
from lime import lime_base
from functools import partial
import sklearn
# from sklearn import metrics
# from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
import glob
import sys
sys.path.append('../')
from evaluateModel import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

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
        self.base = lime_base.LimeBase(
            kernel_fn, True, random_state=self.random_state)

    def explain_instances(self, video, classifier_fn):
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
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.explain_instance_with_data(data, labels, distances, label)
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


    def explain_instance_with_data(self, neighbourhood_data, neighbourhood_labels, distances, label, num_features=1000):
        # delete this
        def k(d):
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        kf = partial(k)

        print("neighbourhood_data", neighbourhood_data.shape)
        print("neighbourhood_labels", neighbourhood_labels.shape)
        print("distances", distances.shape)
        print("label", label.shape, label)

        weights = kf(distances).ravel()
        labels_column = np.squeeze(neighbourhood_labels[:, label])
        used_features = self.base.feature_selection(neighbourhood_data,
                                                    labels_column,
                                                    weights,
                                                    num_features,
                                                    method='none')

        print("Creating auxilary model...")
        batch_size, frame_num, width, height, depth = neighbourhood_data.shape

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(neighbourhood_data))
        print(normalizer.mean.numpy())

        model = self.build_and_compile_model(normalizer)
        print(model.summary())

    def build_and_compile_model(self, norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1e-3))
        return model

if __name__ == '__main__':
    global model
    model = keras.models.load_model('../../data/models/predict_model')
    originl_video = '../../data/LIMEset/0.mp4'
    explainer = LimeVideoExplainer()
    print("\nExplainer created")
    explanation = explainer.explain_instances(originl_video, model.predict)
