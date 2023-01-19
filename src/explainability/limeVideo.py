import lime
from lime import lime_base
from functools import partial
import sklearn
from sklearn.utils import check_random_state
import glob
import sys
sys.path.append('../')
from evaluateModel import *
import numpy as np

class VideoExplanation(object):
    def __init__(self, video):
        self.video = video


class LimeVideoExplainer(object):
    def __init__(self):
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        kernel_fn = partial(kernel, kernel_width=0.25)
        self.random_state = check_random_state(None)
        self.base = lime_base.LimeBase(kernel_fn, True, random_state=self.random_state)

    def explain_instances(self, video, classifier_fn):
        data, labels = self.data_labels(classifier_fn)
        distances = sklearn.metrics.pairwise_distance(
            data,
            data[0].reshape(1,-1),
            metric = 'cosine'
        ).ravel()


    def data_labels(self, classifier_fn):
        data, labels = [], []
        files = glob.glob('../../data/LIMEset/*')
        for f in files:
            d = load_sample('../../data/LIMEset/'+str(f))
            data.append(d)
            label = predict(data)
            labels.append(label)
        return np.array(data), np.array(labels)
