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
import numpy as np

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
        kernel_fn = partial(kernel, kernel_width=0.25)
        self.random_state = check_random_state(None)
        self.base = lime_base.LimeBase(kernel_fn, True, random_state=self.random_state)

    def explain_instances(self, video, classifier_fn):
        data, labels = self.data_labels(classifier_fn)
        print("\nData and labels created")


        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1,-1),
            metric = 'cosine'
        ).ravel()

        ret_exp = VideoExplanation(video)
        print("/nVideo explaination created")
        top = np.argsort(labels[0])[-5:]
        print("Top:", top)
        ret_exp.top_labels = list(top)
        ret_exp.top_labels.reverse()

        for label in top:
            print("\nLabel in top:", label)
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(data,
                                                                               labels,
                                                                               distances,
                                                                               label,
                                                                               100000)
        print("Done")
        return ret_exp

    def data_labels(self, classifier_fn):
        data, labels = [], []
        # might have to sort this
        files = glob.glob('../../data/LIMEset/*')
        for f in files:
            print("\nFile:", f)
            d = load_sample(f)
            data.append(d)
            label = classifier_fn(d)
            print(label)
            labels.append(label)
        return np.array(data), np.array(labels)

    def distanceVideos(v1, v2):



if __name__ == '__main__':
    global model
    model = keras.models.load_model('../../data/models/predict_model')
    originl_video = '../../data/LIMEset/0.mp4'
    explainer = LimeVideoExplainer()
    print("\nExplainer created")
    explanation = explainer.explain_instances(originl_video, model.predict)
