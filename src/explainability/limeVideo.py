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
from sklearn.linear_model import Ridge, lars_path

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
        def k(d):
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        kf = partial(k)

        data, labels, order = self.data_labels(classifier_fn)

        print("\nData and labels created")
        distances = []
        vidcap = cv2.VideoCapture(video)
        success, original_image = vidcap.read()
        for f in order:
            distances.append(self.distance(f, original_image[:,:,0]))
        distances = np.asarray(distances).ravel()
        print(distances.shape)
        ret_exp = VideoExplanation(video)
        print("\nVideo explaination created")
        top = np.argsort(labels[0])
        print("Top:", top)
        ret_exp.top_labels = list(top)
        ret_exp.top_labels.reverse()
        labels = labels.reshape(20, 2, 1)
        for label in top:
            weights = kf(distances)
            labels_column = labels[:, label]
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            print("Data shape:", data.shape)
            print("Labels shape:", labels_column.shape)

            clf.fit(data, labels_column, sample_weight=weights)
            sys.exit()
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
        data, labels, order = [], [], []
        # might have to sort this
        files = glob.glob('../../data/LIMEset/*')
        for f in files:
            order.append(f)
            print("\nFile:", f)
            d = load_sample(f)
            data.append(d)
            label = classifier_fn(d)
            print(label)
            labels.append(label)
        return np.squeeze(np.array(data)), np.squeeze(np.array(labels)), order

    def distance(self, video, original_image):
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        image = image[:,:,0]

        if success:
            d = sklearn.metrics.pairwise_distances(
                image,
                original_image,
                metric = 'cosine'
            )
            return d
        else:
            print("Distance calcultion failed")

if __name__ == '__main__':
    global model
    model = keras.models.load_model('../../data/models/predict_model')
    originl_video = '../../data/LIMEset/0.mp4'
    explainer = LimeVideoExplainer()
    print("\nExplainer created")
    explanation = explainer.explain_instances(originl_video, model.predict)
