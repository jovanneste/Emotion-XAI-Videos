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
        print("Explain instance\n")

        print("Data shape:", data.shape)
        print("Labels shape:", labels.shape)

        print()
        distances = []
        vidcap = cv2.VideoCapture(video)
        success, original_image = vidcap.read()
        for f in order:
            distances.append(self.distance(f, original_image[:,:,0]))
        distances = np.asarray(distances).ravel()

        ret_exp = VideoExplanation(video)

        print("Distances shape:", distances.shape)
        top = np.argsort(labels[0])[-5:]
        ret_exp.top_labels = list(top)
        ret_exp.top_labels.reverse()
        labels = labels.reshape(20, 2, 1)
        print("Top", top)
        for label in top:
            print("Calling explain_instance_with_data with:")
            print("data", data.shape)
            print("labels", labels.shape)
            print("distances", distances.shape)
            print("label", label.shape, label)
            print("num_features", num_features)
            sys.exit()
            # imitating what happens in lime base when we call explain with instance
            weights = kf(distances)
            labels_column = labels[:, label]
            used_features = self.base.feature_selection(data,
                                               labels_column,
                                               weights,
                                               100000,
                                               method='auto')
            print()
            print(used_features)
            print()
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            print("Data shape:", data.shape)
            print("Labels shape:", labels_column.shape)

            clf.fit(data[:,used_features], labels_column, sample_weight=weights)
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
