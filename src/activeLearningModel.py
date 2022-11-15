from evaluateModel import *
from  annotateVideos import annotate
import os.path
import subprocess, sys
from tensorflow import keras
import random

global pretrained_model
pretrained_model = keras.models.load_model('../data/models/predict_model')

def sortDictionary(d):
    return sorted(d.items(), key=lambda x: abs(x[1][0] - 0.5))

def annotate_batch(batch, annotate_num, videos):
    j=0
    annotations = {}
    for key, v in batch.items():
        if j<annotate_num:
            annotations.update({key:annotate(key, videos)})
            j+=1
    return annotations

def normaliseValues(train_values, train_labels):
    train_values = np.asarray(train_values).astype('float32').reshape(len(train_values), 10, 512)
    train_labels = np.asarray(train_labels)
    return train_values, train_labels


def randomSampling(annotate_num, bootstrap_steps):
    videos = []
    batch = {}

    for filename in os.listdir("../data/videos/train_videos"):
       with open(os.path.join("../data/videos/train_videos", filename), 'r') as f:
           videos.append(filename)

    random.shuffle(videos)

    for i in range(bootstrap_steps):
        print("Iteration", i)
        j=0
        for video in videos:
            print(video)
            print(j)
            if j<annotate_num:
                video_path = "../data/videos/train_videos/"+str(video)
                print(video_path)
                if os.path.exists(video_path):
                    data = load_sample(video_path)
                    batch.update({j:data})
                j+=1
            else:
                break

        videos = videos[annotate_num:]
        annotations = annotate_batch(batch, annotate_num, videos)

        train_values = []
        train_labels = []

        for key, v in annotations.items():
            train_values.append(batch[key][0])
            train_labels.append([int(v[1]), int(v[2])])

        train_values, train_labels = normaliseValues(train_values, train_labels)

        print("Training model on new labelled instances")

        try:
            model = keras.models.load_model('../data/models/random_sampling_model')
            print("Random sampling model found")
        except:
            print("Random sampling model not found")
            model = keras.models.load_model('../data/models/predict_model')

        model.fit(
            train_values,
            train_labels,
            batch_size=4,
            epochs=10,
        )
        print("Saving model")
        model.save('../data/models/random_sampling_model')

        print("Evaluating model after " + str(i) + " iterations...")

        print(evaluateModel(model))


def uncertaintySampling(n, annotate_num):
    funny_batch = {}
    exciting_batch = {}
    videos = []

    for filename in os.listdir("../data/videos/train_videos"):
       with open(os.path.join("../data/videos/train_videos", filename), 'r') as f:
           videos.append(filename)

    for i in range(1, 500):
        print(i)
        video_path = "../data/videos/train_videos/"+str(i)+".mp4"
        if os.path.exists(video_path):
            data = load_sample(video_path)
            result = predict(data)
        else:
            continue
        if (0.5-n) < result[0] < (0.5+n):
            exciting_batch.update({i:[float(result[0]), data]})
        elif (0.5-n) < result[1] < (0.5+n):
            # if video already in list it will probably be annotated anyway
            funny_batch.update({i:[float(result[1]), data]})


    funny_batch = sortDictionary(funny_batch)
    exciting_batch = sortDictionary(exciting_batch)

    datas = {}
    datas.update(funny_batch)
    datas.update(exciting_batch)

    funny_videos_info = annotate_batch(funny_batch, annotate_num, videos)
    exciting_videos_info = annotate_batch(exciting_batch, annotate_num, videos)

    annotated_insts_info = {}
    annotated_insts_info.update(funny_videos_info)
    annotated_insts_info.update(exciting_videos_info)

    train_values = []
    train_labels = []

    for key, v in annotated_insts_info.items():
        train_values.append(datas[key][1])
        train_labels.append([int(v[1]), int(v[2])])



    print('Training model on new labelled instances...')

    train_values, train_labels = normaliseValues(train_values, train_labels)

    pretrained_model.fit(
        train_values,
        train_labels,
        batch_size=4,
        epochs=10,
    )
    print("Saving model")
    model.save('../data/models/uncertainty_sampling_model')


randomSampling(5, 5)
# print("\nUncertainty Sampling\n")
# uncertaintySampling(0.4, 16)
