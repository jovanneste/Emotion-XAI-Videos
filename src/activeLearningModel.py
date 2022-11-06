from predict import *
from  annotateVideos import annotate
import os.path
import subprocess, sys


def sortDictionary(d):
    return sorted(d.items(), key=lambda x: abs(x[1] - 0.5))

def annotate_batch(batch, annotate_num, videos):
    j=0
    annotations = {}
    for key in batch:
        if j<annotate_num:
            annotations.update({key:annotate(key, videos)})
            j+=1
    return annotations


def uncertaintySampling(n, annotate_num):
    funny_batch = {}
    exciting_batch = {}
    videos = []
    for i in range(1, 10):
        print(i)
        video_path = "../data/videos/"+str(i)+".mp4"
        if os.path.exists(video_path):
            data = load_sample(video_path)
            result = predict(data)
        else:
            continue
        if (0.5-n) < result[0] < (0.5+n):
            exciting_batch.update({i:float(result[0])})
        elif (0.5-n) < result[1] < (0.5+n):
            # if video already in list it will probably be annotated anyway
            funny_batch.update({i:float(result[1])})

    funny_batch = sortDictionary(funny_batch)
    exciting_batch = sortDictionary(exciting_batch)

    for filename in os.listdir("../data/videos"):
       with open(os.path.join("../data/videos", filename), 'r') as f:
           videos.append(filename)

    print("Batch to annotate (funny)...", funny_batch)
    #funny_labels = annotate_batch(funny_batch, annotate_num, videos)
    print("Batch to annotate (exciting)...", exciting_batch)
    #funny_labels = annotate_batch(exciting_batch, annotate_num, videos)

    for k,v in funny_batch:
        print(k)
        video_path = "../data/videos/"+str(k)+".mp4"
        data = load_sample(video_path)
        print(data)



uncertaintySampling(0.3, 5)
