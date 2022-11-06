from predict import *
from  annotateVideos import annotate
import os.path


def sortDictionary(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

def annotate_batch(batch, annotate_num):
    j=0
    annotations = []
    for key in batch:
        if j<annotate_num:
            annotations.append(annotate(key))
            j+=1
    return annotations


def uncertaintySampling(n, annotate_num):
    funny_batch = {}
    exciting_batch = {}
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
        if (0.5-n) < result[1] < (0.5+n):
            funny_batch.update({i:float(result[1])})


    funny_batch = sortDictionary(funny_batch)
    exciting_batch = sortDictionary(exciting_batch)
    print("Batch to annotate (funny)...", funny_batch)
    funny_labels = annotate_batch(funny_batch, annotate_num)




uncertaintySampling(0.1, 5)
