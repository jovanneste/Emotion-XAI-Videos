from evaluateModel import *
from  annotateVideos import annotate
import os.path
import subprocess, sys
from tensorflow import keras

def sortDictionary(d):
    return sorted(d.items(), key=lambda x: abs(x[1][0] - 0.5))

def annotate_batch(batch, annotate_num, videos):
    j=0
    annotations = {}
    for key, v in batch:
        if j<annotate_num:
            annotations.update({key:annotate(key, videos)})
            j+=1
    return annotations

def uncertaintySampling(n, annotate_num):
    funny_batch = {}
    exciting_batch = {}
    videos = []

    for filename in os.listdir("../data/videos"):
       with open(os.path.join("../data/videos", filename), 'r') as f:
           videos.append(filename)

    for i in range(1, 3):
        print(i)
        video_path = "../data/videos/"+str(i)+".mp4"
        if os.path.exists(video_path):
            data = load_sample(video_path)
            result = predict(data)
        else:
            continue
        if (0.5-n) < result[0] < (0.5+n):
            exciting_batch.update({i:[float(result[0]), data]})
            print(result[0])
            print(data)
        elif (0.5-n) < result[1] < (0.5+n):
            # if video already in list it will probably be annotated anyway
            funny_batch.update({i:[float(result[1]), data]})
            print(result[1])
            print(data)


    funny_batch = sortDictionary(funny_batch)
    exciting_batch = sortDictionary(exciting_batch)

    datas = {}
    datas.update(funny_batch)
    datas.update(exciting_batch)


#    print("Batch to annotate (funny)...", funny_batch)
    funny_videos_info = annotate_batch(funny_batch, annotate_num, videos)
#    print("Batch to annotate (exciting)...", exciting_batch)
    exciting_videos_info = annotate_batch(exciting_batch, annotate_num, videos)

    annotated_insts_info = {}
    annotated_insts_info.update(funny_videos_info)
    annotated_insts_info.update(exciting_videos_info)

    train_values = []
    train_labels = []


    for key, v in annotated_insts_info.items():
        print('--------------------')
        train_values.append(datas[key][1])
        train_labels.append([int(v[1]), int(v[2])])


    model = keras.models.load_model('../data/predict_model')
    print('Training model on new labelled instances...')

    train_values = np.asarray(train_values).astype('float32').reshape(len(train_values), 10, 512)
    train_labels = np.asarray(train_labels)

    print(train_values.shape)
    print(train_labels.shape)
    
    sys.exit()
    model.fit(
        train_values,
        train_labels,
        batch_size=4,
        epochs=10,
    )
    model.save('../../data/predict_model')

print("\nUncertainty Sampling\n")
uncertaintySampling(0.5, 2)
print(evaluateModel())
