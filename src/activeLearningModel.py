from predict import *
import os.path

def uncertaintySampling(range):
    funny_batch = []
    exciting_batch = []
    for i in range(1, 100):
        print(i)
        video_path = "../data/videos/"+str(i)+".mp4"
        if os.path.exists(video_path):
            data = load_sample(video_path)
            result = predict(data)
        else:
            continue

        if 0.45 < result[0] < 0.55:
            exciting_batch.append({i:result[0]})
        if 0.45 < result[1] < 0.55:
            funny_batch.append({i:result[1]})
    print(exciting_batch)
    print(funny_batch)


uncertaintySampling()
