import csv
import os
import random
import subprocess, sys
import pandas as pd

def annotate(i, videos):
    video_id = videos[i]
    try:
        video = '../data/videos/train_videos/'+str(video_id)
    except:
        video = '../data/videos/test_videos/'+str(video_id)
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, video])
    exciting = input("Exciting (0/1): ")
    funny = input("Funny (0/1): ")
    return [str(video_id[:len(video_id)-4]), exciting, funny]

def main():
    videos=[]
    annotations=[]
    NUM_OF_VIDEOS = 100

    for filename in os.listdir("../data/videos/test_videos"):
       with open(os.path.join("../data/videos/test_videos", filename), 'r') as f:
           videos.append(filename)

    random.shuffle(videos)

    for i in range(1, NUM_OF_VIDEOS):
        try:
            annotations.append(annotate(i, videos))
        except:
            print("No video:", i)

    df = pd.DataFrame(annotations, columns = ['id', 'Exciting', 'Funny'])
    df.to_csv("../data/annotatedVideos.csv", sep='\t')
    print(df.head())

if __name__ == '__main__':
    main()
