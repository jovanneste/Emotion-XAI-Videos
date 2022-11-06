import csv
import os
import random
import subprocess, sys
import pandas as pd

videos=[]
annotations=[]
print('what')
NUM_OF_VIDEOS = 100

def annotate(i):
    print('the')
    video_id = videos[i]
    video = '../data/videos/'+str(video_id)
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, video])
    print("Video: ", i)
    exciting = input("Exciting (0/1): ")
    funny = input("Funny (0/1): ")
    return [str(video_id[:len(video_id)-4]), exciting, funny]


for filename in os.listdir("../data/videos"):
   with open(os.path.join("../data/videos", filename), 'r') as f:
       videos.append(filename)

random.shuffle(videos)

for i in range(NUM_OF_VIDEOS):
	annotations.append(annotate(i))

df = pd.DataFrame(annotations, columns = ['id', 'Exciting', 'Funny'])
df.to_csv("../data/annotatedVideos.csv", sep='\t')
print(df.head())
