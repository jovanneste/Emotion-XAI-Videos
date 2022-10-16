import csv
import os
import random
import subprocess, sys

videos=[]
for filename in os.listdir("../data/videos"):
   with open(os.path.join("../data/videos", filename), 'r') as f:
       videos.append(filename)


random.shuffle(videos)
print(videos[0])


f = '../data/videos/'+str(videos[0])


opener = "open" if sys.platform == "darwin" else "xdg-open"
subprocess.call([opener, f])