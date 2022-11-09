import cv2
import datetime
import os
import matplotlib.pyplot as plt

times = []
i=0
for filename in os.listdir("../../data/videos/train_videos/"):
   with open(os.path.join("../../data/videos/train_videos/", filename), 'r') as f:
       try:
           data = cv2.VideoCapture("../../data/videos/train_videos/" + filename)
           frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
           fps = data.get(cv2.CAP_PROP_FPS)
           seconds = round(frames/fps)
           if seconds > 300:
               i+=1
           else:
               times.append(seconds)
       except:
           print("Video failed")


fig, ax = plt.subplots(1, 1)
ax.hist(times, bins=9)
ax.set_title("Video time distribution, over 300 seconds=" + str(i))
ax.set_ylabel('Number of videos')
ax.set_xlabel('Lenght of video (seconds)')

plt.savefig('videoTimeDistribution.png')
plt.show()
