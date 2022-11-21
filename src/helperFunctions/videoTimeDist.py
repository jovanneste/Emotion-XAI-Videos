import cv2
import datetime
import os
import matplotlib.pyplot as plt

times = []
kept = []
i = 0

delete = True
for filename in os.listdir("../../data/videos/train_videos/"):
    with open(os.path.join("../../data/videos/train_videos/", filename), 'r') as f:
        try:
            data = cv2.VideoCapture(
                "../../data/videos/train_videos/" + filename)
            frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = data.get(cv2.CAP_PROP_FPS)
            seconds = round(frames/fps)
            if seconds > 300:
                i += 1
            else:
                times.append(seconds)
            if (delete):
                if seconds > 60:
                    print("Delete")
                    os.remove("../../data/videos/train_videos/"+str(filename))
                else:
                    kept.append(filename)
        except:
            print("Video failed")


fig, ax = plt.subplots(1, 1)
ax.hist(times, bins=9)
ax.set_title("Video time distribution, over 300 seconds=" + str(i))
ax.set_ylabel('Number of videos')
ax.set_xlabel('Length of video (seconds)')

plt.savefig('videoTimeDistribution.png')
plt.show()


print("Kept", len(kept))
