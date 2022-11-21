import os 
import subprocess, sys
import cv2
import numpy as np

print("Removing frames from last video")
for filename in os.listdir("../data/frames/"):
    with open(os.path.join("../data/frames/", filename), 'r') as f:
    	print("Deleting", filename)
    	os.remove(filename)

video_id = '239.mp4'
video_name = '../data/videos/train_videos/'+str(video_id)

# opener = "open" if sys.platform == "darwin" else "xdg-open"
# subprocess.call([opener, video_name])

cap = cv2.VideoCapture(video_name)
frame_seq = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
time_length = round(frame_seq/fps)

print("Video:", video_id)
print("Number of frames:", frame_seq)

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('../data/frames/frame'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()


