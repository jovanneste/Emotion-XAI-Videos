import os
import subprocess
import sys
import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity

# video test_videos/2496.mp4 is correclty labelled as exciting=1, funny=0 - WHY


video_id = '2496.mp4'
video_path = '../data/videos/test_videos/'+str(video_id)

# opener = "open" if sys.platform == "darwin" else "xdg-open"
# subprocess.call([opener, video_name])


def getKeyFrames(video_path):
    print("Removing files from last video...")
    files = glob.glob('../data/frames/*')
    for f in files:
        os.remove(f)

    print("Frame extraction for new video")

    cap = cv2.VideoCapture(video_path)
    frame_seq = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_length = round(frame_seq/fps)

    print("Video:", video_path)
    print("Number of frames:", frame_seq)

    j = 1
    for i in range(1, int(frame_seq), 2):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('../data/frames/frame'+str(j)+'.jpg', frame)
        j+=1

    cap.release()
    cv2.destroyAllWindows()


def test():
	for i in range(1, 100, 10):
		before = cv2.imread('../data/frames/frame' + str(i) + '.jpg')
		after = cv2.imread('../data/frames/frame' + str(i+10) + '.jpg')

		before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
		after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

		(score, diff) = structural_similarity(before_gray, after_gray, full=True)
		if score<0.6:
			print("Image similarity between frame " + str(i) + " and " + str(i+10) + ": "+str(score))


test()
