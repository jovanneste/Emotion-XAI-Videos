import os
import subprocess
import sys
import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity
from PIL import Image

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


	# assume dependence between frames
    for i in range(int(frame_seq)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('../data/frames/frame'+str(i)+'.jpg', frame)


    cap.release()
    cv2.destroyAllWindows()


def frameSimilarity(frame1, frame2):
	frame1 = cv2.imread('../data/frames/frame' + str(frame1) + '.jpg')
	frame2 = cv2.imread('../data/frames/frame' + str(frame2) + '.jpg')
	before = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	after = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	(score, diff) = structural_similarity(before, after, full=True)
	return score

# n: how many frame we want to use to represent the video
def quantiseVideo(n):
	print("Quantising video")
	frame_num = 904
	frames = {}
	for i in range(0, frame_num-5, 10):
		score = frameSimilarity(i, i+10)
		if score<0.5:
			frames.update({i:score})
	frames = {k: v for k, v in sorted(frames.items(), key=lambda item: item[1])}
	frames = [k for k in frames.keys()][:n]

	ranges = []

	for frame in frames:
		r = [frame]
		for i in range(1,frame_num):
			try:
				if (frameSimilarity(frame, frame-i))<0.5:
					r.append(frame-i)
					break
			except:
				r.append(1)
				break
		for i in range(1,frame_num):
			try:
				if (frameSimilarity(frame, frame+i))<0.5:
					r.append(frame+i)
					break
			except:
				r.append(frame_num)
				break
		ranges.append(r)
	return ranges




	# for i in frames.keys():
	# 	print(i)
	# 	im = Image.open('../data/frames/frame' + str(i) + '.jpg')
	# 	im.show()

#getKeyFrames(video_path)
#print(quantiseVideo(10))

ranges = [[40, 1, 49], [320, 272, 323], [350, 337, 353], [810, 809, 812],
		[260, 231, 268], [850, 822, 854], [860, 853, 866], [430, 382, 431],
		[380, 371, 382], [750, 749, 752]]
