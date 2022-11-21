import os 
import subprocess, sys
import cv2
import numpy as np


video_id = '239.mp4'
video_name = '../data/videos/train_videos/'+str(video_id)
print(video_id)

# opener = "open" if sys.platform == "darwin" else "xdg-open"
# subprocess.call([opener, video_path])

# for fno in range(0, 10, 2):
# 	cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
# 	_, image1 = cap.read()
# 	cap.set(cv2.CAP_PROP_POS_FRAMES, fno+1)
# 	_, image2 = cap.read()

# 	image1 = cv2.imread(image1, 0)
# 	image2 = cv2.imread(image2, 0)

# Calculate the per-element absolute difference between 
# two arrays or between an array and a scalar
	# diff = 255 - cv2.absdiff(image1, image2)

	# cv2.imshow('diff', diff)
	# cv2.waitKey()



#Set frame_no in range 0.0-1.0
#In this example we have a video of 30 seconds having 25 frames per seconds, thus we have 750 frames.
#The examined frame must get a value from 0 to 749.
#For more info about the video flags see here: https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
#Here we select the last frame as frame sequence=749. In case you want to select other frame change value 749.
#BE CAREFUL! Each video has different time length and frame rate. 
#So make sure that you have the right parameters for the right video!

cap = cv2.VideoCapture(video_name)
frame_seq = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
time_length = round(frame_seq/fps)
frame_no = (frame_seq /(time_length*fps))
print(frame_no)

#The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
#Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
#The second argument defines the frame number in range 0.0-1.0
cap.set(2,frame_no);

#Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
ret, frame = cap.read()

#Set grayscale colorspace for the frame. 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Cut the video extension to have the name of the video
my_video_name = video_name.split(".")[0]
print(my_video_name)

#Display the resulting frame
cv2.imshow(my_video_name+' frame '+ str(frame_seq),gray)

#Set waitKey 
cv2.waitKey()

#Store this frame to an image
cv2.imwrite(my_video_name+'_frame_'+str(frame_seq)+'.jpg',gray)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# video_capture = cv2.VideoCapture(video_path) 
# fps = int(video_capture.get(5))
# print("Frame Rate : ",fps,"frames per second")  
 
#   # Get frame count
# frame_count = video_capture.get(7)
# print("Frame count : ", frame_count)

# while(video_capture.isOpened()):
#   ret, frame = video_capture.read()
#   if ret == True:
#     cv2.imshow('Frame',frame)
#     k = cv2.waitKey(200000)
#     # 113 is ASCII code for q key
#     if k == 113:
#       break
#   else:
    # break
