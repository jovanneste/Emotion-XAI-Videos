import pandas as pd
import os


df = pd.read_csv('../../data/annotatedVideos.csv', delim_whitespace=True)

for index, row in df.iterrows():
	try:
		idn = row['id']
		video = '../../data/videos/' + str(idn) + '.mp4'
		new_path = '../../data/videos/test_videos/' + str(idn) + '.mp4'
		os.rename(video, new_path)
		print('Moved:', video)
	except:
		print('Video does not exist')
