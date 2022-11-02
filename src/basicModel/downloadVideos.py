from pytube import YouTube
from pytube import exceptions
import pandas as pd
import tkinter
from pathlib import Path
url_head = 'https://www.youtube.com/watch?v='
pd_reader = pd.read_csv("../../data/final_video_id_list.csv",header=None)
list = []
a=0

for x in range(3476):
    url_tail = str(pd_reader.loc[x][0]).replace("'",'')
    url = url_head+url_tail
    try:
        yt_video = YouTube(url)
        yt_video.streams.get_lowest_resolution().download(output_path='../../data/basicModelVideos/',filename=url_tail+".mp4")
        print(url_tail+' Download successful')
        list.append("'"+url_tail+"'")
        a += 1
    except:
        a = a
    finally:
        if a>=4000:
            break

df = pd.DataFrame(list)
save_path = Path('../../data/basicModelVideos/video_list.csv')
df.to_csv(save_path,header=False,index=False)



