## Deployment 

CLI deployment

### crawlVideos.py

Crawl the internet archive website looking for commercial videos - these links are saved in the file 'videolinks.csv' in the data directory 

### downloadVideos.py

Download each video from 'videolinks.csv' as MP4 as long as size of video < 30 MBs to 'videos' folder

### annotateVideos.py

Select a small subset of downloaded videos at random to be annotated manually - creates pandas dataframe with video id and labels assigned 