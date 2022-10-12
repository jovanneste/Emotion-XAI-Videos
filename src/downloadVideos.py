import requests
from bs4 import BeautifulSoup
import urllib.request
import csv


MAX_AD_SIZE = 100
allcount = 0
failedcount = 0 
with open('../data/videolinks.csv','r') as f:
	reader_obj = csv.reader(f)
	for row in reader_obj:
		allcount+=1
		videoName = '../data/adVideos/' + str(row[0])[9:] + '.mp4'
		print("Video: " + videoName)
		url = "https://archive.org" + str(row[0])

		r = requests.get(url, timeout = 10)
		soup = BeautifulSoup(r.content, 'html.parser')
					
		links = soup.find_all('a', class_='stealth')
		for link in links:
			if(str(link['href']).endswith('mp4')):
				try:
					if float(str(link['title'])[:-1]) <= MAX_AD_SIZE:
						l = 'https://archive.org' + str(link['href'])
						urllib.request.urlretrieve(l, videoName)
						print("SUCCESS")
					else:
						print("FAILED - video too large")
						failedcount+=1

				except:
					print("FAILED - malformed html")
					failedcount+=1
		
			
print(allcount)
print(failedcount)
