import requests
from bs4 import BeautifulSoup
import urllib.request
import csv


MAX_AD_SIZE = 30
count = 1
with open('../data/videolinks.csv','r') as f:
	reader_obj = csv.reader(f)
	for row in reader_obj:
		print("Video: " + str(count))

		videoName = '../data/videos/' + str(count) + '.mp4'
		print("Video: " + videoName)
		url = "https://archive.org" + str(row[0])

		r = requests.get(url, timeout = 30)
		soup = BeautifulSoup(r.content, 'html.parser')
					
		links = soup.find_all('a', class_='stealth')
		for link in links:
			if(str(link['href']).endswith('mp4')):
				try:
					if float(str(link['title'])[:-1]) <= MAX_AD_SIZE:
						l = 'https://archive.org' + str(link['href'])
						urllib.request.urlretrieve(l, videoName)
						print("SUCCESS")
						count += 1
						break;
					else:
						print("FAILED - video too large")

				except:
					print("FAILED - malformed html")
					break;
		
		
print("Videos downloaded: " + str(count))