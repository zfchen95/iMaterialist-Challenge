import json
import sys
import urllib
import os

def main():
	if len(sys.argv) != 3:
		print("Usage: python geturl.py <filepath> <dest>")
		return
	filepath = sys.argv[1]
	directory = sys.argv[2]
	json_file = open(filepath)
	json_str = json_file.read().decode("UTF-8")
	data = json.loads(json_str)

	if len(data) == 1:
		images = data['images']
		urls = []
		if not os.path.exists(directory):
			os.makedirs(directory)

		for i in range(0, len(images)):
			entry = images[i]
			urls.append((entry['url'], entry['image_id']))
			download_img = urllib.URLopener()
			try:
				path = directory + str(entry['image_id']) + ".jpg"
				download_img.retrieve(entry['url'][0],path)
			except:
				print("Reading " + str(entry['image_id'])+" error")
				break
	else:
		images = data['images']
		urls = []
		annotation = []
		if not os.path.exists(directory):
			os.makedirs(directory)

		for i in range(0, len(images)):
			entry = images[i]
			urls.append((entry['url'], entry['image_id']))
			annotation.append((data['annotations'][i]['image_id'],data['annotations'][i]['label_id']))
			download_img = urllib.URLopener()
			try:
				path = directory + str(entry['image_id']) + ".jpg"
				download_img.retrieve(entry['url'][0], path)
			except IOError:
				print("Reading " + str(entry['image_id'])+" error")
				continue
		with open('./annotations.txt', 'r+') as file:
			for i in range(0, len(annotations)):
				file.write(str(annotations[i][0]) + ", "+ str(annotations[i][1]))

if __name__ == '__main__':
	main()