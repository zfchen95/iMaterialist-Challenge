import json
import sys
import urllib
import os

def main():
	if len(sys.argv) != 4:
		print("Usage: python geturl.py <filepath> <dest> <number_images>")
		return
	filepath = sys.argv[1]
	directory = sys.argv[2]
	num_img = sys.argv[3]
	json_file = open(filepath)
	json_str = json_file.read()
	data = json.loads(json_str)

	images = data['images']
	urls = []
	annotation = []
	images_dict = {}
	class_dict = {}

	if not os.path.exists(directory):
		os.makedirs(directory)

	for i in range(0, len(images)):
		entry = images[i]
		if data['annotations'][i]['label_id'] not in class_dict:
			class_dict[data['annotations'][i]['label_id']] = [(entry['image_id'], entry['url'][0])]
		else:
			class_dict[data['annotations'][i]['label_id']].append((entry['image_id'], entry['url'][0]))
		#print(class_dict)
	# dict key: label_id,  value: list((image_id, url))
	with open('annotations.txt', 'w+') as file:
		for key,val in class_dict.iteritems():
			print("Downloading class " + str(key))
			for i in range(0, min(num_img, len(val))):
				download_img = urllib.URLopener()
				try:
					path = directory + str(val[i][0]) + ".jpg"
					download_img.retrieve(val[i][1], path)
					#store annotation of the image
					file.write(str(val[i][1])+ "," + str(key))
				except IOError:
					print("Can not read image "+ str(val[i][0])+ " from url...")
					i-=1
					
					
				





	# for i in range(0, len(images)):
	# 	entry = images[i]
	# 	urls.append((entry['url'], entry['image_id']))
	# 	annotation.append((data['annotations'][i]['image_id'],data['annotations'][i]['label_id']))
	# 	download_img = urllib.URLopener()
	# 	try:
	# 		path = directory + str(entry['image_id']) + ".jpg"
	# 		download_img.retrieve(entry['url'][0], path)
	# 	except IOError:
	# 		print("Reading " + str(entry['image_id'])+" error")
	# 		continue
	# with open('./annotations.txt', 'r+') as file:
	# 	for i in range(0, len(annotations)):
	# 		file.write(str(annotations[i][0]) + ", "+ str(annotations[i][1]))

if __name__ == '__main__':
	main()
