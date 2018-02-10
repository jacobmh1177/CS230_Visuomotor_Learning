import os

import numpy as np
import cv2
from analysis import parse_protobufs
from ipdb import set_trace as debug


DATASETS_ROOT = 'datasets'
DATASET_NAME = 'TeleOpVRSession_2018-02-05_15-44-11/'
MINI_BATCH_SIZE = 100


class struct():
	pass

def extract_data_from_proto():
	path = struct()
	path.data_folder = os.path.join(DATASETS_ROOT, DATASET_NAME)
	path.data_name = '_SessionStateData.proto'
	data = parse_protobufs(path)
	return data

def parse_data_mini_batch(data, batch_index, save=True):
	num_items = []  # number of items in each example
	labels = []
	X_rgb = []
	X_d = []
	num_items_index = 0
	for i in range(batch_index * MINI_BATCH_SIZE, (batch_index + 1) * MINI_BATCH_SIZE):
		print('Image {}'.format(i))
		num_items.append(len(data.states[i].items))
		img_name = str(data.states[i].snapshot.name)
		depth_name = img_name[:-4] + '-Depth.jpg'

		# read in rgb and depth images and add a new axis to them to indicate which snapshot index for each image
		rgb_img = np.expand_dims(
			cv2.imread(os.path.join(DATASETS_ROOT, img_name), 1),
			axis=0
		)
		depth_img = np.expand_dims(
			cv2.imread(os.path.join(DATASETS_ROOT, depth_name), 0),
			axis=0)

		for j in range(num_items[num_items_index]):
			# input data (X)
			X_rgb.append(rgb_img)
			X_d.append(depth_img)

			# Output label (Y)
			rlabel = data.states[i].items[j]
			current_label = [data.states[i].snapshot.name, rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch,
							 rlabel.yaw]
			labels.append(current_label)
		num_items_index += 1

	# convert to numpy array
	X_rgb = np.array(X_rgb).reshape((-1, 299, 299, 3))
	X_d = np.array(X_d).reshape((-1, 299, 299))
	y = np.array(labels)

	if save:
		save_path = 'datasets/parsed_data/'
		np.save(save_path + "batch_{}_X_rgb.npy".format(batch_index + 1), X_rgb)
		np.save(save_path + "batch_{}_X_d.npy".format(batch_index + 1), X_d)
		np.save(save_path + "batch_{}_y.npy".format(batch_index + 1), y)

if __name__ == '__main__':
	data = extract_data_from_proto()
	num_mini_batches = len(data.states) / MINI_BATCH_SIZE
	for batch_index in range(num_mini_batches):
		print("Parsing data for batch #{}".format(batch_index))
		parse_data_mini_batch(data, batch_index, save=True)

