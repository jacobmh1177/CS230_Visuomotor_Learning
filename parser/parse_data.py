import os

import numpy as np
import cv2
from analysis import parse_protobufs
from ipdb import set_trace as debug


DATASETS_ROOT = 'datasets'
OBJ_DATABASE_NAME = 'objects'
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
	meta_data = []
	X_scene_rgb = []
	X_scene_d = []
	X_obj_rgb = []
	X_obj_d = []
	num_items_index = 0
	for i in range(batch_index * MINI_BATCH_SIZE, (batch_index + 1) * MINI_BATCH_SIZE):
		print('Image {}'.format(i))
		num_items.append(len(data.states[i].items))
		scene_rgb_name = str(data.states[i].snapshot.name)
		scene_depth_name = scene_rgb_name[:-4] + '-Depth.jpg'


		# read in rgb and depth images and add a new axis to them to indicate which snapshot index for each image
		scene_rgb_img = np.expand_dims(
			cv2.imread(os.path.join(DATASETS_ROOT, scene_rgb_name), 1),
			axis=0
		)
		scene_depth_img = np.expand_dims(
			cv2.imread(os.path.join(DATASETS_ROOT, scene_depth_name), 0),
			axis=0
		)

		for j in range(num_items[num_items_index]):
			obj_rgb_name = os.path.join(DATASETS_ROOT, OBJ_DATABASE_NAME, "object_" + str(rlabel.id) + ".jpg")
			obj_depth_name = os.path.join(DATASETS_ROOT, OBJ_DATABASE_NAME, "object_" + str(rlabel.id) + "-Depth.jpg")
			obj_rgb_img = np.expand_dims(
				cv2.imread(obj_rgb_name, 1),
				axis=0
			)
			obj_depth_img = np.expand_dims(
				cv2.imread(obj_depth_name, 0),
				axis=0
			)
			# input data (X)
			X_scene_rgb.append(scene_rgb_img)
			X_scene_d.append(scene_depth_img)
			X_obj_rgb.append(obj_rgb_img)
			X_obj_d.append(obj_depth_img)

			# Output label (Y)
			rlabel = data.states[i].items[j]
			current_meta_data = [data.states[i].snapshot.name, rlabel.id]
			current_label = [rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch,
							 rlabel.yaw]
			labels.append(current_label)
			meta_data.append(current_meta_data)
		num_items_index += 1

	# convert to numpy array
	X_scene_rgb = np.array(X_scene_rgb).reshape((-1, 299, 299, 3))
	X_scene_d = np.array(X_scene_d).reshape((-1, 299, 299))
	X_obj_rgb = np.array(X_obj_rgb).reshape((-1, 299, 299))
	X_obj_d= np.array(X_obj_d).reshape((-1, 299, 299))
	y = np.array(labels)
	meta = np.array(meta_data)

	if save:
		save_path = 'datasets/parsed_data/'
		np.save(save_path + "batch_{}_X_scene_rgb.npy".format(batch_index + 1), X_scene_rgb)
		np.save(save_path + "batch_{}_X_scene_d.npy".format(batch_index + 1), X_scene_d)
		np.save(save_path + "batch_{}_X_obj_rgb.npy".format(batch_index + 1), X_obj_rgb)
		np.save(save_path + "batch_{}_X_obj_d.npy".format(batch_index + 1), X_obj_d)
		np.save(save_path + "batch_{}_y.npy".format(batch_index + 1), y)
		np.save(save_path + "batch_{}_meta.npy".format(batch_index + 1), meta)

if __name__ == '__main__':
	data = extract_data_from_proto()
	num_mini_batches = len(data.states) / MINI_BATCH_SIZE
	for batch_index in range(num_mini_batches):
		print("Parsing data for batch #{}".format(batch_index))
		parse_data_mini_batch(data, batch_index, save=True)
