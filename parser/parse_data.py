import os

import numpy as np
import cv2
from analysis import parse_protobufs
from ipdb import set_trace as debug
from tqdm import tqdm


DATASETS_ROOT = 'datasets'
OBJ_DATABASE_NAME = 'objects'
DATASET_NAME = 'TeleOpVRSession_2018-02-05_15-44-11/'
MINI_BATCH_SIZE = 10
THRESHOLD = 200
RANDOM_SEED = 12345


class struct():
	pass

def extract_data_from_proto():
	path = struct()
	path.data_folder = os.path.join(DATASETS_ROOT, DATASET_NAME)
	path.data_name = '_SessionStateData.proto'
	data = parse_protobufs(path)
	return data

def parse_data_mini_batch(data, batch_index=None, save=True):
	num_items = []  # number of items in each example
	labels = []
	meta_data = []
	X_scene_rgb = []
	X_scene_d = []
	X_obj_rgb = []
	X_obj_d = []
	num_items_index = 0
	parsed_counter = 0
	batch_num = 0
	with tqdm(total=len(data.states)) as t:
		data_range = np.random.shuffle(range(len(data.states)))
		for i in range(len(data.states)):#range(batch_index * MINI_BATCH_SIZE, (batch_index + 1) * MINI_BATCH_SIZE):
			#print('Image {}'.format(i))
			keep_example = True
			num_items.append(len(data.states[i].items))
			scene_rgb_name = str(data.states[i].snapshot.name)
			scene_depth_name = scene_rgb_name[:-4] + '-Depth.jpg'


			# read in rgb image as rgb and depth image also as rgb
			scene_rgb_img = np.expand_dims(
				cv2.imread(os.path.join(DATASETS_ROOT, scene_rgb_name), 1),
				axis=0
			)
			scene_depth_img = np.expand_dims(
				cv2.imread(os.path.join(DATASETS_ROOT, scene_depth_name), 1),
				axis=0
			)

			for j in range(num_items[num_items_index]):
				rlabel = data.states[i].items[j]
				if rlabel.id == 20:
					continue
				obj_rgb_name = os.path.join(DATASETS_ROOT, OBJ_DATABASE_NAME, "object_" + str(rlabel.id) + ".jpg")
				obj_depth_name = os.path.join(DATASETS_ROOT, OBJ_DATABASE_NAME, "object_" + str(rlabel.id) + "-Depth.jpg")
				obj_rgb_img = np.expand_dims(
					cv2.imread(obj_rgb_name, 1),
					axis=0
				)
				obj_depth_img = np.expand_dims(
					cv2.imread(obj_depth_name, 1),
					axis=0
				)
				# input data (X)
				X_scene_rgb.append(scene_rgb_img)
				X_scene_d.append(scene_depth_img)
				# if scene_rgb_img.shape != obj_rgb_img.shape: # Work-around for missing images
				# 	obj_rgb_img = np.zeros_like(scene_rgb_img)
				# 	obj_depth_img = np.zeros_like(scene_depth_img)
				X_obj_rgb.append(obj_rgb_img)
				X_obj_d.append(obj_depth_img)

				# Output label (Y)
				current_meta_data = [data.states[i].snapshot.name, rlabel.id]
				current_label = [rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch,
								 rlabel.yaw]
				labels.append(current_label)
				meta_data.append(current_meta_data)
				parsed_counter += 1

				if parsed_counter == MINI_BATCH_SIZE:
					X_scene_rgb = np.array(X_scene_rgb).reshape((-1, 299, 299, 3))
					X_scene_d = np.array(X_scene_d).reshape((-1, 299, 299, 3))
					# print np.array(X_obj_rgb).shape
					X_obj_rgb = np.array(X_obj_rgb).reshape((-1, 299, 299, 3))
					X_obj_d = np.array(X_obj_d).reshape((-1, 299, 299, 3))
					y = np.array(labels)
					meta = np.array(meta_data)

					if save:
						save_path = 'datasets/parsed_data/'
						batch_num += 1
						np.savez(save_path + "batch_{}_X_scene_rgb".format(batch_num), X_scene_rgb)
						np.savez(save_path + "batch_{}_X_scene_d".format(batch_num), X_scene_d)
						np.savez(save_path + "batch_{}_X_obj_rgb".format(batch_num), X_obj_rgb)
						np.savez(save_path + "batch_{}_X_obj_d".format(batch_num), X_obj_d)
						np.savez(save_path + "batch_{}_y".format(batch_num), y)
						np.savez(save_path + "batch_{}_meta".format(batch_num), meta)
						if batch_num == THRESHOLD:
							exit("Reached Threshold!")

					labels = []
					meta_data = []
					X_scene_rgb = []
					X_scene_d = []
					X_obj_rgb = []
					X_obj_d = []
					parsed_counter = 0
			num_items_index += 1
			t.update()

	# convert to numpy array
	#print np.array(X_scene_rgb).shape
	# X_scene_rgb = np.array(X_scene_rgb).reshape((-1, 299, 299, 3))
	# X_scene_d = np.array(X_scene_d).reshape((-1, 299, 299))
	# #print np.array(X_obj_rgb).shape
	# X_obj_rgb = np.array(X_obj_rgb).reshape((-1, 299, 299, 3))
	# X_obj_d= np.array(X_obj_d).reshape((-1, 299, 299))
	# y = np.array(labels)
	# meta = np.array(meta_data)
    #
	# if save:
	# 	save_path = 'datasets/parsed_data/'
	# 	np.save(save_path + "batch_{}_X_scene_rgb.npy".format(batch_index + 1), X_scene_rgb)
	# 	np.save(save_path + "batch_{}_X_scene_d.npy".format(batch_index + 1), X_scene_d)
	# 	np.save(save_path + "batch_{}_X_obj_rgb.npy".format(batch_index + 1), X_obj_rgb)
	# 	np.save(save_path + "batch_{}_X_obj_d.npy".format(batch_index + 1), X_obj_d)
	# 	np.save(save_path + "batch_{}_y.npy".format(batch_index + 1), y)
	# 	np.save(save_path + "batch_{}_meta.npy".format(batch_index + 1), meta)

if __name__ == '__main__':
	random.seed(RANDOM_SEED)
	data = extract_data_from_proto()
	parse_data_mini_batch(data)
	# num_mini_batches = len(data.states) / MINI_BATCH_SIZE
	# for batch_index in range(num_mini_batches):
	# 	print("Parsing data for batch #{}".format(batch_index))
	# 	parse_data_mini_batch(data, batch_index, save=True)
