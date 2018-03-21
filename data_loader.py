import random
import os

import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch import Tensor
import numpy as np
from transforms3d.taitbryan import axangle2euler
import torch

import utils

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class SIMDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.batch_prefixes = [os.path.join(data_dir, f)[:-6] for f in self.filenames if f.endswith('_y.npz')]
#        if "dev" in data_dir:
#            self.batch_prefixes = self.batch_prefixes[:10]
#        elif "traindev" in data_dir:
#            self.batch_prefixes = self.batch_prefixes[:10]
#        else:
#            self.batch_prefixes = self.batch_prefixes[:10]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.batch_prefixes)

    def crop(self, img, cropx, cropy):
        _, y, x, c = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[:, starty:starty + cropy, startx:startx + cropx, :]

    def transform_pose(self, labels):
        poses = labels[:, 3:]
        for i, pose in enumerate(poses):
            rotation_axis = pose / np.linalg.norm(pose)
            angle = np.linalg.norm(pose)
            z, y, x = axangle2euler(rotation_axis, angle)
            labels[i, 3] = x #* (180./np.pi)
            labels[i, 4] = y #* (180./np.pi)
            labels[i, 5] = z #* (180./np.pi)
            #print x, y , z
        return labels

    def normalize_inputs(self, inputs):
        inpts = inputs/255.0 # scale inputs to [0, 1]
        means = [0.485, 0.456, 0.406]
        stds = [0.299, 0.224, 0.225]
        for i in range(inputs.shape[0]):
            for j in range(len(means)):
                inputs[i, :, :, j] = (inputs[i, :, :, j] - means[j]) / stds[j]
        return inputs
        
    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        batch_prefix = self.batch_prefixes[idx]
        scene_rgb = self.normalize_inputs(self.crop(np.load(batch_prefix + "_X_scene_rgb.npz")['arr_0'], 224, 224))
        scene_depth = self.normalize_inputs(self.crop(np.load(batch_prefix + "_X_scene_d.npz")['arr_0'], 224, 224))
        obj_rgb = self.normalize_inputs(self.crop(np.load(batch_prefix + "_X_obj_rgb.npz")['arr_0'], 224, 224))
        obj_depth = self.normalize_inputs(self.crop(np.load(batch_prefix + "_X_obj_d.npz")['arr_0'], 224, 224))

        # print(scene_rgb.shape)
        # print(scene_depth.shape)
        # print(obj_rgb.shape)
        # print(obj_depth.shape)
        X = Tensor(np.concatenate([scene_rgb, scene_depth, obj_rgb, obj_depth], axis=-1))
        #X = np.swapaxes(X, 1, 3)
        X = Tensor(np.transpose(X, (0, 3, 1, 2)))

        y = Tensor(self.transform_pose(np.load(batch_prefix + "_y.npz")['arr_0']))
        # print(X.shape, y.shape)

        #return (Tensor(scene_rgb), Tensor(scene_depth), Tensor(obj_rgb), Tensor(obj_depth)), y
        return X, y

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'traindev', 'dev', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            dl = DataLoader(SIMDataset(path, transform=None), batch_size=params.batch_size, shuffle=True,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders


if __name__ == '__main__':
    json_path = os.path.join('params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = True
    loader = fetch_dataloader(['train'], 'datasets/parsed_data', params)['train']
    for (i, batch) in enumerate(loader):
        print(i)
