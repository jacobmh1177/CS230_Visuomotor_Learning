import random
import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch import Tensor
import numpy as np

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
        if "train" in data_dir:
            self.batch_prefixes = self.batch_prefixes[:2]
        if "val" in data_dir:
            self.batch_prefixes = self.batch_prefixes[:1]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.batch_prefixes)

    def crop(self, img, cropx, cropy):
        _, y, x, c = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[:, starty:starty + cropy, startx:startx + cropx, :]

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
        scene_rgb = self.crop(np.load(batch_prefix + "_X_scene_rgb.npz")['arr_0'], 224, 224)
        scene_depth = self.crop(np.load(batch_prefix + "_X_scene_d.npz")['arr_0'], 224, 224)
        obj_rgb = self.crop(np.load(batch_prefix + "_X_obj_rgb.npz")['arr_0'], 224, 224)
        obj_depth = self.crop(np.load(batch_prefix + "_X_obj_d.npz")['arr_0'], 224, 224)

        # print(scene_rgb.shape)
        # print(scene_depth.shape)
        # print(obj_rgb.shape)
        # print(obj_depth.shape)
        X = Tensor(np.concatenate([scene_rgb, scene_depth, obj_rgb, obj_depth], axis=-1))
        #X = np.swapaxes(X, 1, 3)
        X = np.transpose(X, (0, 3, 1, 2))

        y = Tensor(np.load(batch_prefix + "_y.npz")['arr_0'])
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

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            dl = DataLoader(SIMDataset(path, transform=None), batch_size=1, shuffle=True,
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
