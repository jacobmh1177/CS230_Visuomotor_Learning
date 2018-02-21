import random
import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch import Tensor
import numpy as np

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
        self.batch_prefixes = [os.path.join(data_dir, f)[:-6] for f in self.filenames if f.endswith('_y.npy')]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.batch_prefixes)

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
        scene_rgb = np.load(batch_prefix + "_X_scene_rgb.npy")
        scene_depth = np.expand_dims(np.load(batch_prefix + "_X_scene_d.npy"), axis=-1)
        obj_rgb = np.load(batch_prefix + "_X_obj_rgb.npy")
        obj_depth = np.expand_dims(np.load(batch_prefix + "_X_obj_d.npy"), axis=-1)

        print scene_rgb.shape
        print scene_depth.shape
        print obj_rgb.shape
        print obj_depth.shape
        X = Tensor(np.concatenate([scene_rgb, scene_depth, obj_rgb, obj_depth], axis=-1))

        y = Tensor(np.load(batch_prefix + "_y.npy"))

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