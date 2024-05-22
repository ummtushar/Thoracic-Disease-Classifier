import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import requests
import io
from os import path
from typing import Tuple, List
from pathlib import Path
import os


class ImageDataset:
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    """

    def __init__(self, x: Path, y: Path) -> None:
        # Target labels
        self.targets = ImageDataset.load_numpy_arr_from_npy(y) 
        # Images
        self.imgs = ImageDataset.load_numpy_arr_from_npy(x)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        # Template code
        image = torch.from_numpy(self.imgs[idx] / 255).float() 
        label = self.targets[idx]

        # Preprocessing
        # Metrics for Normalization of the images
        mean = image.mean()
        std = image.std()

        # Compose: Composes several transforms together (torch documentation)
        compose = T.Compose([
            T.Normalize(mean, std),  # Normalization
            T.Resize(156),  # Resizing to 156x156
            T.CenterCrop(128),  # Cropping to focus on the center 128x128 region
            T.Lambda(lambda x: TF.rotate(x, angle=90)),  # Rotating by 90 degrees
            T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with a 50% probability
            T.RandomVerticalFlip(p=0.5),  # Random vertical flip with a 50% probability
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # Adding random noise
        ])

        # Apply the transformation done by composee
        image = compose(image)
        
        return image, label
    
    def get_labels(self) -> List[np.ndarray]:
        return self.targets.tolist()

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.

        Input:
        path: local path of file

        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)


def load_numpy_arr_from_url(url: str) -> np.ndarray:
    """
    Loads a numpy array from surfdrive.

    Input:
    url: Download link of dataset

    Outputs:
    dataset: numpy array with input features or labels
    """

    response = requests.get(url)
    response.raise_for_status()

    return np.load(io.BytesIO(response.content))


if __name__ == "__main__":
    cwd = os.getcwd()
    if path.exists(path.join(cwd + "data/")):
        print("Data directory exists, files may be overwritten!")
    else:
        os.mkdir(path.join(cwd, "data/"))
    ### Load labels
    train_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/i6MvQ8nqoiQ9Tci/download"
    )
    np.save("data/Y_train.npy", train_y)
    test_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/wLXiOjVAW4AWlXY/download"
    )
    np.save("data/Y_test.npy", test_y)
    ### Load data
    train_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/4rwSf9SYO1ydGtK/download"
    )
    np.save("data/X_train.npy", train_x)
    test_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/dvY2LpvFo6dHef0/download"
    )
    np.save("data/X_test.npy", test_x)
