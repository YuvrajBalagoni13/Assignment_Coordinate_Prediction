import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import os
import json
import logging
from typing import Callable, Optional

class CoordinateDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_dir : str,
            coordinates_json_path: str,
            device: str,
            transforms: Optional[Callable] = None
    ) -> None:
        assert os.path.isdir(image_dir), f"Image directory {image_dir} nor found"
        assert os.path.isfile(coordinates_json_path), f"json file {coordinates_json_path} not found"
        self.image_dir = image_dir
        self.transforms = transforms

        image_list = os.listdir(image_dir)
        self.image_list = [image[:-4] for image in image_list]  # gets list of image ids

        with open(coordinates_json_path, "r") as f:
            self.coordinates_dict = json.load(f)
        
    def __len__(self) -> None:
        return len(self.image_list)
    
    def __getitem__(self, idx) -> dict:
        sample_name = self.image_list[idx]
        coordinates = np.array(self.coordinates_dict[sample_name])
        coordinates = coordinates / 50
        try:
            image = Image.open(os.path.join(self.image_dir, f"{sample_name}.png"))
            if self.transforms:
                image = self.transforms(image)
                # image /= 255.0
            return {
                'image' : image,
                'coordinates' : coordinates
            }
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Could not read files for index {idx}. File: {e.filename}. Skipping.")
            return self.__getitem__((idx + 1) % self.__len__())   # Increments index and tries again with the next valid pair
        
        
def get_dataloaders(
        dataset_dir: str,
        batch_size: int,
        shuffle: bool,
        device: str
) -> dict:
    """
    gives a dictionary of dataloaders for train, val & test

    Args:
        dataset_dir (str): directory path for the dataset
        batch_size (int): batch size of the dataloaders
        shuffle (bool): to shuffle or not
        device (str): device
    
    Returns:
        dataloaders (dict): dictionary of dataloaders for train, val & test
    """
    coordinates_json_path = os.path.join(dataset_dir, "target_coordinates.json")
    datasets = {x : CoordinateDataset(os.path.join(dataset_dir, "images", x), coordinates_json_path, device, transforms.ToTensor()) 
                for x in ['train', 'val', 'test']}
    
    dataloaders = {x : torch.utils.data.DataLoader(datasets[x], batch_size, shuffle)
                   for x in ['train', 'val', 'test']}
    
    return dataloaders
