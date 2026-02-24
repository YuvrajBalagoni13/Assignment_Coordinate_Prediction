import os
import torch
import numpy as np
import json
from PIL import Image
import argparse
from tqdm.auto import tqdm
import random

class CreateDataset:
    """
    this class generates all possible 50x50 grayscale images where only one pixel has 255 value & rest all have 0.
    at the end we will have 2500 images with each image having unique id & a json file which maps id to coordinates.
    """
    def __init__(
            self,
            image_size: int,
            dataset_dir : str,
            train_split: float,
            random_seed: int = 42
    ) -> None:
        
        self.image_size = image_size

        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.image_dir = os.path.join(self.dataset_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.train_len = int(train_split * len(self))
        val_test_split = round((1 - train_split) / 2, 1)
        self.val_len = self.train_len + int(val_test_split * len(self))
        
        self.train_dir = os.path.join(self.image_dir, "train")
        self.val_dir = os.path.join(self.image_dir, "val")
        self.test_dir = os.path.join(self.image_dir, "test")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

        np.random.seed(random_seed)

    def __len__(self) -> int:
        return self.image_size ** 2
    
    def create_samples(self) -> None:

        # creating a list of possible coordinate values & 
        # randomly shuffling it so that we have randomly shuffled values of coordinates.
        coordinates_list = []
        for i in range(self.image_size):
            for j in range(self.image_size):
                coordinates_list.append([i, j])
        
        random.shuffle(coordinates_list)

        target_coordinate_dict = {}
        
        # Initializing image only once & just turning pixels on & off for every sample.
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        for i in tqdm(range(len(coordinates_list))):
            x, y = coordinates_list[i]
            sample_name = f"{i:06}"

            image[y, x] = 255
            pil_image = Image.fromarray(image)
            if i < self.train_len:
                pil_image.save(os.path.join(self.train_dir, f"{sample_name}.png"))
            elif i < self.val_len:
                pil_image.save(os.path.join(self.val_dir, f"{sample_name}.png"))
            else:
                pil_image.save(os.path.join(self.test_dir, f"{sample_name}.png"))
            image[y, x] = 0

            target_coordinate_dict[sample_name] = [x, y]

        json_file_path = os.path.join(self.dataset_dir, "target_coordinates.json")
        with open(json_file_path, "w") as f:
            json.dump(target_coordinate_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="creating dataset"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        required=True,
        default=50,
        help="size of the images in dataset"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        default="Coordinate_Dataset",
        help="directory where dataset will be saved"
    )
    
    parser.add_argument(
        "--train_split",
        type=float,
        required=True,
        default=0.8,
        help="splitting the dataset into train, val & test"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for random shuffling & sampling"
    )

    args = parser.parse_args()
    create_dataset = CreateDataset(
        image_size = args.image_size,
        dataset_dir = args.dataset_dir,
        train_split = args.train_split,
        random_seed = args.seed
    )
    create_dataset.create_samples()


"""
python create_dataset.py \
--image_size 50 \
--dataset_dir Coordinate_Dataset \
--train_split 0.8 \
--seed 42
"""