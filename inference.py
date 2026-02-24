import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import argparse
import os
from tqdm.auto import tqdm

import model

class CoordinateInference:
    def __init__(self, model_path: str, device:str = 'cpu') -> None:
        self.device = device

        model_type = "_".join(model_path.split("/")[-1].split(".")[0].split("_")[:-1])
        if model_type == "simple_conv":
            self.model = model.CoordinateRegressor(input_shape=1, mlp_units=128, output_shape=2)
        elif model_type == "spatial_softmax":
            self.model = model.CoordinateRegressorSpatialSoftmax(input_shape=1, output_shape=2)
        else:
            self.model = model.NeuralNetRegressor(input_shape=1, mlp_units=128, output_shape=2)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        np_image = np.array(image, dtype=np.float32) / 255.0
        torch_image = torch.from_numpy(np_image).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(torch_image).cpu().numpy()
        output = (output * 50).squeeze().round().astype(int)
        return output
    
def test_models(
        checkpoint_path: str,
        test_data_path: str,
        target_coordinate_path: str
) -> None:
    inference = CoordinateInference(checkpoint_path)
    with open(target_coordinate_path, "r") as f:
        target_coordinates = json.load(f)
    
    img_id_list = [images[:-4] for images in os.listdir(test_data_path)]
    accuracy = 0.0
    for id in tqdm(img_id_list):
        pred = inference(os.path.join(test_data_path, f"{id}.png"))
        gt = target_coordinates[id]
        accuracy += (pred == gt).all()
    accuracy /= len(os.listdir(test_data_path))
    print(f"test accuracy is {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type = str,
        required = True,
        help = "checkpoint path"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help = "Do you want do testing of model"
    )

    parser.add_argument(
        "--inference",
        action="store_true",
        help = "want to do inference only"
    )

    parser.add_argument(
        "--test_data_path",
        type = str
    )
    
    parser.add_argument(
        "--target_coordinate_path",
        type = str
    )

    parser.add_argument(
        "--infer_img_path",
        type = str,
        default = None
    )

    args = parser.parse_args()
    if args.test:
        test_models(
            checkpoint_path = args.checkpoint_path,
            test_data_path = args.test_data_path,
            target_coordinate_path = args.target_coordinate_path
        )

    if args.inference:
        inference = CoordinateInference(args.checkpoint_path)
        x, y = inference(args.infer_img_path)
        print(f"x coordinate = {x} | y coordinate = {y}")


"""
python inference.py \
--test \
--checkpoint_path checkpoints/neural_nets_14.pt \
--test_data_path Coordinate_Dataset/images/test \
--target_coordinate_path Coordinate_Dataset/target_coordinates.json \
--infer_img_path None
"""