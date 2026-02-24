import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

import model

class CoordinateInference:
    def __init__(self, model_path: str, device:str = 'cpu') -> None:
        self.device = device
        self.model = model.CoordinateRegressor(input_shape=1, mlp_units=128, output_shape=2)
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
    

if __name__ == "__main__":
    inference = CoordinateInference("checkpoints/base_model_again_61.pt")
    x, y = inference("Coordinate_Dataset/images/test/002251.png")
    print(f"x coordinate = {x} | y coordinate = {y}")