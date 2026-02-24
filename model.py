import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateRegressor(nn.Module):
    """CNN architecture for predicting (x, y) coordinates of a pixel."""

    def __init__(self, input_shape:int, mlp_units:int, output_shape:int, image_size: int = 50, dropout: float = 0.3):
        super(CoordinateRegressor, self).__init__()

        self.image_size = image_size

        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2) # Reduces image to 25x25
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2) # Reduces image to 12x12
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2) # Reduces image to 6x6
        
        size = image_size
        for _ in range(3):
            size = (size - 2) // 2 + 1   
        flatten_dim = 64 * size * size

        self.fc1 = nn.Linear(flatten_dim, mlp_units) # Fully Connected Block
        self.fc2 = nn.Linear(mlp_units, output_shape) 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)    # Flatten layer
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return torch.sigmoid(x)


# Conv model with Spatial Softmax 
class SpatialSoftmax(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        b, h, w = logits.shape
        logits_flatten = logits.view(b, -1)
        probs_flatten = torch.softmax(logits_flatten / self.temperature, dim=1)
        probs = probs_flatten.view(b, h, w)
        return probs
    
class CoordinateRegressorSpatialSoftmax(nn.Module):
    """CNN architecture for predicting (x, y) coordinates of a pixel with spatial softmax."""

    def __init__(self, input_shape:int, output_shape:int, image_size: int = 50):
        super(CoordinateRegressorSpatialSoftmax, self).__init__()

        self.image_size = image_size

        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.logits_conv = nn.Conv2d(in_channels = 32, out_channels=1, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax()

        x_coords = torch.arange(image_size, dtype=torch.float32)
        y_coords = torch.arange(image_size, dtype=torch.float32)

        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        self.register_buffer('grid_x', grid_x.clone())
        self.register_buffer('grid_y', grid_y.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        logits = self.logits_conv(x)
        logits = logits.squeeze(1)
        probs = self.spatial_softmax(logits)
        x_pred = torch.einsum("bhw,hw->b", probs, self.grid_x)
        y_pred = torch.einsum("bhw,hw->b", probs, self.grid_y)
        output = torch.stack([x_pred, y_pred], dim=1) / 50.0
        return output


# performed poorly - not able to grasp the spatial structure.
class NeuralNetRegressor(nn.Module):
    def __init__(self, input_shape: int, mlp_units: int, output_shape: int, dropout_rate: float = 0.3) -> None:
        super(NeuralNetRegressor, self).__init__()
        self.fc1 = nn.Linear(input_shape * 50 * 50, mlp_units * 2)
        self.fc2 = nn.Linear(mlp_units * 2, mlp_units)
        self.fc3 = nn.Linear(mlp_units, output_shape)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return torch.sigmoid(x)
    