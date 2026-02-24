import torch
import model

def print_model_parameters(model):
    # Counts all parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Counts only the parameters being updated by the optimizer
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

coordinateregressor = model.NeuralNetRegressor(input_shape=1, mlp_units=128, output_shape=1)
print_model_parameters(coordinateregressor)