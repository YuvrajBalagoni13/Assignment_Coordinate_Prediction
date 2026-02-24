import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
import wandb

import model 
import dataloaders

DATASET_DIR = "Coordinate_Dataset"

def log_metrics(epoch, loss, accuracy, val_loss, val_accuracy):
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })


def save_checkpoint(epoch, run_name, model, loss, description):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', f'{run_name}_{epoch + 1}.pt')
    torch.save({
        'epoch': epoch,
        'description': description,
        'model': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def get_accuracy(
        ground_truth: np.ndarray,
        predictions: np.ndarray
) -> int:
    """
    This function gets the exact pixel accuracy for the outputs
    """
    ground_truth = (ground_truth * 50).round().int()
    predictions = (predictions * 50).round().int()
    accuracy = (ground_truth == predictions).all(dim=1)
    batch_accuracy = accuracy.float().mean()
    return batch_accuracy

def get_lenient_accuracy(
        ground_truth: torch.Tensor,
        predictions: torch.Tensor,
        threshold: float = 2.0 
) -> float:
    """
    This adds a leniency of threshold pixels for the accuracy
    """
    gt_scaled = ground_truth * 50
    pred_scaled = predictions * 50
    
    distances = torch.sqrt(torch.sum((pred_scaled - gt_scaled) ** 2, dim=1))
    
    correct = (distances <= threshold).float().mean()
    return correct

def train():
    wandb.init(project="CoordinatePrediction", name="base_model_again", config={
        "name": "base_model_again",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "description": "deepening the base model",
    })
    configs = wandb.config
    run_name = configs.name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loaders = dataloaders.get_dataloaders(
        dataset_dir = DATASET_DIR,
        batch_size = configs.batch_size,
        shuffle = True,
        device = device
    )
    train_dataloader = data_loaders['train']
    val_dataloader = data_loaders['val']
    criterion = nn.MSELoss()

    coordinateregressor = model.CoordinateRegressor(input_shape=1, mlp_units=128, output_shape=2).to(device)

    model_optim = torch.optim.AdamW(
        params = coordinateregressor.parameters(),
        lr = configs.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=configs.epochs)

    prev_min_train_loss = 1e6            # train_loss for the previous minimum loss 
    early_stoping_count = 10             # early stoping count for epochs with less than min loss
    count = 0                            
    earlystoping_threshold = 1e-6        # Threshold for early stoping counts to increase

    for epoch in tqdm(range(configs.epochs)):
        print (f"epoch {epoch + 1}")

        ##################### Training Loop #####################

        mean_loss, mean_accuracy = 0.0, 0.0

        for samples in tqdm(train_dataloader):
            samples['image'], samples['coordinates'] = samples['image'].to(device), samples['coordinates'].to(device)
            coordinateregressor.train()
            output = coordinateregressor(samples['image'])
            ground_truth = samples['coordinates'].float()

            accuracy = get_accuracy(ground_truth, output)
            mean_accuracy += accuracy.item() / len(train_dataloader)

            model_optim.zero_grad()
            loss = criterion(output, ground_truth)
            loss.backward()
            model_optim.step()
            mean_loss += loss.item() / len(train_dataloader)
        scheduler.step()

        ##################### Validation Loop #####################
        mean_val_loss, mean_val_accuracy = 0.0, 0.0
        coordinateregressor.eval()
        with torch.inference_mode():
            for samples in tqdm(val_dataloader):
                output = coordinateregressor(samples['image'])
                ground_truth = samples['coordinates'].float()

                accuracy = get_accuracy(ground_truth, output)
                mean_val_accuracy += accuracy.item() / len(val_dataloader)

                loss = criterion(output, ground_truth)
                mean_val_loss += loss.item() / len(val_dataloader)     

        ##################### Handling Early Stoping #####################
        if (prev_min_train_loss - mean_val_loss) <= earlystoping_threshold:
            count += 1
            if count >= early_stoping_count:
                print(f"Stoping early epoch {epoch + 1} due to no major change in loss | train loss {mean_loss:.5f} train accuracy {mean_accuracy:.5f} val loss {mean_val_loss:.5f} val accuracy {mean_val_accuracy:.5f}")
                save_checkpoint(epoch, run_name, coordinateregressor, loss, configs.description)
                break
        else:
            count = 0  
            prev_min_train_loss = min(prev_min_train_loss, mean_val_loss)

        ##################### logging & saving checkpoints #####################

        if epoch % 10 == 0:
            log_metrics(epoch, mean_loss, mean_accuracy, mean_val_loss, mean_val_accuracy)
            print(f"Epoch {epoch + 1} | mse loss: {mean_loss:.5f} | accuracy: {mean_accuracy:.5f}")
            print("#" * 50)
            print(f"validation = Epoch {epoch + 1} | mse loss: {mean_val_loss:.5f} | accuracy: {mean_val_accuracy:.5f}")
            save_checkpoint(epoch, run_name, coordinateregressor, loss, configs.description)

if __name__ == "__main__":
    train()

"""
python train.py
"""