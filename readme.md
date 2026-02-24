# Coordinate Prediction 

## Problem

Using Deep Learning techniques, predict the coordinates (x,y) of a pixel which has a value of
255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0.
You can test the codes both with scripts & the notebook 'coordinate_predictor.ipynb' too.

## Installation

clone the repository & install dependencies
``` bash
git clone https://github.com/YuvrajBalagoni13/Assignment_Coordinate_Prediction.git
conda create -n env_name python
pip install -r requirements.txt
```

## Creating Dataset

So we need to create dataset with images size 50 x 50, which had only pixel value as 255 (white) & rest 0 (black).
Based on the image size the total number of possible images are 50 * 50 = 2500 images. So, I generated this dataset & split them in 80-10-10 for train-val-test. (2000 - 250 - 250 samples)

To create dataset, you can run:
``` bash
python create_dataset.py \
--image_size 50 \
--dataset_dir Coordinate_Dataset \
--train_split 0.8 \
--seed 42
```
this will create images & a target_coordinates.json file which will have coordinates corresponding to each image id.

## Model

As the problem was fairly simple, I focused more on keeping the model pretty simple.
I did 2 approaches :
1. Simple Convolution + MLP based regression.
2. Spatial Softmax based convolution approach. In this approach the model predicts a 2 dimensional joint probability distribution for the white pixel.

out of these 2, spatial softmax approach performed the best.

| Model Name      | parameters   |  Training Accuracy | Validation Accuracy | Test Accuracy |
| --------------- | ------------ |  ----------------- | ------------------- | ------------- |
| Simple conv     | 318,689      | 99.7%             | 97.9%               | 96.8%         |
| Spatial softmax | 4,929        | 100%              | 100%                | 99.6%         |


Loss Curves - 
<a href="https://github.com/YuvrajBalagoni13/Assignment_Coordinate_Prediction/blob/main/imgs/loss_curve.png">
  <img width="100%" alt="Integration" src="https://raw.githubusercontent.com/YuvrajBalagoni13/Assignment_Coordinate_Prediction/main/imgs/loss_curve.png" />
</a>

Accuracy Curves -
<a href="https://github.com/YuvrajBalagoni13/Assignment_Coordinate_Prediction/blob/main/imgs/accuracy_curve.png">
  <img width="100%" alt="Integration" src="https://raw.githubusercontent.com/YuvrajBalagoni13/Assignment_Coordinate_Prediction/main/imgs/accuracy_curve.png" />
</a>

## Training

if you want to train your own model then run this:

```bash
python train.py \
--name name_of_the_run \ 
--epochs 100 \
--batch_size 32 \
--learning_rate 0.001 \
--model_name name_of_the_model \ # spatial_softmax or simple_conv
--early_stop True
```

only use (spatial_softmax or simple_conv) as these will be in the models checkpoints name & will be used later for inference.

## Inference

in inference.py you can do both just an image inference & also do inference on entire test dataset & get performance.

for Testing:
```bash
python inference.py \
--test \
--checkpoint_path path_to_checkpoint \
--test_data_path path_to_test_dataset \
--target_coordinate_path path_to_coordinate_json 
```

for single inference:
```bash
python inference.py \
--inference \
--checkpoint_path path_to_checkpoint \
--infer_img_path path_to_image
```
