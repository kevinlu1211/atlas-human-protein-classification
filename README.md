# Atlas Human Protein Classification

### Introduction
This is the code for the Atlas Human Protein Classification competition that received top 14% in the final leaderboard

#### Setup

1. Clone this repo and `cd` to it
2. Download Docker from [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/) if you don't have it
3. Build the Docker image, this will take a while due to the dependencies: `docker build --tag human-protein-image-classification .`

#### Rough outline of problem/solution

In this competition we need to classify protein images from microscope scans, this main issue in this competition is huge class imbalance in the dataset.

Data: Official Kaggle Data, and external data from HPAv18
Data Preprocessing: Using phash to get rid of duplicate images
Data Augmentation: Random crop to (512, 512), and random flip, rotate90, random brightness, and elastic transforms
Loss function: Lovasz loss (experimentally I get better results than Macro Soft F1), and Focal Loss (to deal with class imbalance)
Model: Squeeze Excitation ResNeXt50 model
Prediction: 5 Crop TTA with max probs to deal with

Training a
#### If you want to do a full training run from scratch:

1. Get Kaggle API Keys for downloading the data
2. Run this command to set the Kaggle API Keys as environmental variables and also to bind a volume to the Docker image
(warning the full download and preprocessing is quite big ~400GB):
`docker container run -t --mount type=bind,source="$(pwd)"/data,target=/code/data -e KAGGLE_USERNAME=<YOUR_KAGGLE_USERNAME> -e KAGGLE_KEY=<YOUR_KAGGLE_KEY> human-protein-image-classification scripts/download_and_preprocess_data.sh`
3. To do the training, first install `nvidia-docker` to utilize the GPU [here](https://github.com/NVIDIA/nvidia-docker)
`mkdir results && docker build --tag human-protein-image-classification . && nvidia-docker container run -t --ipc=host --cpus=28 --mount type=bind,source="$(pwd)"/results,target=/code/results --mount type=bind,source="$(pwd)"/data,target=/code/data  human-protein-image-classification scripts/train.sh`

This will start the training process, and a folder will be created at the position `results/YYYYMMDD-HHMMSS` (this is in UTC time)
In the folder there will be subfolders, one for each fold. After training, the folder will contain:

- `config.yml` (this is the configuration file that defines the whole training process, so that the experiment is reproducible)
- `history.csv` (this loss, and metrics recorded after every training epoch)
- `model_checkpoints` (this folder contains the checkpoints during each epoch)
- `submission.csv`
- `submission_optimal_threshold.csv`
- `training_logs` (this folder contains two CSV files, one for training, one for validation, each row of the CSV file records the name, prediction, ground truth, and losses associated with one sample, look at `notebook/diagnosis.ipynb` to see how to visualize these training logs)


#### To have a play around with the pretrained model
`docker container run -t -p 8888:8888 human-protein-image-classification scripts/notebooks.sh`

Then open the `inference_node.ipynb` notebook
