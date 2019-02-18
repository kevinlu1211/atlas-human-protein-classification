# Atlas Human Protein Classification

### Introduction
This is the code for the Atlas Human Protein Classification competition creates a top 6% submission in the final leaderboard. The submission `.csv` file is in the `results/submission_optimal_threshold.csv`

#### Setup

1. Clone this repo and `cd` to it
2. Download Docker from [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/) if you don't have it
3. Build the Docker image, this will take a while due to the dependencies: `docker build --tag human-protein-image-classification .`

#### Rough outline of problem/solution

In this competition we need to classify protein images from microscope scans. The main issue is the large imbalance in the classes, below I will roughly describe the solution outline:

Data: Official Kaggle Data, and external data from HPAv18
Data Preprocessing: Use phash to get rid of duplicate images, as microscope scans are made up of a stack of slices, and if the slices are close they will look very similar. This will give a big disparity between the validation set performance, and real world performance
Validation split: Since this is multi-class classification, we have to do iterative splitting to make sure that our validation set is representative of the test set
Data Augmentation: Random crop to (512, 512), and random flip, rotate90, random brightness, and elastic transforms, the random crops were very helpful in decreasing the network overfitting
Loss function: Lovasz loss (experimentally I get better results than Macro Soft F1), and Focal Loss (to deal with class imbalance)
Model: Squeeze Excitation ResNeXt50 model
Training: Used a one schedule cycle with LR of 4e-8, this was found experimentally via `lr_find`, an example of this can be seen in `notebook/learning_rate_finder.ipynb`
Prediction: 5 Crop TTA (top left, top right, bottom left, bottom right, and center) with max probs to deal with the fact that some proteins only appear once in the image. By taking the maximum probability, instead of the average we will be able to capture this information.
Postprocessing: Due to the imbalance of classes, using a threshold of 0.5 to determine true/false would not be efficient. Instead, these thresholds are determined by optimizing for the best thresholds on the validation set.

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
- `submission.csv` results for using thresholds of 0.5
- `submission_optimal_threshold.csv` thresholds optimized on the
- `training_logs` (this folder contains two CSV files, one for training, one for validation, each row of the CSV file records the name, prediction, ground truth, and losses associated with one sample, look at `notebook/diagnosis.ipynb` to see how to visualize these training logs)


#### To have a play around with the pretrained model
`docker container run -t -p 8888:8888 human-protein-image-classification scripts/notebooks.sh`

Then open the `inference_node.ipynb` notebook
