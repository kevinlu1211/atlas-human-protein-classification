# Atlas Human Protein Classification

### Introduction
This is the code for the Atlas Human Protein Classification competition that received top 14% in the final leaderboard

#### Setup

1. Clone this repo and `cd` to it
2. Download Docker from [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/) if you don't have it
3. Build the Docker image, this will take a while due to the dependencies: `docker build --tag human-protein-image-classification .`

#### If you want to do a full training run from scratch:

1. Get Kaggle API Keys for downloading the data
2. Run this command to set the Kaggle API Keys as environmental variables and also to bind a volume to the Docker image
(warning the full download and preprocessing is quite big ~300GB+):
`docker container run -t --mount type=bind,source="$(pwd)"/data,target=/code/data -e KAGGLE_USERNAME=<YOUR_KAGGLE_USERNAME> -e KAGGLE_KEY=<YOUR_KAGGLE_KEY> human-protein-image-classification scripts/download_and_preprocess_data.sh`
3. To do the training:
`docker container run -t --mount type=bind,source="$(pwd)"/data,target=/code/data human-protein-image-classification scripts/train.sh`

#### To have a play around with the pretrained model
1. To have a look at the notebooks: `docker container run -t -p 8888:8888 human-protein-image-classification scripts/notebooks.sh`
