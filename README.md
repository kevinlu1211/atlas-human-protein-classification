# Atlas Human Protein Classification

### Introduction
This is the code for the Atlas HUman Protein Classification competition that received top 14% in the final leaderboard

### How to run?

1. Create a new Anaconda environment using the `environment.yml` file
2. Download the data from [Kaggle](https://www.kaggle.com/c/human-protein-atlas-image-classification/data)
3. Change the `DataPaths` class in the `src/data.py` file to where the files are downloaded. `ROOT_DATA_PATH` should
be the root directory where the Kaggle data is downloaded
4. Use the `notebook/image_merging.ipynb` notebook to create the merged numpy images, this is becuase the images that
are used have 4 color channels and it's faster to load them when they are in one file, instead of loading them individually.
5. Change the `LoadTrainingData` and `LoadTestingData` resources in `configs/resnet34_d_template.yml` to where the
combined image are stored
6. To run the training loop: `src/train_with_template.py -cfg src/configs/resnet34_d_template.yml`

