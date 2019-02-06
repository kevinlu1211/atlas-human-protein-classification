import subprocess
import sys
print(sys.path)
from src.data import DataPaths


DataPaths.ROOT_DATA_PATH.mkdir(parents=True, exist_ok=True)
print("Downloading training labels")
subprocess.check_call(["kaggle", "competitions", "download", "-f",
                       "train.csv", "-p", str(DataPaths.ROOT_DATA_PATH),
                       "human-protein-atlas-image-classification"])

DataPaths.TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
print("Downloading Kaggle training data")
subprocess.check_call(["kaggle", "competitions", "download", "-f",
                       "train.zip", "-p", str(DataPaths.ROOT_DATA_PATH),
                       "human-protein-atlas-image-classification"])
print("Unzipping training data")
subprocess.check_call(["unzip", str(DataPaths.ROOT_DATA_PATH / "train.zip"), "-d", str(DataPaths.TRAIN_IMAGES)])
subprocess.check_call(["rm", str(DataPaths.ROOT_DATA_PATH / "train.zip")])

DataPaths.TEST_IMAGES.mkdir(parents=True, exist_ok=True)
print("Downloading Kaggle testing data")
subprocess.check_call(["kaggle", "competitions", "download", "-f",
                       "test.zip", "-p", str(DataPaths.ROOT_DATA_PATH),
                       "human-protein-atlas-image-classification"])
print("Unzipping training data")
subprocess.check_call(["unzip", str(DataPaths.ROOT_DATA_PATH / "test.zip"), "-d", str(DataPaths.TEST_IMAGES)])
subprocess.check_call(["rm", str(DataPaths.ROOT_DATA_PATH / "test.zip")])

