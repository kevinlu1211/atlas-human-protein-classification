from miniutils.progress_bar import parallel_progbar, progbar
import numpy as np
import pandas as pd
import imagehash
from PIL import Image

from src.data import DataPaths
from src.image import open_numpy


def load_names_and_image_paths():
    image_paths = list(DataPaths.TRAIN_COMBINED_IMAGES.glob("*")) + list(
        DataPaths.TRAIN_COMBINED_IMAGES_HPAv18.glob("*"))
    image_paths.sort(key=lambda x: x.stem)
    df = pd.concat([pd.read_csv(DataPaths.TRAIN_LABELS), pd.read_csv(DataPaths.TRAIN_LABELS_HPAv18)])
    df = df.sort_values(by=['Id'])
    df['image_paths'] = image_paths
    assert np.all(df['Id'].values == [p.stem for p in df['image_paths'].values])
    names, image_paths = df['Id'], df['image_paths']
    return names, image_paths


names, image_paths = load_names_and_image_paths()
names_and_image_paths = zip(names, image_paths)


def calculate_phash(data):
    name, image_path = data
    image = open_numpy(image_path)
    phash = imagehash.phash(Image.fromarray(image.px))
    return name, phash.hash.flatten()


names_and_phashes = parallel_progbar(calculate_phash, names_and_image_paths)
names, phashes = zip(*names_and_phashes)

SIMILARITY_THRESHOLD = 0.75


def create_phash_similarity_df():
    phash_df_data = []
    for i, (name, phash) in progbar(enumerate(zip(names, phashes))):
        similarities = (len(phash) - (phash ^ phashes).sum(axis=1)) / len(phash)
        for similarity, name_of_image_compared in zip(similarities, names):
            if similarity > SIMILARITY_THRESHOLD and name != name_of_image_compared:
                phash_df_data.append({
                    "original_name": name,
                    "compared_image_name": name_of_image_compared,
                    "similarity": similarity
                })
    return pd.DataFrame(phash_df_data).sort_values(['similarity'], ascending=[False])


similarity_df = create_phash_similarity_df()
get_samples_with_similarity_above_and_equal = lambda df, similarity: df['similarity'].map(lambda x: x >= similarity)

filtered_similarity_df = similarity_df[get_samples_with_similarity_above_and_equal(similarity_df, 0.93750)]


def filter_duplicate_names():
    filtered_names = []
    for i, (name1, name2) in enumerate(zip(filtered_similarity_df['original_name'].values.tolist(),
                                           filtered_similarity_df['compared_image_name'].values.tolist())):
        both_samples_are_from_kaggle = "-" in name1 and "-" in name2
        sample_1_is_from_kaggle = "-" in name1
        sample_2_is_from_kaggle = "-" in name2
        if both_samples_are_from_kaggle:
            filtered_names.append(name1)
        elif sample_1_is_from_kaggle:
            filtered_names.append(name1)
        elif sample_2_is_from_kaggle:
            filtered_names.append(name2)
        else:
            filtered_names.append(name1)
    return filtered_names


duplicate_names = filter_duplicate_names()
df = pd.concat([pd.read_csv(DataPaths.TRAIN_LABELS), pd.read_csv(DataPaths.TRAIN_LABELS_HPAv18)])
df_without_dupes = df[df['Id'].map(lambda x: x not in duplicate_names)]
df_without_dupes.to_csv(DataPaths.TRAIN_LABELS_ALL_NO_DUPES, index=False)
