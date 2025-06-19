import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from tqdm import tqdm

from data.reader.custom import CustomDataReader


def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {"data_input": data_input, "data_label": data_label}
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as fp:
            pickle.dump(data_dict, fp)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=243)
    n_frames = parser.parse_args().n_frames

    # Create a temporary reader instance to access methods
    temp_reader = CustomDataReader(
        keypoints_path="../keypoints", data_split="train", n_frames=n_frames
    )

    # Load raw data for both splits
    train_data_2d, train_data_3d = temp_reader.load_data("../keypoints", "train")
    test_data_2d, test_data_3d = temp_reader.load_data("../keypoints", "test")

    # Use split_into_clips function for train data
    train_clips_2d, train_clips_3d, _ = temp_reader.split_into_clips(
        train_data_2d, train_data_3d, n_frames, stride=81
    )

    # Use split_into_clips function for test data
    test_clips_2d, test_clips_3d, _ = temp_reader.split_into_clips(
        test_data_2d, test_data_3d, n_frames, stride=n_frames
    )

    assert len(train_clips_2d) == len(train_clips_3d)
    assert len(test_clips_2d) == len(test_clips_3d)

    root_path = f"../motion3d/H36M-{n_frames}"
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    save_clips("train", root_path, train_clips_2d, train_clips_3d)
    save_clips("test", root_path, test_clips_2d, test_clips_3d)


if __name__ == "__main__":
    main()
