"""Adapted from MotionBERT paper code (https://github.com/Walter0807/MotionBERT/blob/main/tools/convert_h36m.py)"""

import os
import pickle
import sys

sys.path.append(os.getcwd())

from tqdm import tqdm

from data.reader.ap3d import DataReaderH36M


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
    parser.add_argument("--n-frames", type=int, default=81)
    parser.add_argument("--dataset-path", type=str, default="./data/ap3d/")
    n_frames = parser.parse_args().n_frames
    dataset_path = parser.parse_args().dataset_path

    datareader = DataReaderH36M(
        n_frames=n_frames,
        sample_stride=1,
        data_stride_train=n_frames // 3,
        data_stride_test=n_frames,
        dt_root=dataset_path,
    )
    train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
    print(train_data.shape, test_data.shape)
    assert len(train_data) == len(train_labels)
    assert len(test_data) == len(test_labels)
    root_path = f"{dataset_path}/frame_{n_frames}"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    save_clips("train", root_path, train_data, train_labels)
    save_clips("test", root_path, test_data, test_labels)


if __name__ == "__main__":
    main()
