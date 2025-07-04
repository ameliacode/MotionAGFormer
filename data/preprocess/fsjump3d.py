import os
import pickle
import sys

sys.path.append(os.getcwd())

from tqdm import tqdm

from data.reader.fsjump3d import FsJumpDataReader


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


def split_data(data_2d, data_3d, train_ratio=0.8):
    keys = list(data_2d.keys())
    n_samples = len(keys)
    n_train = int(n_samples * train_ratio)

    train_keys = keys[:n_train]
    test_keys = keys[n_train:]

    train_data_2d = {k: data_2d[k] for k in train_keys}
    train_data_3d = {k: data_3d[k] for k in train_keys}
    test_data_2d = {k: data_2d[k] for k in test_keys}
    test_data_3d = {k: data_3d[k] for k in test_keys}

    return train_data_2d, train_data_3d, test_data_2d, test_data_3d


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=243)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    n_frames = args.n_frames
    train_ratio = args.train_ratio

    reader = FsJumpDataReader(
        keypoints_path="./data/keypoints", data_split=None, n_frames=n_frames
    )

    data_2d, data_3d = reader.load_data("./data/keypoints", data_split=None)

    train_data_2d, train_data_3d, test_data_2d, test_data_3d = split_data(
        data_2d, data_3d, train_ratio
    )

    train_clips_2d, train_clips_3d, _ = reader.split_into_clips(
        train_data_2d, train_data_3d, n_frames, stride=81
    )

    test_clips_2d, test_clips_3d, _ = reader.split_into_clips(
        test_data_2d, test_data_3d, n_frames, stride=n_frames
    )

    assert len(train_clips_2d) == len(train_clips_3d)
    assert len(test_clips_2d) == len(test_clips_3d)

    root_path = f"./data/keypoints/H36M-{n_frames}"
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    save_clips("train", root_path, train_clips_2d, train_clips_3d)
    save_clips("test", root_path, test_clips_2d, test_clips_3d)


if __name__ == "__main__":
    main()
