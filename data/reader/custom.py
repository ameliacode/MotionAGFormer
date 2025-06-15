import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from utils.data import flip_data
from utils.learning import AverageMeter


class CustomDataReader(Dataset):
    def __init__(
        self,
        keypoints_path,
        data_split,
        n_frames=243,
        stride=81,
        res_h=900,
        res_w=900,
        flip=False,
    ):
        self.data_split = data_split
        self.n_frames = n_frames
        self.res_h, self.res_w = res_h, res_w
        self.flip = flip
        self.stride = stride if data_split == "train" else n_frames

        # Load 2D and 3D keypoints data
        data_2d, data_3d = self.load_data(keypoints_path, data_split)

        # Split data into clips and store them along with camera information
        self.data_list_2d, self.data_list_3d, self.data_list_camera = (
            self.split_into_clips(data_2d, data_3d, n_frames, stride)
        )

        # Validate the lengths of 2D and 3D data lists
        assert len(self.data_list_2d) == len(self.data_list_3d)
        assert len(self.data_list_2d) == len(self.data_list_camera)

    def load_data(self, keypoints_path, data_split):
        data_list_2d, data_list_3d = {}, {}

        # Use keypoints_path directly instead of creating subdirectory
        split_path = (
            keypoints_path  # Changed from: os.path.join(keypoints_path, data_split)
        )

        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Data path does not exist: {split_path}")

        for filename in os.listdir(split_path):
            if filename.endswith("_2D.npy"):
                sequence_name = filename.replace("_2D.npy", "")
                keypoints_2d_file = os.path.join(split_path, filename)
                keypoints_3d_file = os.path.join(split_path, sequence_name + "_3D.npy")

                if not os.path.isfile(keypoints_2d_file) or not os.path.isfile(
                    keypoints_3d_file
                ):
                    print(f"Skipping missing file: {sequence_name}")
                    continue
                try:
                    keypoints_2d = np.load(keypoints_2d_file)
                    keypoints_3d = np.load(keypoints_3d_file)
                except Exception as e:
                    print(f"Error loading file {filename}: {e}")
                    continue

                if keypoints_2d.ndim != 3 or keypoints_3d.ndim != 3:
                    print(f"Invalid data dimensions for sequence: {sequence_name}")
                    continue

                data_list_2d[sequence_name] = keypoints_2d
                data_list_3d[sequence_name] = {
                    "keypoints": keypoints_3d,
                    "res_h": self.res_h,
                    "res_w": self.res_w,
                }

        if not data_list_2d or not data_list_3d:
            print("Warning: Data lists are empty after loading.")

        return data_list_2d, data_list_3d

    def split_into_clips(self, data_2d, data_3d, n_frames, stride):
        data_list_2d, data_list_3d, data_list_camera = [], [], []

        for sequence_name in data_2d:
            keypoints_2d = data_2d[sequence_name]
            keypoints_3d = data_3d[sequence_name]["keypoints"]
            res_h = data_3d[sequence_name]["res_h"]
            res_w = data_3d[sequence_name]["res_w"]

            if keypoints_2d.shape[0] != keypoints_3d.shape[0]:
                print(
                    f"Warning: Mismatch in sequence length for {sequence_name}. Skipping sequence."
                )
                continue

            # Normalize keypoints
            keypoints_2d = self.normalize(keypoints_2d, res_w, res_h)
            keypoints_3d = self.normalize(keypoints_3d, res_w, res_h, is_3d=True)

            # Partition into clips
            clips_2d, clips_3d = self.partition(
                keypoints_2d, keypoints_3d, n_frames, stride
            )

            data_list_2d.extend(clips_2d)
            data_list_3d.extend(clips_3d)
            data_list_camera.extend([(res_h, res_w)] * len(clips_2d))

        return data_list_2d, data_list_3d, data_list_camera

    def normalize(self, keypoints, w, h, is_3d=False):
        result = np.copy(keypoints)
        result[..., :2] = keypoints[..., :2] / w * 2 - [
            1,
            h / w,
        ]  # for width and height
        if is_3d:
            result[..., 2:] = keypoints[..., 2:] / w * 2  # for depth in 3D keypoints
        return result

    def denormalize(self, keypoints, idx, is_3d=False):
        h, w = self.data_list_camera[idx]
        result = np.copy(keypoints)
        result[..., :2] = (keypoints[..., :2] + np.array([1, h / w])) * w / 2
        if is_3d:
            result[..., 2:] = keypoints[..., 2:] * w / 2
        return result

    def partition(self, keypoints_2d, keypoints_3d, clip_length, stride):
        if self.data_split == "val":
            stride = clip_length

        clips_2d, clips_3d = [], []
        video_length = keypoints_2d.shape[0]
        if video_length <= clip_length:
            new_indices = self.resample(video_length, clip_length)
            clips_2d.append(keypoints_2d[new_indices])
            clips_3d.append(keypoints_3d[new_indices])
        else:
            start_frame = 0
            while (video_length - start_frame) >= clip_length:
                clips_2d.append(keypoints_2d[start_frame : start_frame + clip_length])
                clips_3d.append(keypoints_3d[start_frame : start_frame + clip_length])
                start_frame += stride
            new_indices = (
                self.resample(video_length - start_frame, clip_length) + start_frame
            )
            clips_2d.append(keypoints_2d[new_indices])
            clips_3d.append(keypoints_3d[new_indices])
        return clips_2d, clips_3d

    def __len__(self):
        return len(self.data_list_2d)

    def __getitem__(self, index):
        keypoints_2d = self.data_list_2d[index]
        keypoints_3d = self.data_list_3d[index]

        if self.flip and random.random() > 0.5:
            keypoints_2d = self.flip_data(keypoints_2d)
            keypoints_3d = self.flip_data(keypoints_3d)

        keypoints_2d = torch.from_numpy(keypoints_2d).float()
        keypoints_3d = torch.from_numpy(keypoints_3d).float()

        if self.data_split == "train":
            return keypoints_2d, keypoints_3d
        else:
            return keypoints_2d, keypoints_3d, index

    @staticmethod
    def resample(original_length, target_length):
        """
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68

        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result
