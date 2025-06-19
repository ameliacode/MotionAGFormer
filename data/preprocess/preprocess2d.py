import copy
import glob
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from demo.lib.preprocess import coco_coco, h36m_coco_format
from dwpose.scripts.dwpose import DWposeDetector
from dwpose.scripts.tool import read_frames

warnings.filterwarnings("ignore")
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def load_json(json_file: str) -> dict:
    with open(json_file, "r") as f:
        return json.load(f)


ranges = load_json("./ranges.json")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = DWposeDetector(
    det_config="D:\\github\\skating-ai\\v3\\pose\\dwpose\\config\\yolox_l_8xb8-300e_coco.py",
    pose_config="D:\\github\\skating-ai\\v3\\pose\\dwpose\\config\\dwpose-l_384x288.py",
    keypoints_only=True,
).to(device)
video_paths = glob.glob("D:\\github\\FS-Jump3D\\data\\**\\*.mp4", recursive=True)
output_dir = "D:\\github\\MotionAGFormer\\data\\keypoints"
os.makedirs(output_dir, exist_ok=True)
lock = Lock()


def estimate2d(video_path, detector):
    path_parts = Path(video_path).parts
    skater = path_parts[4].lower()
    jump = Path(video_path).stem.lower()

    range_key = f"{skater}-{jump}"
    start = int(ranges[range_key]["start"])
    end = int(ranges[range_key]["end"]) + 1

    frames = read_frames(video_path)[start:end]

    num_frames = len(frames)
    duration = int(ranges[range_key]["duration"])
    assert num_frames == duration

    kpts2d = []
    score2d = []
    kpts2d.reserve(num_frames) if hasattr(kpts2d, "reserve") else None
    score2d.reserve(num_frames) if hasattr(score2d, "reserve") else None

    person_idx = 0

    for frame in frames:
        pose = detector(frame)
        candidate = pose["bodies"]["candidate"]
        subset = pose["bodies"]["subset"]
        num_person = subset.shape[0]

        if num_person == 0:
            kpts2d.append(np.zeros((17, 2)))
            score2d.append(np.zeros(17))
            continue

        num_joints = subset.shape[1]
        keypoint = candidate.reshape(num_person, num_joints, 2)

        if num_person == 1:
            kpt, score = coco_coco(keypoint[0], subset[0])
            kpts2d.append(kpt)
            score2d.append(score)
        else:
            areas = []
            for i in range(num_person):
                valid_kpts = keypoint[i][keypoint[i][:, 0] > 0]
                if len(valid_kpts) > 0:
                    x_min, y_min = valid_kpts.min(axis=0)
                    x_max, y_max = valid_kpts.max(axis=0)
                    area = (x_max - x_min) * (y_max - y_min)
                    areas.append(area)
                else:
                    areas.append(0)

            person_idx = np.argmax(areas)
            kpt, score = coco_coco(keypoint[person_idx], subset[person_idx])
            kpts2d.append(kpt)
            score2d.append(score)

    kpts2d = np.expand_dims(np.array(kpts2d), axis=1)
    score2d = np.expand_dims(np.array(score2d), axis=1)

    kpts2d, score2d, _ = h36m_coco_format(kpts2d, score2d)
    kpts2d = kpts2d.squeeze(axis=1)
    score2d = score2d.squeeze(axis=1)

    keypoints = np.concatenate([kpts2d, score2d[..., np.newaxis]], axis=-1)
    assert keypoints.shape[0] == duration

    return keypoints


def process_video(video_path):
    path_parts = Path(video_path).parts
    skater = path_parts[-3].lower()
    camera = path_parts[-2].lower()
    filename = Path(video_path).stem.lower()
    output_name = f"{skater}_{camera}_{filename}_2D.npy"
    output_path = os.path.join(output_dir, output_name)

    with lock:
        if os.path.exists(output_path):
            return

    keypoints = estimate2d(video_path, detector)

    with lock:
        np.save(output_path, keypoints)


with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_video, path) for path in video_paths]
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()
