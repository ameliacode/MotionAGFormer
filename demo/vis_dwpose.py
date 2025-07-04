import argparse
import copy
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from demo.lib.preprocess import coco_coco, h36m_coco_format
from demo.lib.utils import camera_to_world, normalize_screen_coordinates
from dwpose.scripts.dwpose import DWposeDetector
from dwpose.scripts.tool import read_frames
from model.MotionAGFormer import MotionAGFormer

plt.switch_backend("agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def show2Dpose(kps, img):
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16],
    ]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(
            img,
            (start[0], start[1]),
            (end[0], end[1]),
            lcolor if LR[j] else rcolor,
            thickness,
        )
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15.0, azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect("auto")  # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params("x", labelbottom=False)
    ax.tick_params("y", labelleft=False)
    ax.tick_params("z", labelleft=False)


def img2video(video_path, output_dir, video_name):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + "pose/", "*.png")))
    if not names:
        print("No pose images found for video creation")
        return

    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + ".mp4", fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()
    cap.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis("off")
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=27, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    downsample = None

    if n_frames <= 27:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 27):
            keypoints_clip = keypoints[:, start_idx : start_idx + 27, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 27:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)

    return clips, downsample


def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (
        new_keypoints[..., 0, :] + new_keypoints[..., 8, :]
    ) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(
    data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]
):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[
        ..., right_joints + left_joints, :
    ]  # Change orders
    return flipped_data


def process_2d_frame(frame, width, height, detector):
    pose = detector(frame)
    candidate = pose["bodies"]["candidate"]
    subset = pose["bodies"]["subset"]
    num_person = subset.shape[0]

    if num_person == 0:
        return np.zeros((17, 2)), np.zeros(17)

    num_joints = subset.shape[1]
    keypoint = candidate.reshape(num_person, num_joints, 2)

    if num_person == 1:
        kpt, score = coco_coco(keypoint[0], subset[0])
        kpt[:, 0] *= width
        kpt[:, 1] *= height
        return kpt, score
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
        kpt[:, 0] *= width
        kpt[:, 1] *= height
        return kpt, score


def get_pose2D(video_path, output_dir, detector):
    frames = read_frames(video_path)
    print("\nGenerating 2D pose...")

    kpts2d = []
    score2d = []
    width, height = frames[0].size

    # Process frames sequentially for better stability
    for frame in tqdm(frames, desc="Processing frames"):
        kpt, score = process_2d_frame(frame, width, height, detector)
        kpts2d.append(kpt)
        score2d.append(score)

    keypoints = np.expand_dims(np.array(kpts2d), axis=1)
    scores = np.expand_dims(np.array(score2d), axis=1)

    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    keypoints = keypoints.squeeze(axis=1)
    scores = scores.squeeze(axis=1)

    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)
    keypoints = np.expand_dims(keypoints, axis=0)

    output_dir_2d = output_dir + "input_2D/"
    os.makedirs(output_dir_2d, exist_ok=True)

    output_npz = output_dir_2d + "keypoints.npz"
    np.savez_compressed(output_npz, reconstruction=keypoints)

    return keypoints, scores


def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = (
        12,
        3,
        64,
        512,
        3,
    )
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = (
        True,
        0.00001,
        True,
    )
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = (
        True,
        2,
        1,
    )
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 27
    args = vars(args)

    ## Reload model
    model = nn.DataParallel(MotionAGFormer(**args)).cuda()
    model_path = "./checkpoint/h36m_ap3d/best_epoch.pth.tr"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    pre_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(pre_dict["model"], strict=False)
    model.eval()

    ## Input keypoints
    keypoints_path = output_dir + "input_2D/keypoints.npz"
    if not os.path.exists(keypoints_path):
        raise FileNotFoundError(f"2D keypoints not found at: {keypoints_path}")

    keypoints = np.load(keypoints_path, allow_pickle=True)["reconstruction"]

    clips, downsample = turn_into_clips(keypoints)
    cap = cv2.VideoCapture(video_path)
    ret, img = cap.read()
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")

    img_size = img.shape
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Processing {len(clips)} clips...")

    # Process clips sequentially to avoid memory issues
    for idx, clip in enumerate(tqdm(clips, desc="Processing 3D pose")):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype("float32")).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype("float32")).cuda()

        generate_3d_pose(
            model, input_2D, input_2D_aug, downsample, idx, output_dir, len(clips)
        )

    print("Generating 3D pose successful!")


def generate_3d_pose(
    model, input_2D, input_2D_aug, downsample, idx, output_dir, total_clips
):
    with torch.no_grad():  # Add no_grad for memory efficiency
        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == total_clips - 1 and downsample is not None:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

        for j, post_out in enumerate(post_out_all):
            rot = [
                0.1407056450843811,
                -0.1500701755285263,
                -0.755240797996521,
                0.6223280429840088,
            ]
            rot = np.array(rot, dtype="float32")
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            if max_value > 0:  # Avoid division by zero
                post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection="3d")
            show3Dpose(post_out, ax)

            output_dir_3D = output_dir + "pose3D/"
            os.makedirs(output_dir_3D, exist_ok=True)
            plt.savefig(
                output_dir_3D + str(("%04d" % (idx * 27 + j))) + "_3D.png",
                dpi=200,
                format="png",
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, default="sample_video.mp4", help="input video"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU device")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = "./demo/video/" + args.video

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_name = video_path.split("/")[-1].split(".")[0]
    output_dir = "./demo/output/" + video_name + "/"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    detector = DWposeDetector(
        det_config="./dwpose/config/yolox_l_8xb8-300e_coco.py",
        pose_config="./dwpose/config/dwpose-l_384x288.py",
        keypoints_only=True,
    ).to(device)

    try:
        keypoints_2d, scores_2d = get_pose2D(video_path, output_dir, detector)
        get_pose3D(video_path, output_dir)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
