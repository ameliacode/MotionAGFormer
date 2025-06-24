# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import numpy as np


def camera_to_image_frame(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = weak_project(
        pose3d.copy(), camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    )
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame


def infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 1000.0
    br_joint = root_joint.copy()
    br_joint[:2] += 1000.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = weak_project(
        tl_joint, camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    ).flatten()

    br2d = weak_project(
        br_joint, camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    ).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d
