import json
import os

import numpy as np

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
coco_to_coco = {
    0: 0,
    16: 3,
    14: 1,
    15: 2,
    17: 4,
    2: 5,
    3: 7,
    4: 9,
    5: 6,
    6: 8,
    7: 10,
    8: 11,
    9: 13,
    10: 15,
    11: 12,
    12: 14,
    13: 16,
}

spple_keypoints = [10, 8, 0, 7]


def coco_coco(keypoints, scores):
    keypoints_coco = np.zeros((17, keypoints.shape[1]), dtype=np.float32)
    scores_coco = np.zeros(17, dtype=np.float32)

    for old_idx, new_idx in coco_to_coco.items():
        keypoints_coco[new_idx] = keypoints[old_idx]
        scores_coco[new_idx] = scores[new_idx]

    return keypoints_coco, scores_coco


def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = (
        np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    )
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 3, :] = np.mean(
        keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32
    )

    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    keypoints_h36m[:, 9, :] -= (
        keypoints_h36m[:, 9, :]
        - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    ) / 4
    keypoints_h36m[:, 7, 0] += 2 * (
        keypoints_h36m[:, 7, 0]
        - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32)
    )
    keypoints_h36m[:, 8, 1] -= (
        (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])
        * 2
        / 3
    )

    # half body: the joint of ankle and knee equal to hip
    # keypoints_h36m[:, [2, 3]] = keypoints_h36m[:, [1, 1]]
    # keypoints_h36m[:, [5, 6]] = keypoints_h36m[:, [4, 4]]

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]

    return keypoints_h36m, valid_frames


def h36m_coco_format(keypoints, scores):
    assert len(keypoints.shape) == 4 and len(scores.shape) == 3

    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.0:
            kpts, valid_frame = coco_h36m(kpts)
            h36m_kpts.append(kpts)
            valid_frames.append(valid_frame)

            new_score[:, h36m_coco_order] = score[:, coco_order]
            new_score[:, 0] = np.mean(score[:, [11, 12]], axis=1, dtype=np.float32)
            new_score[:, 8] = np.mean(score[:, [5, 6]], axis=1, dtype=np.float32)
            new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
            new_score[:, 10] = np.mean(score[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

            h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts, h36m_scores, valid_frames


def revise_kpts(h36m_kpts, h36m_scores, valid_frames):

    new_h36m_kpts = np.zeros_like(h36m_kpts)
    for index, frames in enumerate(valid_frames):
        kpts = h36m_kpts[index, frames]
        score = h36m_scores[index, frames]

        index_frame = np.where(np.sum(score < 0.3, axis=1) > 0)[0]

        for frame in index_frame:
            less_threshold_joints = np.where(score[frame] < 0.3)[0]

            intersect = [i for i in [2, 3, 5, 6] if i in less_threshold_joints]

            if [2, 3, 5, 6] == intersect:
                kpts[frame, [2, 3, 5, 6]] = kpts[frame, [1, 1, 4, 4]]
            elif [2, 3, 6] == intersect:
                kpts[frame, [2, 3, 6]] = kpts[frame, [1, 1, 5]]
            elif [3, 5, 6] == intersect:
                kpts[frame, [3, 5, 6]] = kpts[frame, [2, 4, 4]]
            elif [3, 6] == intersect:
                kpts[frame, [3, 6]] = kpts[frame, [2, 5]]
            elif [3] == intersect:
                kpts[frame, 3] = kpts[frame, 2]
            elif [6] == intersect:
                kpts[frame, 6] = kpts[frame, 5]
            else:
                continue

        new_h36m_kpts[index, frames] = kpts

    return new_h36m_kpts
