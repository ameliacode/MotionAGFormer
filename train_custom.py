import argparse
import os

import numpy as np
import pkg_resources
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.const import (
    H36M_1_DF,
    H36M_2_DF,
    H36M_3_DF,
    H36M_JOINT_TO_LABEL,
    H36M_LOWER_BODY_JOINTS,
    H36M_UPPER_BODY_JOINTS,
)
from data.reader.custom import CustomDataReader
from data.reader.motion_dataset import MotionDataset3D
from loss.pose3d import acc_error as calculate_acc_err
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import (
    loss_angle,
    loss_angle_velocity,
    loss_limb_gt,
    loss_limb_var,
    loss_mpjpe,
    loss_velocity,
)
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import n_mpjpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from utils.data import Augmenter2D, flip_data
from utils.learning import AverageMeter, decay_lr_exponentially, load_model
from utils.tools import (
    count_param_numbers,
    create_directory_if_not_exists,
    get_config,
    print_args,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h36m/MotionAGFormer-base.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, metavar="PATH", help="checkpoint directory"
    )
    parser.add_argument(
        "--new-checkpoint",
        type=str,
        metavar="PATH",
        default="checkpoint",
        help="new checkpoint directory",
    )
    parser.add_argument("--checkpoint-file", type=str, help="checkpoint file name")
    parser.add_argument("-sd", "--seed", default=0, type=int, help="random seed")
    parser.add_argument("--num-cpus", default=16, type=int, help="Number of CPU cores")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-name", default=None, type=str)
    parser.add_argument("--wandb-run-id", default=None, type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    opts = parser.parse_args()
    return opts


def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = (
                    y[..., 2] - y[:, 0:1, 0:1, 2]
                )  # Place the depth of first frame root to be 0

        pred = model(x)  # (N, T, 17, 3)

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = (
            loss_3d_pos
            + args.lambda_scale * loss_3d_scale
            + args.lambda_3d_velocity * loss_3d_velocity
            + args.lambda_lv * loss_lv
            + args.lambda_lg * loss_lg
            + args.lambda_a * loss_a
            + args.lambda_av * loss_av
        )

        losses["3d_pose"].update(loss_3d_pos.item(), batch_size)
        losses["3d_scale"].update(loss_3d_scale.item(), batch_size)
        losses["3d_velocity"].update(loss_3d_velocity.item(), batch_size)
        losses["lv"].update(loss_lv.item(), batch_size)
        losses["lg"].update(loss_lg.item(), batch_size)
        losses["angle"].update(loss_a.item(), batch_size)
        losses["angle_velocity"].update(loss_av.item(), batch_size)
        losses["total"].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()


def evaluate(args, model, test_loader, device):
    print("[INFO] Evaluation")
    model.eval()
    mpjpe_all, p_mpjpe_all = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for x, y, indices in tqdm(test_loader):
            batch_size = x.shape[0]
            x = x.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1 = model(x)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(x)
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            predicted_3d_pos = predicted_3d_pos.detach().cpu().numpy()
            y = y.cpu().numpy()

            denormalized_predictions = []
            for i, prediction in enumerate(predicted_3d_pos):
                prediction = test_loader.dataset.denormalize(
                    prediction, indices[i].item(), is_3d=True
                )
                denormalized_predictions.append(prediction[None, ...])
            denormalized_predictions = np.concatenate(denormalized_predictions)

            # Root-relative Errors
            predicted_3d_pos = (
                denormalized_predictions - denormalized_predictions[..., 0:1, :]
            )
            y = y - y[..., 0:1, :]

            mpjpe = calculate_mpjpe(predicted_3d_pos, y)
            p_mpjpe = calculate_p_mpjpe(predicted_3d_pos, y)
            mpjpe_all.update(mpjpe, batch_size)
            p_mpjpe_all.update(p_mpjpe, batch_size)

    print(f"Protocol #1 error (MPJPE): {mpjpe_all.avg} mm")
    print(f"Protocol #2 error (P-MPJPE): {p_mpjpe_all.avg} mm")
    return mpjpe_all.avg, p_mpjpe_all.avg


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save(
        {
            "epoch": epoch + 1,
            "lr": lr,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "min_mpjpe": min_mpjpe,
            "wandb_id": wandb_id,
        },
        checkpoint_path,
    )


def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MotionDataset3D(args, args.subset_list, "train")
    test_dataset = MotionDataset3D(args, args.subset_list, "test")

    common_loader_params = {
        "batch_size": args.batch_size,
        "num_workers": opts.num_cpus - 1,
        "pin_memory": True,
        "prefetch_factor": (opts.num_cpus - 1) // 3,
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    datareader = CustomDataReader(keypoints_path="./keypoints", data_split="train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float("inf")  # Used for storing the best model
    wandb_id = (
        opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()
    )

    if opts.checkpoint:
        checkpoint_path = os.path.join(
            opts.checkpoint,
            opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr",
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
            model.load_state_dict(checkpoint["model"], strict=True)

            if opts.resume:
                lr = checkpoint["lr"]
                epoch_start = checkpoint["epoch"]
                optimizer.load_state_dict(checkpoint["optimizer"])
                min_mpjpe = checkpoint["min_mpjpe"]
                if "wandb_id" in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint["wandb_id"]
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(
                    id=wandb_id,
                    project="MotionMetaFormer",
                    resume="must",
                    settings=wandb.Settings(start_method="fork"),
                )
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(
                    id=wandb_id,
                    name=opts.wandb_name,
                    project="MotionMetaFormer",
                    settings=wandb.Settings(start_method="fork"),
                )
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {
                    d.project_name: d.version for d in pkg_resources.working_set
                }
                wandb.config.update({"installed_packages": installed_packages})

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, "latest_epoch.pth.tr")
    checkpoint_path_best = os.path.join(opts.new_checkpoint, "best_epoch.pth.tr")

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(args, model, test_loader, datareader, device)
            exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = [
            "3d_pose",
            "3d_scale",
            "2d_proj",
            "lg",
            "lv",
            "3d_velocity",
            "angle",
            "angle_velocity",
            "total",
        ]
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses)

        mpjpe, p_mpjpe, joints_error, acceleration_error = evaluate(
            args, model, test_loader, datareader, device
        )

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(
                checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id
            )
        save_checkpoint(
            checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id
        )

        joint_label_errors = {}
        for joint_idx in range(args.num_joints):
            joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = (
                joints_error[joint_idx]
            )
        if opts.use_wandb:
            wandb.log(
                {
                    "lr": lr,
                    "train/loss_3d_pose": losses["3d_pose"].avg,
                    "train/loss_3d_scale": losses["3d_scale"].avg,
                    "train/loss_3d_velocity": losses["3d_velocity"].avg,
                    "train/loss_2d_proj": losses["2d_proj"].avg,
                    "train/loss_lg": losses["lg"].avg,
                    "train/loss_lv": losses["lv"].avg,
                    "train/loss_angle": losses["angle"].avg,
                    "train/angle_velocity": losses["angle_velocity"].avg,
                    "train/total": losses["total"].avg,
                    "eval/mpjpe": mpjpe,
                    "eval/acceleration_error": acceleration_error,
                    "eval/min_mpjpe": min_mpjpe,
                    "eval/p-mpjpe": p_mpjpe,
                    "eval_additional/upper_body_error": np.mean(
                        joints_error[H36M_UPPER_BODY_JOINTS]
                    ),
                    "eval_additional/lower_body_error": np.mean(
                        joints_error[H36M_LOWER_BODY_JOINTS]
                    ),
                    "eval_additional/1_DF_error": np.mean(joints_error[H36M_1_DF]),
                    "eval_additional/2_DF_error": np.mean(joints_error[H36M_2_DF]),
                    "eval_additional/3_DF_error": np.mean(joints_error[H36M_3_DF]),
                    **joint_label_errors,
                },
                step=epoch + 1,
            )

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

    if opts.use_wandb:
        artifact = wandb.Artifact(f"model", type="model")
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)

    train(args, opts)


if __name__ == "__main__":
    main()
