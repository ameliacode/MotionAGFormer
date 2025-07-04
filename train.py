import argparse
import os

import numpy as np
import pkg_resources
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

from data.reader.fsjump3d import FsJumpDataReader
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
        default="configs/h36m/MotionAGFormer-xsmall.yaml",
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
    parser.add_argument("--root_rel", default=False)
    opts = parser.parse_args()
    return opts


def monitor_gpu_memory():
    """Monitor and print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    scaler = torch.amp.GradScaler("cuda")

    # Use gradient accumulation if batch size is very small
    accumulation_steps = 1
    if args.batch_size <= 8:
        accumulation_steps = max(
            1, 32 // args.batch_size
        )  # Target effective batch size of 32
        print(f"[INFO] Using gradient accumulation with {accumulation_steps} steps")

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        batch_size = x.shape[0]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = (
                    y[..., 2] - y[:, 0:1, 0:1, 2]
                )  # Place the depth of first frame root to be 0

        # Use mixed precision training to save memory
        with torch.amp.autocast("cuda"):
            pred = model(x)  # (N, T, 17, 3)

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

            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss_total = loss_total / accumulation_steps

        # Update losses (scale back for logging if using accumulation)
        loss_scale = accumulation_steps if accumulation_steps > 1 else 1
        losses["3d_pose"].update((loss_3d_pos * loss_scale).item(), batch_size)
        losses["3d_scale"].update((loss_3d_scale * loss_scale).item(), batch_size)
        losses["3d_velocity"].update((loss_3d_velocity * loss_scale).item(), batch_size)
        losses["lv"].update((loss_lv * loss_scale).item(), batch_size)
        losses["lg"].update((loss_lg * loss_scale).item(), batch_size)
        losses["angle"].update((loss_a * loss_scale).item(), batch_size)
        losses["angle_velocity"].update((loss_av * loss_scale).item(), batch_size)
        losses["total"].update((loss_total * loss_scale).item(), batch_size)

        # Backward pass
        scaler.scale(loss_total).backward()

        # Update weights every accumulation_steps batches or clear gradients
        if accumulation_steps > 1:
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Clear cache periodically to prevent fragmentation
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        # Clean up intermediate variables
        del pred, loss_3d_pos, loss_3d_scale, loss_3d_velocity
        del loss_lv, loss_lg, loss_a, loss_av, loss_total


@torch.no_grad()
def evaluate(args, model, test_loader, test_dataset, device):
    print("[INFO] Evaluation")
    model.eval()
    mpjpe_all, p_mpjpe_all = AverageMeter(), AverageMeter()

    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        x, y, indices = batch_data

        batch_size = x.shape[0]
        x = x.to(device, non_blocking=True)

        if args.flip:
            batch_input_flip = flip_data(x)
            predicted_3d_pos_1 = model(x)
            predicted_3d_pos_flip = model(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            del (
                batch_input_flip,
                predicted_3d_pos_1,
                predicted_3d_pos_flip,
                predicted_3d_pos_2,
            )
        else:
            predicted_3d_pos = model(x)

        if args.root_rel:
            predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
        else:
            y[:, 0, 0, 2] = 0

        predicted_3d_pos = predicted_3d_pos.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        denormalized_predictions = []
        denormalized_ground_truth = []

        for i in range(predicted_3d_pos.shape[0]):
            if isinstance(indices, torch.Tensor):
                sample_idx = (
                    indices[i].item()
                    if i < len(indices)
                    else batch_idx * test_loader.batch_size + i
                )
            else:
                sample_idx = batch_idx * test_loader.batch_size + i

            pred_denorm = test_loader.dataset.denormalize(
                predicted_3d_pos[i], sample_idx, is_3d=True
            )
            denormalized_predictions.append(pred_denorm[None, ...])

            gt_denorm = test_loader.dataset.denormalize(y[i], sample_idx, is_3d=True)
            denormalized_ground_truth.append(gt_denorm[None, ...])

        denormalized_predictions = np.concatenate(denormalized_predictions)
        denormalized_ground_truth = np.concatenate(denormalized_ground_truth)

        predicted_3d_pos = (
            denormalized_predictions - denormalized_predictions[..., 0:1, :]
        )
        y = denormalized_ground_truth - denormalized_ground_truth[..., 0:1, :]

        batch_size, n_frames, n_joints, _ = predicted_3d_pos.shape

        predicted_reshaped = predicted_3d_pos.reshape(-1, n_joints, 3)
        y_reshaped = y.reshape(-1, n_joints, 3)

        mpjpe = calculate_mpjpe(predicted_reshaped, y_reshaped)
        p_mpjpe = calculate_p_mpjpe(predicted_reshaped, y_reshaped)

        mpjpe_scalar = np.mean(mpjpe) if isinstance(mpjpe, np.ndarray) else mpjpe
        p_mpjpe_scalar = (
            np.mean(p_mpjpe) if isinstance(p_mpjpe, np.ndarray) else p_mpjpe
        )

        total_poses = batch_size * n_frames
        mpjpe_all.update(mpjpe_scalar, total_poses)
        p_mpjpe_all.update(p_mpjpe_scalar, total_poses)

        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()

    print(f"Protocol #1 error (MPJPE): {mpjpe_all.avg} mm")
    print(f"Protocol #2 error (P-MPJPE): {p_mpjpe_all.avg} mm")
    return mpjpe_all.avg, p_mpjpe_all.avg


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    if hasattr(model, "module"):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(
        {
            "epoch": epoch + 1,
            "lr": lr,
            "optimizer": optimizer.state_dict(),
            "model": model_state_dict,
            "min_mpjpe": min_mpjpe,
            "wandb_id": wandb_id,
        },
        checkpoint_path,
    )


def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = FsJumpDataReader(
        keypoints_path=os.path.join(
            args.keypoints_path,
            args.subset_list[0],
        ),
        data_split="train",
        n_frames=args.n_frames,
        stride=81,
        res_h=1080,
        res_w=1920,
    )

    test_dataset = FsJumpDataReader(
        keypoints_path=os.path.join(
            args.keypoints_path,
            args.subset_list[0],
        ),
        data_split="test",
        n_frames=args.n_frames,
        stride=81,
        res_h=1080,
        res_w=1920,
    )

    common_loader_params = {
        "batch_size": args.batch_size,
        "num_workers": min(opts.num_cpus - 1, 4),
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args)

    # Only use DataParallel if you have multiple GPUs and enough memory
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model.to(device)

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    monitor_gpu_memory()

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

            # Handle loading state dict for DataParallel models
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["model"], strict=True)
            else:
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

    ## Train ##
    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(args, model, test_loader, test_dataset, device)
            exit()

        print(f"[INFO] epoch {epoch}")
        monitor_gpu_memory()

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

        try:
            train_one_epoch(args, model, train_loader, optimizer, device, losses)

            mpjpe, p_mpjpe = evaluate(args, model, test_loader, test_dataset, device)

            if mpjpe < min_mpjpe:
                min_mpjpe = mpjpe
                save_checkpoint(
                    checkpoint_path_best,
                    epoch,
                    lr,
                    optimizer,
                    model,
                    min_mpjpe,
                    wandb_id,
                )
            save_checkpoint(
                checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id
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
                        "eval/min_mpjpe": min_mpjpe,
                        "eval/p-mpjpe": p_mpjpe,
                    },
                    step=epoch + 1,
                )

            lr = decay_lr_exponentially(lr, lr_decay, optimizer)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[ERROR] CUDA out of memory at epoch {epoch}")
                print("Try reducing batch size or model complexity")
                torch.cuda.empty_cache()
                break
            else:
                raise e

        torch.cuda.empty_cache()

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
