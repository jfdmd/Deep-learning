from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import models
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import  load_model, save_model

def masked_l1_loss(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean L1 loss only over valid waypoints.

    Args:
        preds: (b, n, 2)
        labels: (b, n, 2)
        mask: (b, n) boolean or float
    """
    maskf = mask.float()
    diff = (preds - labels).abs() * maskf[..., None]
    denom = maskf.sum() * preds.shape[-1]
    if denom == 0:
        return torch.tensor(0.0, device=preds.device)
    return diff.sum() / denom


def make_dataloaders(dataset_path: Path, model_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders. If subdirs `train`/`val` exist, use them.
    Otherwise use the provided path for both (with shuffle differences).
    """
    # choose transform pipeline based on model type
    if model_name == "cnn_planner":
        transform = "default"
    else:
        transform = "state_only"

    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    if train_path.is_dir():
        train_loader = load_data(str(train_path), transform_pipeline=transform, batch_size=batch_size, shuffle=True)
    else:
        train_loader = load_data(str(dataset_path), transform_pipeline=transform, batch_size=batch_size, shuffle=True)

    if val_path.is_dir():
        val_loader = load_data(str(val_path), transform_pipeline=transform, batch_size=batch_size, shuffle=False)
    else:
        # reuse same dataset for quick validation if separate val not provided
        val_loader = load_data(str(dataset_path), transform_pipeline=transform, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    transform_pipeline: str = 'state_only',
    dataset_name="drive_data",
    num_workers: int = 4,
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 2024,
    **kwargs,
):
    """Trains a model."""
    #def train_epoch(
    #    model: torch.nn.Module
    #    loader: DataLoader
    #    optimizer: torch.optim.Optimizer
    #    device: torch.device
    #) -> float:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, **kwargs).to(device)
    train_loader, val_loader = make_dataloaders(Path(dataset_name), model_name, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    best_val_loss = float("inf")
    start_time = time.time()


    for epoch in range(1, num_epoch + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # prepare inputs depending on model type
            if isinstance(model, models.CNNPlanner):
                image = batch["image"].to(device)
                preds = model(image=image)
            else:
                # Assuming batch is a dictionary based on load_data and transform_pipeline
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                preds = model(track_left=track_left, track_right=track_right)


            labels = batch["waypoints"].to(device)
            labels_mask = batch["waypoints_mask"].to(device)

            loss = masked_l1_loss(preds, labels, labels_mask)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)


        val_loss, stats = validate(model, val_loader, device)
        t1 = time.time()

        print(
            f"Epoch {epoch}/{num_epoch}  time={t1-t0:.1f}s  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "+
            f"long={stats['longitudinal_error']:.4f} lat={stats['lateral_error']:.4f} samples={stats['num_samples']}"
        )

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = models.save_model(model)
            print(f"Saved best model to {output_path} (val_loss={best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"Time to train {total_time/60:.2f} minutes")


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, dict]:
    model.eval()
    metric = PlannerMetric()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        if isinstance(model, models.CNNPlanner):
            image = batch["image"].to(device)
            preds = model(image=image)
        else:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            preds = model(track_left=track_left, track_right=track_right)

        labels = batch["waypoints"].to(device)
        labels_mask = batch["waypoints_mask"].to(device)

        loss = masked_l1_loss(preds, labels, labels_mask)

        running_loss += float(loss.item())
        n_batches += 1

        # metrics expect cpu numpy
        metric.add(preds.cpu(), labels.cpu(), labels_mask.cpu())

    avg_loss = running_loss / max(1, n_batches)
    stats = metric.compute()

    return avg_loss, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(models.MODEL_FACTORY.keys()), required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Training {args.model} on {args.dataset} for {args.epochs} epochs (device={device})")

    # instantiate model
    ModelClass = models.MODEL_FACTORY[args.model]

    # default constructor should be fine; the grader will call load_model when evaluating
    model = ModelClass()
    model.to(device)

    train_loader, val_loader = make_dataloaders(Path(args.dataset), args.model, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, stats = validate(model, val_loader, device)
        t1 = time.time()

        print(
            f"Epoch {epoch}/{args.epochs}  time={t1-t0:.1f}s  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "+
            f"long={stats['longitudinal_error']:.4f} lat={stats['lateral_error']:.4f} samples={stats['num_samples']}"
        )

        # save best
        if val_loss < best_val_loss:
            best_val_loss = best_val_loss
            output_path = models.save_model(model)
            print(f"Saved best model to {output_path} (val_loss={best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"Time to train {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()