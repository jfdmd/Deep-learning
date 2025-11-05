import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .road_dataset import load_data
from .metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "detection",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    depth_loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    #metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        #for key in metrics:
            #metrics[key].clear()
        
        metric = DetectionMetric()
        model.train()

        for batch in train_data:
            depth_labels = batch["depth"].to(device ="cuda")
            track_labels = batch["track"].to(device ="cuda") 
            img = batch["image"].to(device ="cuda")

            # forward segmentaion loss crossentropy
            logits, depth_preds = model(img)
            loss = loss_func(logits, track_labels)
            depth_loss = depth_loss_func(depth_labels, depth_preds)

            # backward + step
            optimizer.zero_grad()
            (loss+depth_loss).backward()
            optimizer.step()
            track_preds, depth_preds = model.predict(img)

            # compute accuracy for this batch and save
            #preds = torch.argmax(logits, dim=1)
            #batch_acc_track = (track_pred == track_label).float().mean()
            #batch_acc_depth = (depth_pred == depth_label).float().mean()
            #metrics["depth_error"].append(batch_acc_depth.detach().cpu())
            #metrics["track_error"].append(batch_acc_track.detach().cpu())

            # log training loss every iteration
            logger.add_scalar("train_loss", float(loss.detach().cpu()), global_step)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                DetectionMetric.add(batch)
            IOU, abs_depth_error, tp_depth_error = DetectionMetric.compute()
                
              #depth_label = batch["depth"].to(device ="cuda")
              #track_label= batch["track"].to(device ="cuda") 
              #img = batch["image"].to(device ="cuda")
              #track_pred, depth_pred = model.predict(img)
               
             # compute accuracy for this batch and save
            #preds = torch.argmax(logits, dim=1)
            #batch_acc_track = (track_pred == track_label).float().mean()
            #batch_acc_depth = (depth_pred == depth_label).float().mean()
            #metrics["depth_error"].append(batch_acc_depth.detach().cpu())
            #metrics["track_error"].append(batch_acc_track.detach().cpu())
            #metrics["val_acc"].append(batch_acc.detach().cpu())

        # log average train and val accuracy to tensorboard
        #epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        #epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        #logger.add_scalar("train_accuracy", float(epoch_train_acc), global_step)
        #logger.add_scalar("val_accuracy", float(epoch_val_acc), global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"IOU={IOU:.4f} "
                f"Depth_error={abs_depth_error:.4f}"
                f"Lane_boundry_error={tp_depth_error:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
