#!/usr/bin/env python3

import re
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib

import torch
import torch.cuda
import torch.optim
import torch.utils.data
import torchvision.datasets
import torchvision.utils

from brain_segmentation_pytorch.loss import DiceLoss
import brain_segmentation_pytorch.unet

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import transforms
import visualize
import logutils

classnames = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'potted plant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv/monitor'
}

CLASS_MAX = np.amax([c for c in classnames.keys()])
CLASS_DOG = 12


class MetricsWriter():
    def __init__(self, tb_writer: SummaryWriter, max_len: int, scalar_tags: Optional[List[str]]=None):
        self.__tb_writer: SummaryWriter = tb_writer
        self.__max_len: int = max_len

        if scalar_tags is None:
            scalar_tags = list()

        self.__scalar_metrics: Dict[str, np.array] = dict()

        for scalar in scalar_tags:
            self.add_scalar(scalar)
    
        self.__current_step: int = 0
    
    def add_scalar(self, tag: str):
        assert tag not in self.__scalar_metrics.keys(), "Tag has already been added"
        self.__scalar_metrics[tag] = np.empty((self.__max_len,))
    
    def set_step(self, step: int):
        self.__current_step = step

    def add_sample(self, tag: str, value: float):
        self.__scalar_metrics[tag][self.__current_step] = value
    
    def get(self, tag: str):
        return np.nanmean(self.__scalar_metrics[tag])

    def write(self, global_step: int):
        for tag in self.__scalar_metrics.keys():
            metric_value = self.get(tag)
            self.__tb_writer.add_scalar(tag, metric_value, global_step=global_step)

    def clear(self):
        self.__current_step = 0
        for samples in self.__scalar_metrics.values():
            samples.fill(np.nan)

class CheckpointSaver:
    def __init__(self, basepath: str,
                 model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_size: int, checkpoint_freq: int) -> None:
        self.__basepath: str = basepath
        self.__model: torch.nn.Module = model
        self.__optimizer: torch.optim.Optimizer = optimizer
        self.__batch_size: int = batch_size
        self.__checkpoint_freq = checkpoint_freq

        self.__disable = checkpoint_freq == 0

    def save_now(self, epoch: int, numbered_chpt: bool = False):
        if self.__checkpoint_freq < 0:
            return

        # if checkpointing is enabled, save final training checkpoint
        out_checkpoint = {
            'batch_size': self.__batch_size,
            'epoch': epoch+1,
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict()
        }

        torch.save(out_checkpoint, self.__basepath + ".chpt.pt") # save current/final

        if numbered_chpt:
            # copy to numbered checkpoint if requested
            shutil.copy2(f"{self.__basepath}.chpt.pt", f"{self.__basepath}.chpt{epoch+1}.pt")

    def save_if_needed(self, epoch: int, numbered_chpt: bool = False):
        if self.__checkpoint_freq < 1:
            return

        if epoch % self.__checkpoint_freq == self.__checkpoint_freq-1:
            self.save_now(epoch=epoch, numbered_chpt=numbered_chpt)
        else:
            pass # don't save if it's not time yet


device = 'cuda' if torch.cuda.is_available() else 'cpu'
nproc=8
input_debug=False
boundary_loss_weight=0.5
unet_features=32
write_hparams=False
activations_log_layers=[
    "bottleneck"
]

def make_datasets(datadir: str="./data/", use_2012=True, use_2007=False):
    assert use_2007 or use_2012, "Please select one or both of the datasets"
    ds_train_list = list()
    ds_val_list = list()

    if use_2012:
        # create datasets with our transforms. assume they're already downloaded
        ds_train_list.append(torchvision.datasets.VOCSegmentation(
            root=datadir, year="2012", image_set="train", download=False,
            transform=transforms.input_transform(transforms.PASCAL_VOC_2012_MEAN, transforms.PASCAL_VOC_2012_STD),
            target_transform=transforms.target_transform(max_class=transforms.PASCAL_VOC_2012_CLASS_MAX)
        ))
        ds_val_list.append(torchvision.datasets.VOCSegmentation(
            root=datadir,
            year="2012", image_set="val", download=False,
            transform=transforms.input_transform(transforms.PASCAL_VOC_2012_MEAN, transforms.PASCAL_VOC_2012_STD),
            target_transform=transforms.target_transform(max_class=transforms.PASCAL_VOC_2012_CLASS_MAX)
        ))

    ds_train = torch.utils.data.ConcatDataset(ds_train_list)
    ds_val = torch.utils.data.ConcatDataset(ds_val_list)

    return ds_train, ds_val


def train(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float=None,
    val_epoch_freq: int=10,
    resume: Optional[str]=None,
    run_comment: str="",
    checkpointfreq: int=0,
    log_activations: Optional[str]=None
):
    matplotlib.use('Agg')

    ds_train, ds_val = make_datasets()
    print(f"Training dataset length: {len(ds_train)}")
    print(f"Validation dataset length: {len(ds_val)}")

    if checkpointfreq != 0:
        print(f"Writing checkpoints to {model_name}.chpt*.pt and {model_name}.chpt.pt")

    # needed for visualization
    inv_normalize = transforms.inv_normalize(transforms.PASCAL_VOC_2012_MEAN, transforms.PASCAL_VOC_2012_STD)


    model = brain_segmentation_pytorch.unet.UNet(
        in_channels=3,
        out_channels=CLASS_MAX+1,
        init_features=unet_features,
    )

    checkpoint = None
    if resume is not None:
        # load checkpoint when resuming
        checkpoint = torch.load(resume)

        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=nproc)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    criterion_boundaries = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if checkpointfreq != 0:
        chpt_saver = CheckpointSaver(basepath=model_name, model=model, optimizer=optimizer,
                                     batch_size=batch_size, checkpoint_freq=checkpointfreq)
    else:
        chpt_saver = None

    resume_epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_epoch = checkpoint['epoch']

        del checkpoint # not needed after this points
        print(f"Resuming training from checkpoint {resume} at epoch {resume_epoch}")

    writer = SummaryWriter(comment=run_comment)
    writer.add_graph(model, ds_train[0][0].unsqueeze(0).to(device))

    # initialize epoch metrics
    metrics_train_epoch = MetricsWriter(writer, max_len=len(train_dataloader), scalar_tags=[
        "Loss/train/total",
        "Loss/train/pixelclass",
        "Loss/train/boundary",
        "Accuracy/train/pixelwise"
    ])

    metrics_val_epoch = MetricsWriter(writer, max_len=len(val_dataloader), scalar_tags=[
        "Loss/val/total",
        "Loss/val/pixelclass",
        "Loss/val/boundary",
        "Accuracy/val/pixelwise",
    ])

    activations_logger = None
    if log_activations is not None:
        activations_logger = logutils.ActivationsLogger(model=model, writer=writer, layers=activations_log_layers)


    ds_train_len = len(ds_train)
    global_step = lambda: (
        ds_train_len*epoch + batch_size * (batch_idx+1)
        - batch_size + batch[0].size(0)
    )
    for epoch in tqdm(range(resume_epoch, num_epochs+resume_epoch), desc="Epochs", unit="ep"):
        # ensure model is in training mode
        model.train(True)

        # hook in activations logger if needed
        if log_activations is not None:
            if log_activations in ("all", "train"):
                activations_logger.enable()
            else:
                activations_logger.disable()

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Batches (train)", unit="batch")):
            img, mask = batch

            metrics_train_epoch.set_step(batch_idx)

            # bring sample to device
            img: torch.Tensor = img.to(device=device)
            mask: torch.Tensor = mask.to(device=device)

            if input_debug:
                img_std, img_mean = torch.std_mean(img)
                writer.add_scalar("DbgTrainImageDist/mean", img_mean, global_step=global_step())
                writer.add_scalar("DbgTrainImageDist/std", img_std, global_step=global_step())

                # TODO plot img, mask into tensorboard for testing
                if batch_idx == 0:
                    dbg_input_len = min(img.size(0), 4)
                    dbg_input_img = img[0:dbg_input_len,:,:,:]
                    dbg_input_mask = visualize.classmask_to_colormask(mask[0:dbg_input_len,:,:])

                    dbg_input_grid = torchvision.utils.make_grid(
                        torch.cat((dbg_input_img, dbg_input_mask), dim=0),
                        nrow=dbg_input_len)
                    writer.add_image("DbgTrainInput", dbg_input_grid, global_step=global_step())

            pred = model(img)
            pred_s = torch.nn.functional.softmax(pred, dim=1)

            #target_mask = torch.zeros((CLASS_MAX+1, mask.shape[-2], mask.shape[-1]), dtype=torch.float32, device=device)
            target_mask = torch.zeros_like(pred)
            target_mask.scatter_(1, mask.unsqueeze(1), 1.)

            # compose loss by boundary loss and pixel classification
            loss_pixelclass = criterion(pred_s, target_mask)
            loss_boundary = criterion_boundaries(pred_s, target_mask)

            loss = (1.-boundary_loss_weight) * loss_pixelclass + boundary_loss_weight * loss_boundary

            metrics_train_epoch.add_sample('Loss/train/total', loss.item())
            metrics_train_epoch.add_sample('Loss/train/pixelclass', loss_pixelclass.item())
            metrics_train_epoch.add_sample('Loss/train/boundary', loss_boundary.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # calculate pixel-wise annoation accuracy for all imgs in batch
                acc = (mask == torch.argmax(pred, dim=1))
                acc = (acc.sum()/acc.numel())
                metrics_train_epoch.add_sample('Accuracy/train/pixelwise', acc.item())

        metrics_train_epoch.write(global_step=global_step())

        # flush activation histograms for this epoch
        if log_activations is not None and log_activations in ("all", "train"):
            activations_logger.flush(global_step=global_step(), phase="train")

        if checkpointfreq > 0:
            chpt_saver.save_if_needed(epoch, numbered_chpt=True)

        tqdm.write(f"Epoch {epoch+1}; Training Loss: {metrics_train_epoch.get('Loss/train/total'):.4f}; " +
            f"Training Pixel Accuracy: {metrics_train_epoch.get('Accuracy/train/pixelwise'):.3f}")

        if epoch % val_epoch_freq == val_epoch_freq - 1: # if validating
            # use eval mode for validation, disabled batchnorm layers?
            model.train(False)

            # hook in activations logger if needed
            if log_activations is not None and log_activations in ("all", "val"):
                activations_logger.enable()

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(tqdm(val_dataloader, desc="Batches (val)")):
                    img, mask = val_batch
                    img = img.to(device=device)
                    mask = mask.to(device=device)

                    pred = model(img)
                    pred_s = torch.nn.functional.softmax(pred, dim=1)

                    target_mask = torch.zeros_like(pred)
                    target_mask.scatter_(1, mask.unsqueeze(1), 1.)

                    # compose loss by boundary loss and pixel classification
                    loss_pixelclass = criterion(pred_s, target_mask)
                    loss_boundary = criterion_boundaries(pred_s, target_mask)

                    loss = (1.-boundary_loss_weight) * loss_pixelclass + boundary_loss_weight * loss_boundary

                    metrics_val_epoch.set_step(val_batch_idx)
                    metrics_val_epoch.add_sample('Loss/val/total', loss.item())
                    metrics_val_epoch.add_sample('Loss/val/pixelclass', loss_pixelclass.item())
                    metrics_val_epoch.add_sample('Loss/val/boundary', loss_boundary.item())

                    # calculate pixel-wise annoation accuracy for alyl imgs in batch
                    acc = (mask == torch.argmax(pred, dim=1))
                    acc = (acc.sum()/acc.numel())
                    metrics_val_epoch.add_sample('Accuracy/val/pixelwise', acc.item())


            metrics_val_epoch.write(global_step=global_step())

            # prepare comparison grid for the first three samples in training dataset
            vis_samples = 3 # number of samples to visualize per image
            val_imgs = torch.stack([ds_train[i][0] for i in range(vis_samples)]).to(device=device)
            val_masks = torch.stack([ds_train[i][1] for i in range(vis_samples)]).to(device=device)
            pred = model(val_imgs)
            pred_amax = torch.argmax(pred, dim=1)

            comparison_fig_t = visualize.make_comparison_grid(inv_normalize(val_imgs), pred_amax, val_masks)
            writer.add_image("PredictionComparison/train", comparison_fig_t, global_step=global_step())

            # prepare comparison grid for the first three samples in training dataset
            val_imgs = torch.stack([ds_val[i][0] for i in range(vis_samples)]).to(device=device)
            val_masks = torch.stack([ds_val[i][1] for i in range(vis_samples)]).to(device=device)
            pred = model(val_imgs)
            pred_amax = torch.argmax(pred, dim=1)

            comparison_fig_t = visualize.make_comparison_grid(inv_normalize(val_imgs), pred_amax, val_masks)
            writer.add_image("PredictionComparison/val", comparison_fig_t, global_step=global_step())

            # flush activation histograms for this epoch
            if log_activations is not None and log_activations in ("all", "val"):
                activations_logger.flush(global_step=global_step(), phase="val")

            tqdm.write(f"Epoch {epoch+1}; Validation Loss: {metrics_val_epoch.get('Loss/val/total'):.4f}; " +
                f"Validation Pixel Accuracy: {metrics_val_epoch.get('Accuracy/val/pixelwise'):.3f}")
            
            pass # if validating

    if write_hparams:
        writer.add_hparams(
            hparam_dict={
                "lr" : learning_rate,
                "batchsize": batch_size,
                "features": unet_features
            },
            metric_dict={
                "ValidationLoss": metrics_val_epoch.get('Loss/val/total'),
                "PixelAccuracy": metrics_val_epoch.get('Accuracy/val/pixelwise')
            }
        )

    # save training checkpoint
    if checkpointfreq != 0:
        chpt_saver.save_now(epoch)

    # save model parameters only
    torch.save(model.state_dict(), model_name + ".pt")

    writer.close()
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train dogunet network')
    parser.add_argument('-n', '--name', type=str, default="dogunet",
                        help="Name of the model, used for model state, checkpoint and TB run name.")
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--learningrate', type=float, default=1e-3, help='Learning Rate (default: torch defaults)')
    parser.add_argument('--validationfreq', type=int, default=10, help='Frequency of validation')
    parser.add_argument('--resume', type=str, help='Resume training from this checkpoint', metavar="MODEL.pt")
    parser.add_argument('--runcomment', type=str, default="", help="Comment to append to the name in TensorBoard")
    parser.add_argument('--checkpointfreq', type=int, default=-1, help="Checkpoint frequency in epochs. 0 for off. -1 for only final.")
    parser.add_argument('--log-activations', nargs='?', type=str, choices=["all", "train", "val"],
        help="Log activation histograms to TensorBoard."
    )

    args = parser.parse_args()

    ret = train(
        model_name=args.name,
        num_epochs=args.epochs,
        batch_size=args.batchsize,
        learning_rate=args.learningrate,
        val_epoch_freq=args.validationfreq,
        resume=args.resume,
        run_comment=args.runcomment,
        checkpointfreq=args.checkpointfreq,
        log_activations=args.log_activations
    )
    exit(ret)
