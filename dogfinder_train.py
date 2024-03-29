#!/usr/bin/env python3

import os
import shutil
import subprocess
from typing import Optional

import matplotlib

import torch
import torch.cuda
import torch.optim
import torch.utils.data
import torchvision.datasets
import torchvision.utils

#from brain_segmentation_pytorch.loss import DiceLoss
from monai.losses import DiceLoss
import brain_segmentation_pytorch.unet

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import datasets
import transforms
import visualize
from metrics import MultiMetrics
import logutils
from logutils import MetricsWriter, ActivationsLogger


def get_git_revision_short_hash():
    wd = os.path.dirname(__file__)
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE, cwd=wd)
    except FileNotFoundError:
        # Could not find git binary
        return None

    if result.returncode != 0:
        return None
    else:
        return result.stdout.decode('ascii').strip()

def get_git_diff():
    wd = os.path.dirname(__file__)
    try:
        result = subprocess.run(['git', 'diff', 'HEAD', '*.py'], stdout=subprocess.PIPE, cwd=wd)
    except FileNotFoundError:
        # Could not find git binary
        return None

    if result.returncode != 0:
        return None
    else:
        return result.stdout.decode('ascii', errors="ignore").strip()



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
        if self.__checkpoint_freq == 0:
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
input_debug=False
boundary_loss_weight=0.5
write_hparams=False


def train(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    unet_features: int=32,
    nproc: int=2,
    learning_rate: float=None,
    val_epoch_freq: int=10,
    resume: Optional[str]=None,
    run_comment: Optional[str]=None,
    checkpointdir: str="./checkpoints/",
    checkpointfreq: int=0,
    augment_level: int=0,
    dataset_years=["2012"],
    trainsuite="train",
    log_activations: Optional[str]=None,
    log_weights: bool=False
):
    matplotlib.use('Agg')

    ds_train, ds_val = datasets.make_datasets(augment_level=augment_level, years=dataset_years, trainsuite=trainsuite)
    print(f"Training dataset length: {len(ds_train)}")
    print(f"Validation dataset length: {len(ds_val)}")

    # needed for visualization
    inv_normalize = transforms.inv_normalize(*datasets.get_dataset_mean_std(dataset_years))

    model = brain_segmentation_pytorch.unet.UNet(
        in_channels=3,
        out_channels=datasets.CLASS_MAX+1,
        init_features=unet_features,
    )

    checkpoint = None
    if resume is not None:
        # load checkpoint when resuming
        checkpoint = torch.load(resume)

        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=nproc)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=nproc)

    criterion_class = torch.nn.CrossEntropyLoss()
    #criterion_boundaries = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    if checkpointfreq != 0:
        if not os.path.isdir(checkpointdir):
            os.mkdir(checkpointdir)

        print(f"Writing recent/last checkpoints to {os.path.join(checkpointdir, model_name)}.chpt.pt")
        if checkpointfreq > 0:
            print(f"Writing intermediate checkpoints to {os.path.join(checkpointdir, model_name)}.chpt*.pt")

        chpt_saver = CheckpointSaver(basepath=os.path.join(checkpointdir, model_name), model=model, optimizer=optimizer,
                                     batch_size=batch_size, checkpoint_freq=checkpointfreq)
    else:
        chpt_saver = None

    resume_epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_epoch = checkpoint['epoch']

        del checkpoint # not needed after this points
        print(f"Resuming training from checkpoint {resume} at epoch {resume_epoch}")

    # compose TB run comment from model name and our run comment argument
    comment = "_" + model_name
    if run_comment is not None:
        comment += "_"
        comment += run_comment

    writer = SummaryWriter(comment=comment)
    print("TensorBoard logdir: ", writer.get_logdir())

    # store git hash and diff, if available
    git_hash = get_git_revision_short_hash()
    git_diff = get_git_diff()

    if git_hash is not None:
        writer.add_text("RunInfo/git_hash", git_hash)

    if git_diff is not None and len(git_diff) > 0:
        writer.add_text("RunInfo/git_diff", '```\n'+git_diff+'\n```')

    writer.add_graph(model, ds_train[0][0].unsqueeze(0).to(device))

    # initialize epoch metrics
    metrics_train_epoch = MetricsWriter(writer, max_len=len(train_dataloader), scalar_tags=[
        "Loss/train/total",
        #"Loss/train/pixelclass",
        #"Loss/train/boundary",
    ])

    metrics_val_epoch = MetricsWriter(writer, max_len=len(val_dataloader), scalar_tags=[
        "Loss/val/total",
        #"Loss/val/pixelclass",
        #"Loss/val/boundary",
    ])

    multimetrics = MultiMetrics()

    activations_logger = None
    if log_activations is not None:
        activations_logger = ActivationsLogger(model=model, writer=writer, layers=log_activations.split(","))
        activations_logger.enable()


    ds_train_len = len(ds_train)
    for epoch in tqdm(range(resume_epoch, num_epochs+resume_epoch), desc="Epochs", unit="ep"):
        # will this epoch need a validation run and detailed metrics
        is_validating_poch = (epoch % val_epoch_freq) == (val_epoch_freq - 1)

        # ensure model is in training mode
        model.train(True)

        # reset metrics before epoch
        multimetrics.reset()

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Batches (train)", unit="batch")):
            current_global_step = (
                ds_train_len*epoch + batch_size * (batch_idx+1)
                - batch_size + batch[0].size(0)
            )
            img, mask = batch

            metrics_train_epoch.set_step(batch_idx)

            # bring sample to device
            img: torch.Tensor = img.to(device=device)
            mask: torch.Tensor = mask.to(device=device)
            mask = mask.to(dtype=torch.long)

            if input_debug:
                img_std, img_mean = torch.std_mean(img)
                writer.add_scalar("DbgTrainImageDist/mean", img_mean, global_step=current_global_step)
                writer.add_scalar("DbgTrainImageDist/std", img_std, global_step=current_global_step)

                if batch_idx == 0:
                    dbg_input_len = min(img.size(0), 4)
                    dbg_input_img = inv_normalize(img[0:dbg_input_len,:,:,:])
                    dbg_input_mask = visualize.classmask_to_colormask(mask[0:dbg_input_len,0,:,:])

                    dbg_input_grid = torchvision.utils.make_grid(
                        torch.cat((dbg_input_img, dbg_input_mask), dim=0),
                        nrow=dbg_input_len)
                    writer.add_image("DbgTrainInput", dbg_input_grid, global_step=current_global_step)

            pred = model(img)
            pred_l = torch.logit(pred, eps=1e-6) # model outputs sigmoid, we also need logits
            #pred_s = torch.nn.functional.softmax(pred_l, dim=1)

            #mask_onehot = torch.zeros_like(pred).scatter_(1, mask, 1.)

            # compose loss by boundary loss and pixel classification
            loss_pixelclass = criterion_class(pred_l, mask[:,0,:,:])
            #loss_boundary = criterion_boundaries(pred_l, mask)

            #loss = (1.-boundary_loss_weight) * loss_pixelclass + boundary_loss_weight * loss_boundary
            loss = loss_pixelclass

            metrics_train_epoch.add_sample('Loss/train/total', loss.item())
            #metrics_train_epoch.add_sample('Loss/train/pixelclass', loss_pixelclass.item())
            #metrics_train_epoch.add_sample('Loss/train/boundary', loss_boundary.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # flush activation histograms for this epoch
                if activations_logger is not None: activations_logger.flush(global_step=current_global_step, phase="train")

                multimetrics.update(pred=pred, target=mask, detailed=is_validating_poch)

            # delete in the hopes of saving some memory
            del img, mask, batch #, mask_onehot

            pass # end for loop training batches

        metrics_train_epoch.write(global_step=current_global_step)

        multimetrics_train_epoch = multimetrics.calculate(detailed=is_validating_poch)
        logutils.log_metrics_dict(multimetrics_train_epoch, writer, global_step=current_global_step, prefix="Metrics/train/")

        if checkpointfreq > 0:
            chpt_saver.save_if_needed(epoch, numbered_chpt=True)

        if log_weights:
            logutils.log_weights(model=model, writer=writer, global_step=current_global_step)

        tqdm.write(f"Epoch {epoch+1}; "
            f"Training Loss: {metrics_train_epoch.get('Loss/train/total'):.4f}; " +
            f"Training Pixel Accuracy: {multimetrics_train_epoch['OverallAccuracy']:.3f}")

        if is_validating_poch:
            # use eval mode for validation, disabled batchnorm layers?
            model.train(False)

            multimetrics.reset()

            # hook in activations logger if needed
            if log_activations is not None and log_activations in ("all", "val"):
                activations_logger.enable()

            with torch.no_grad():
                for val_batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Batches (val)")):
                    img, mask = batch
                    img = img.to(device=device)
                    mask = mask.to(device=device)
                    mask = mask.to(dtype=torch.long)

                    pred = model(img)
                    pred_l = torch.logit(pred, eps=1e-6) # model outputs sigmoid, we also need logits
                    #pred_s = torch.nn.functional.softmax(pred_l, dim=1)

                    #mask_onehot = torch.zeros_like(pred).scatter_(1, mask, 1.)

                    # compose loss by boundary loss and pixel classification
                    loss_pixelclass = criterion_class(pred_l, mask[:,0,:,:])
                    #loss_boundary = criterion_boundaries(pred_l, mask)

                    #loss = (1.-boundary_loss_weight) * loss_pixelclass + boundary_loss_weight * loss_boundary
                    loss = loss_pixelclass

                    metrics_val_epoch.set_step(val_batch_idx)
                    metrics_val_epoch.add_sample('Loss/val/total', loss.item())
                    #metrics_val_epoch.add_sample('Loss/val/pixelclass', loss_pixelclass.item())
                    #metrics_val_epoch.add_sample('Loss/val/boundary', loss_boundary.item())

                    multimetrics.update(pred, mask, detailed=is_validating_poch)

                    # flush activation histograms for this epoch
                    if activations_logger is not None: activations_logger.flush(global_step=current_global_step, phase="val")

                # delete in the hopes of saving some memory
                del img, mask, batch #, mask_onehot

                metrics_val_epoch.write(global_step=current_global_step)

                multimetrics_val_epoch = multimetrics.calculate(detailed=is_validating_poch)
                logutils.log_metrics_dict(multimetrics_val_epoch, writer, global_step=current_global_step, prefix="Metrics/val/")

                # prepare comparison grid for the first three samples in training dataset
                vis_samples = 3 # number of samples to visualize per image
                val_samples = [ds_train[i] for i in range(vis_samples)]
                val_imgs = torch.stack([s[0] for s in val_samples]).to(device=device)
                val_masks = torch.stack([s[1][0,:,:] for s in val_samples]).to(device=device)
                pred = model(val_imgs)
                pred_amax = torch.argmax(pred, dim=1)

                comparison_fig_t = visualize.make_comparison_grid(inv_normalize(val_imgs), pred_amax, val_masks)
                writer.add_image("PredictionComparison/train", comparison_fig_t, global_step=current_global_step)

                # prepare comparison grid for the first three samples in training dataset
                val_samples = [ds_val[i] for i in range(vis_samples)]
                val_imgs = torch.stack([s[0] for s in val_samples]).to(device=device)
                val_masks = torch.stack([s[1][0,:,:] for s in val_samples]).to(device=device)
                pred = model(val_imgs)
                pred_amax = torch.argmax(pred, dim=1)

                comparison_fig_t = visualize.make_comparison_grid(inv_normalize(val_imgs), pred_amax, val_masks)
                writer.add_image("PredictionComparison/val", comparison_fig_t, global_step=current_global_step)

                # flush activation histograms for this epoch
                if log_activations is not None and log_activations in ("all", "val"):
                    activations_logger.flush(global_step=current_global_step, phase="val")

                tqdm.write(f"Epoch {epoch+1}; "
                    f"Validation Loss: {metrics_val_epoch.get('Loss/val/total'):.4f}; " +
                    f"Validation Pixel Accuracy: {multimetrics_val_epoch['OverallAccuracy']:.3f}")
                
                del val_samples, val_imgs, val_masks

            pass # if is_validating_poch

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
    torch.save(model.state_dict(), os.path.join(checkpointdir, model_name + ".pt"))

    writer.close()
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train dogunet network')
    parser.add_argument('-n', '--name', type=str, default="dogunet",
                        help="Name of the model, used for model state, checkpoint and TB run name.")
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('-b', '--batchsize', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--unet-features', type=int, default=32,
                        help='Number of feature channels for UNet intermediate convolutions (default: 32)')                         
    parser.add_argument('-j', '--nproc', type=int, default=2,
                        help='Number of CPU workers that will be used to load/transform data (default: 2)')
    parser.add_argument('-l', '--learningrate', type=float, default=1e-3,
                        help='Learning Rate (default: torch defaults)')
    parser.add_argument('--validationfreq', type=int, default=20,
                        help='Frequency of validation')
    parser.add_argument('-r', '--resume', type=str,
                        help='Resume training from this checkpoint', metavar="MODEL.pt")
    parser.add_argument('-c', '--runcomment', type=str, default=None,
                        help="Comment to append to the name in TensorBoard")
    parser.add_argument('--checkpointdir', type=str, default="./checkpoints/",
                        help="Directory where checkpoints will be saved. Default: ./checkpoints")
    parser.add_argument('--checkpointfreq', type=int, default=-1,
                        help="Checkpoint frequency in epochs. 0 for off. -1 for only final.")
    parser.add_argument('-a', '--augmentation-level', type=int, default=1,
                        help="Augmentation level. 0 for disabled, 1 for basic geometric. (default: 1)")
    parser.add_argument('--dataset-years', nargs='+', default=["2012"], 
                        help="Which PASCAL VOC competition years to take into the dataset. Defaults to 2012 only.")
    parser.add_argument('--trainsuite', type=str, default="train", choices=('train', 'val', 'trainval'),
                        help="Dataset split used for training. Defaults to train.")
    parser.add_argument('--log-activations', type=str, metavar="LAYERS",
                        help="Log histograms of activations for LAYERS to TensorBoard. Argument is a comma-separated "+
                        "list of layers, as defined by the model."
                        )
    parser.add_argument('--log-weights', action="store_true",
                        help="Log weight histograms after each epoch to TensorBoard.")

    args = parser.parse_args()

    ret = train(
        model_name=args.name,
        num_epochs=args.epochs,
        batch_size=args.batchsize,
        unet_features=args.unet_features,
        nproc=args.nproc,
        learning_rate=args.learningrate,
        val_epoch_freq=args.validationfreq,
        resume=args.resume,
        run_comment=args.runcomment,
        checkpointdir=args.checkpointdir,
        checkpointfreq=args.checkpointfreq,
        augment_level=args.augmentation_level,
        dataset_years=args.dataset_years,
        trainsuite=args.trainsuite,
        log_activations=args.log_activations,
        log_weights=args.log_weights
    )
    exit(ret)
