#!/usr/bin/env python3

from typing import Iterable, List, Tuple

import numpy as np

import torch
import torch.cuda
import torch.optim
import torch.utils.data
import torchvision.datasets

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import transforms

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


def dog_only(target):
    dogness = torch.where(target == CLASS_DOG, torch.ones_like(target, dtype=torch.int8), torch.zeros_like(target, dtype=torch.int8))
    return dogness


# shamelessly stolen from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


def find_nonzero_masks(ds_iter: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List:
    nonzero_masks = list()
    for idx, sample in enumerate(ds_iter):
        if sample[1].count_nonzero().item() > 0:
            nonzero_masks.append(idx)
    
    return nonzero_masks


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name="dogunet"
nproc=8

def train(num_epochs: int, batch_size: int, learning_rate: float=None, val_epoch_freq: int=10):
    # create datasets with our transforms. assume they're already downloaded
    ds_train = torchvision.datasets.VOCSegmentation(
        root="./data/", year="2012", image_set="train", download=False,
        transform=transforms.input_transform, target_transform=transforms.target_transform)
    ds_val = torchvision.datasets.VOCSegmentation(root="./data/",
        year="2012", image_set="val",
        download=False, transform=transforms.input_transform, target_transform=transforms.target_transform)

    print(f"len(ds_train): {len(ds_train)}")
    print(f"len(ds_val): {len(ds_val)}")

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=CLASS_MAX+1, init_features=32, pretrained=False)
    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=nproc)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()
    writer.add_graph(model, ds_train[0][0].unsqueeze(0).to(device))

    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epochs"):
        train_losses = list()
        train_accuracy = list()
        for batch in tqdm(train_dataloader, desc="Batches (train)", unit="batch"):
            img, mask = batch
            img = img.to(device)

            # bring mask to device
            mask = mask.to(device=device)

            # TODO verify that mean of img ~= 0
            # TODO plot img, mask into tensorboard for testing

            pred = model(img)
            pred_s = torch.nn.functional.softmax(pred, dim=1)

            # clip all classes above 20 to zero, for example 255 is "border regions and difficult objects"
            mask = torch.where(mask <= CLASS_MAX, mask, 0).to(dtype=torch.long)
            #target_mask = torch.zeros((CLASS_MAX+1, mask.shape[-2], mask.shape[-1]), dtype=torch.float32, device=device)
            target_mask = torch.zeros_like(pred)
            target_mask.scatter_(1, mask.to(dtype=torch.int64), 1.)

            loss = criterion(pred_s, target_mask)
            train_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # calculate pixel-wise annoation accuracy for all imgs in batch
                acc = (mask.squeeze() == torch.argmax(pred, dim=1))
                acc = (acc.sum()/acc.numel()).item()
                train_accuracy.append(acc)


        epoch_train_loss = np.mean(train_losses)
        epoch_train_accuracy = np.mean(train_accuracy)

        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('PixelAccuracy/train', epoch_train_accuracy, epoch)

        tqdm.write(f"Epoch {epoch+1}; training loss={epoch_train_loss:.4f}; training pixel accuracy={epoch_train_accuracy:.3f}")

        if epoch % val_epoch_freq == val_epoch_freq - 1:
            val_losses = list()
            val_accuracy = list()
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Batches (val)"):
                    img, mask = batch
                    img = img.to(device)

                    # bring mask to device
                    mask = mask.to(device=device)
                    # clip all classes above 20 to zero, for example 255 is "border regions and difficult objects"
                    mask = torch.where(mask <= CLASS_MAX, mask, 0).to(dtype=torch.long)

                    pred = model(img)
                    pred_s = torch.nn.functional.softmax(pred, dim=1)

                    #target_mask = torch.zeros((CLASS_MAX+1, mask.shape[-2], mask.shape[-1]), dtype=torch.float32, device=device)
                    target_mask = torch.zeros_like(pred)
                    target_mask.scatter_(1, mask.to(dtype=torch.int64), 1.)

                    loss = criterion(pred_s, target_mask)
                    val_losses.append(loss.item())

                    # calculate pixel-wise annoation accuracy for alyl imgs in batch
                    acc = (mask.squeeze() == torch.argmax(pred, dim=1))
                    acc = (acc.sum()/acc.numel()).item()
                    val_accuracy.append(acc)

            epoch_val_loss = np.mean(val_losses)
            epoch_val_accuracy = np.mean(val_accuracy)

            writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            writer.add_scalar('PixelAccuracy/val', epoch_val_accuracy, epoch)

            tqdm.write(f"Epoch {epoch+1}; validation loss={epoch_val_loss:.3f}; validation pixel accuracy={epoch_val_accuracy:.3f}")

    torch.save(model, model_name + ".pth")
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Your script description here')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default: 20)')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--learningrate', type=float, default=1e-3, help='Learning Rate (default: torch defaults)')
    parser.add_argument('--validationfreq', type=int, default=10, help='Frequency of validation')

    args = parser.parse_args()


    exit(train(num_epochs=args.epochs, batch_size=args.batchsize, learning_rate=args.learningrate, val_epoch_freq=args.validationfreq))
