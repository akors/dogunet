#!/usr/bin/env python3

from typing import Iterable, List, Tuple

import numpy as np

import torch
import torch.cuda
import torch.optim
import torch.utils.data
import torchvision.datasets

from tqdm import tqdm

from torchvision import transforms

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


#https://github.com/pytorch/vision/issues/6236#issuecomment-1175971587
import torchvision.transforms.functional as F


class Resize_with_pad:
    def __init__(self, w=1024, h=768, interpolation=transforms.InterpolationMode.BILINEAR):
        self.w = w
        self.h = h

    def __call__(self, image):

        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])


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


#m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
transform = transforms.Compose([
    Resize_with_pad(256,256),
    transforms.ToTensor(),
    #transforms.Normalize(mean=m, std=s),
])

target_transform = transforms.Compose([
    Resize_with_pad(256,256, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor(),
])

def find_nonzero_masks(ds_iter: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List:
    nonzero_masks = list()
    for idx, sample in enumerate(ds_iter):
        if sample[1].count_nonzero().item() > 0:
            nonzero_masks.append(idx)
    
    return nonzero_masks


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name="dogunet"
nproc=8

def train(num_epochs: int, batch_size: int, learning_rate: float=None):
    # create datasets with our transforms. assume they're already downloaded
    ds_train = torchvision.datasets.VOCSegmentation(
        root="./data/", year="2012", image_set="trainval", download=False,
        transform=transform, target_transform=target_transform)
    # ds_val = torchvision.datasets.VOCSegmentation(root="./data/",
    #     year="2012", image_set="val",
    #     download=False, transform=transform, target_transform=target_transform)

    # # filter datasets to contain only dogs
    # ds_train = torch.utils.data.Subset(ds_train, find_nonzero_masks(ds_train))
    # ds_val = torch.utils.data.Subset(ds_train, find_nonzero_masks(ds_val))

    print(f"len(ds_train): {len(ds_train)}")
    #print(f"len(ds_val): {len(ds_val)}")

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=CLASS_MAX+1, init_features=32, pretrained=False)
    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=nproc)
    #val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        
        for batch in tqdm(train_dataloader, desc="Batches"):
            img, mask = batch
            img = img.to(device)

            # bring mask to device
            mask = mask.to(device=device)


            pred = model(img)
            pred_s = torch.nn.functional.softmax(pred, dim=1)

            # clip all classes above 20 to zero, for example 255 is "border regions and difficult objects"
            mask = torch.where(mask <= CLASS_MAX, mask, 0).to(dtype=torch.long)
            #target_mask = torch.zeros((CLASS_MAX+1, mask.shape[-2], mask.shape[-1]), dtype=torch.float32, device=device)
            target_mask = torch.zeros_like(pred)
            target_mask.scatter_(1, mask.to(dtype=torch.int64), 1.)

            loss = criterion(pred, target_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Loss after epoch {epoch+1}: {loss.item()}")

    torch.save(model, model_name + ".pth")
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Your script description here')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default: 20)')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--learningrate', type=float, default=None, help='Learning Rate (default: torch defaults)')

    args = parser.parse_args()


    exit(train(num_epochs=args.epochs, batch_size=args.batchsize, learning_rate=args.learningrate))
