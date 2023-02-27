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
batch_size = 8
learning_rate = 1e-4


def train(num_epochs: int):
    # create datasets with our transforms. assume they're already downloaded
    ds_train = torchvision.datasets.VOCSegmentation(
        root="./data/", year="2012", image_set="train", download=False,
        transform=transform, target_transform=target_transform)
    ds_val = torchvision.datasets.VOCSegmentation(root="./data/",
        year="2012", image_set="val",
        download=False, transform=transform, target_transform=target_transform)

    # # filter datasets to contain only dogs
    # ds_train = torch.utils.data.Subset(ds_train, find_nonzero_masks(ds_train))
    # ds_val = torch.utils.data.Subset(ds_train, find_nonzero_masks(ds_val))

    print(f"len(ds_train): {len(ds_train)}")
    print(f"len(ds_val): {len(ds_val)}")

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=16, pretrained=False)
    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    #val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            img, mask = batch
            img = img.to(device)
            mask = mask.to(device='cuda', dtype=torch.float32)

            pred = model(img)
            loss = criterion(pred, mask)

            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss after epoch {epoch+1}: {loss.item()}")

    torch.save(model, "dogunet.pth")
    return 0

if __name__ == "__main__":
    exit(train(10))
