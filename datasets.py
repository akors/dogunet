

import torch
import torchvision
import transforms

CLASSNAMES = {
    0: 'background',
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

CLASS_MAX = 20
CLASS_DOG = 12

def make_datasets(datadir: str="./data/", years=["2012"], augment_level: int=0):
    assert len(years) > 0, "Need to select at least one year between 2007 and 2012"
    ds_train_list = list()
    ds_val_list = list()


    tr_train = transforms.make_transforms(
        transforms.PASCAL_VOC_2012_MEAN, transforms.PASCAL_VOC_2012_STD, augment=(augment_level > 0))
    tr_val = transforms.make_transforms(
        transforms.PASCAL_VOC_2012_MEAN, transforms.PASCAL_VOC_2012_STD, augment=False)

    for year in years:
        # create datasets with our transforms. assume they're already downloaded
        ds_train_list.append(torchvision.datasets.wrap_dataset_for_transforms_v2(torchvision.datasets.VOCSegmentation(
            root=datadir, year=year, image_set="train", download=False, transforms=tr_train
        )))
        ds_val_list.append(torchvision.datasets.wrap_dataset_for_transforms_v2(torchvision.datasets.VOCSegmentation(
            root=datadir, year=year, image_set="val", download=False, transforms=tr_val
        )))

    ds_train = torch.utils.data.ConcatDataset(ds_train_list)
    ds_val = torch.utils.data.ConcatDataset(ds_val_list)

    return ds_train, ds_val
