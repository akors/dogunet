

import torch
import torchvision
import torchvision.transforms as T

import transforms
import metrics

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
        transforms.PASCAL_VOC_2012_MEAN, transforms.PASCAL_VOC_2012_STD, augment_level=augment_level)
    
    # validation transforms dont get data augmentation
    tr_val = transforms.make_transforms(
        transforms.PASCAL_VOC_2012_MEAN, transforms.PASCAL_VOC_2012_STD, augment_level=0)

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


def calculate_dataset_stats(datadir: str="./data/", year="2012", split="trainval", num_workers=4, progressbar=None):
    # open dataset for which to calculate stats
    ds_check = torchvision.datasets.VOCSegmentation(
        root=datadir,
        year=year,
        image_set=split,
        download=False,
        transform=T.ToTensor(), target_transform=T.PILToTensor()
    )

    # if not set, progress bar object is passthrough
    if progressbar is None:
        progressbar = lambda *args: args[0]

    # create data loader
    loader = torch.utils.data.DataLoader(ds_check, batch_size=1, shuffle=False, num_workers=num_workers)

    # first pass metrics
    firstpass = metrics.DatasetMetricsFirstpass()
    for sample_idx, sample in enumerate(progressbar(loader, desc=year + " First Pass")):
        firstpass.update(sample[0][0,:], sample[1][0,:])

    # finalize firstpass metrics
    result = firstpass.calculate()

    # calculate second pass
    secondpass = metrics.DatasetMetricsSecondpass(result)
    for sample_idx, sample in enumerate(progressbar(loader, desc=year + " Second Pass")):
        secondpass.update(sample[0][0,:], sample[1][0,:], sample_idx)

    # finalize second pass results and append to first pass
    result.update(
        secondpass.calculate()
    )

    return result
