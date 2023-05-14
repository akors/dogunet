

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

# These stats were calculated using the calculate_dataset_stats method below
DATASET_STATS = {
    "2012": {
        "sample_count": 2913,
        "rgb_mean": [
            0.4568465189887431,
            0.44091867824707537,
            0.4047057680605658
        ],
        "rgb_std": [
            0.2712492735621932,
            0.2684503630844764,
            0.28438828894633245
        ],
        "height_mean": 384.6728458633711,
        "height_std": 63.5685314839425,
        "width_mean": 471.9289392378991,
        "width_std": 57.45974064982267,
        "class_pixels": {
            0: 361560627,
            1: 3704393,
            2: 1571148,
            3: 4384132,
            4: 2862913,
            5: 3438963,
            6: 8696374,
            7: 7088203,
            8: 12473466,
            9: 4975284,
            10: 5027769,
            11: 6246382,
            12: 9379340,
            13: 4925676,
            14: 5476081,
            15: 24995476,
            16: 2904902,
            17: 4187268,
            18: 7091464,
            19: 7903243,
            20: 4120989,
            255: 28568409
        },
        "pixel_count": 521582502
    }
}

def make_datasets(datadir: str="./data/", years=["2012"], augment_level: int=0):
    assert len(years) > 0, "Need to select at least one year between 2007 and 2012"
    ds_train_list = list()
    ds_val_list = list()

    for year in years:
        tr_train = transforms.make_transforms(
            DATASET_STATS["2012"]['rgb_mean'], DATASET_STATS["2012"]['rgb_std'], augment_level=augment_level)
        
        # validation transforms dont get data augmentation
        tr_val = transforms.make_transforms(
            DATASET_STATS["2012"]['rgb_mean'], DATASET_STATS["2012"]['rgb_std'], augment_level=0)


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
