

from typing import List
import warnings

import numpy as np

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
  "2007": {
    "sample_count": 209,
    "rgb_mean": [
      0.4461092056269851,
      0.4256838929068529,
      0.3904880787196912
    ],
    "rgb_std": [
      0.2731654057337533,
      0.26910677163127916,
      0.2797351260021947
    ]
  },
  "2008": {
    "sample_count": 511,
    "rgb_mean": [
      0.45463992256503516,
      0.4416930567320079,
      0.4101860570318545
    ],
    "rgb_std": [
      0.2724504041774264,
      0.26871104551810937,
      0.2841974550808964
    ]
  },
  "2009": {
    "sample_count": 749,
    "rgb_mean": [
      0.4552125069990416,
      0.4437961747275494,
      0.4090016108274818
    ],
    "rgb_std": [
      0.272089602133736,
      0.26797705280642703,
      0.2844254158700665
    ]
  },
  "2010": {
    "sample_count": 964,
    "rgb_mean": [
      0.45688186447170887,
      0.44544289744926685,
      0.41054491990311576
    ],
    "rgb_std": [
      0.2718114269124274,
      0.2685014066503421,
      0.28477131628994906
    ]
  },
  "2011": {
    "sample_count": 1112,
    "rgb_mean": [
      0.45378828333797966,
      0.44159090210743945,
      0.4076431971428608
    ],
    "rgb_std": [
      0.27165483480651303,
      0.2680921065344953,
      0.2841974935029781
    ]
  },
  "2012": {
    "sample_count": 1464,
    "rgb_mean": [
      0.456797805118573,
      0.4431319283728635,
      0.4082984168812775
    ],
    "rgb_std": [
      0.27287834297763963,
      0.2693239723248675,
      0.28497994197739135
    ]
  },
  "2007+2008+2009+2010+2011+2012": {
    "sample_count": 5009,
    "rgb_mean": [
      0.45524269659916283,
      0.44245909361169117,
      0.40813989028331643
    ],
    "rgb_std": [
      0.272263535943418,
      0.2686478121215398,
      0.2844137407538725
    ]
  }
}


def make_datasets(datadir: str="./data/", years=["2012"], augment_level: int=0):
    assert len(years) > 0, "Need to select at least one year between 2007 and 2012"
    ds_train_list = list()
    ds_val_list = list()

    for year in years:
        tr_train = transforms.make_transforms(*get_dataset_mean_std(years), augment_level=augment_level)
        
        # validation transforms dont get data augmentation
        tr_val = transforms.make_transforms(*get_dataset_mean_std(years), augment_level=0)


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


def calculate_dataset_stats(
    datadir: str = "./data/",
    years: List[str] = None,
    split="train",
    num_workers=4,
    progressbar=None,
    combined=True
):
    years = years or ["2012"]

    # if not set, progress bar object is passthrough
    if progressbar is None:
        progressbar = lambda *args: args[0]

    loader_list = list()


    # we will continue updating with samples from each year
    firstpass_combined = metrics.DatasetMetricsFirstpass() if combined else None

    result_years = dict()
    for year in years:
        # open dataset for which to calculate stats
        ds_check = torchvision.datasets.VOCSegmentation(
            root=datadir,
            year=year,
            image_set=split,
            download=False,
            transform=T.ToTensor(), target_transform=T.PILToTensor()
        )

        # create data loader
        loader = torch.utils.data.DataLoader(ds_check, batch_size=1, shuffle=False, num_workers=num_workers)
        loader_list.append(loader)

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
        result.update(secondpass.calculate())

        # store in year result dictionary
        result_years[year] = result

        # append this years dataset with combined dataset firstpass stats
        if firstpass_combined: firstpass_combined.concatenate_with(firstpass)

    if combined:
        combined_ds_name = "+".join(sorted(years))

        # after completing first pass for combined dataset, we have to run the second pass again
        result_combined = firstpass_combined.calculate()
        secondpass_combined = metrics.DatasetMetricsSecondpass(result_combined)

        sample_idx = 0
        for loader in loader_list:
            for sample in progressbar(loader, desc=combined_ds_name+" Second Pass"):
                secondpass_combined.update(sample[0][0,:], sample[1][0,:], sample_idx)

                sample_idx += 1

        # store secondpass results in firstpass results
        result_combined.update(secondpass_combined.calculate())

        # add combined dasaet into result dict
        result_years[combined_ds_name] = result_combined

    return result_years

def get_dataset_mean_std(years=["2012"]):
    combinedname = "+".join(sorted(years))

    # check if combined stats are available
    if combinedname in DATASET_STATS:
        return DATASET_STATS[combinedname]['rgb_mean'], DATASET_STATS[combinedname]['rgb_std']

    ds_stats = dict()
    if len(years) > 1:
        warnings.warn("Precomputed stats for year " + combinedname + " are no available. "+\
            "Resorting to weighted avarage over the years.")

        rgb_means = np.empty(shape=(len(years), 3))
        rgb_stds = np.empty(shape=(len(years), 3))
        numsamples = np.empty(shape=(len(years)))

        for idx, year in enumerate(years):
            rgb_means[idx, :] = np.array(DATASET_STATS[year]['rgb_mean'])
            rgb_stds[idx, :] = np.array(DATASET_STATS[year]['rgb_std'])
            numsamples[idx] = DATASET_STATS[year]['sample_count']

        avg_weights = numsamples / np.sum(numsamples)

        rgb_mean = np.sum(np.expand_dims(avg_weights, axis=-1) * rgb_means, axis=0)
        rgb_std = np.sum(np.expand_dims(avg_weights, axis=-1) * rgb_stds, axis=0)

        return rgb_mean, rgb_std
    else:
        KeyError("Precomputed stats for year " + combinedname + " are no available.")
