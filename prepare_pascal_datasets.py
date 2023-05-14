#!/bin/env python3

import json
import argparse
from tqdm import tqdm

import torchvision.datasets

import datasets


def do_prepare(dataroot, years, download, suite):
    if download:
        for year in years:
            # open temporarily to download the datasets
            ds = torchvision.datasets.VOCSegmentation(root=dataroot, year=year, image_set=suite, download=download)
            del ds

    stats = dict()
    for year in years:
        result = datasets.calculate_dataset_stats(datadir=dataroot, year=year, split=suite, progressbar=tqdm)
        stats[year] = result

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dogunet datasets')

    parser.add_argument('years', nargs='+', default=["2012"], metavar="YEAR",
        choices=("2007", "2009", "2010", "2011","2012"),
        help="For which years to process the PASCAL VOC datasets")
    parser.add_argument('-d', '--download',  action="store_true", help="If the dataset is not present, download it")
    parser.add_argument('--dataroot', default="./data/", help="Root of the data directory. Default is ./data")
    parser.add_argument('--suite', default="trainval", choices=("train", "val", "trainval"),
        help="Which image set to process. Default is trainval.")

    args = parser.parse_args()
    stats = do_prepare(dataroot=args.dataroot, years=args.years, download=args.download, suite=args.suite)

    print(json.dumps(stats, indent=4))
