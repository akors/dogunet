#!/bin/env python3

import os
import sys

import torchvision.datasets

dataroot = './data/'
if len(sys.argv) > 1:
    dataroot = sys.argv[1]


#download_ds = not os.path.exists(os.path.join(datapath, 'VOCdevkit'))
ds_train = torchvision.datasets.VOCSegmentation(
        root=dataroot, year="2012", image_set="trainval", download=True
    )
