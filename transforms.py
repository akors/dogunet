
from typing import List
import typing
import torch
import torchvision.datapoints
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F


#https://github.com/pytorch/vision/issues/6236#issuecomment-1175971587
class Resize_with_pad:
    def __init__(self, w=256, h=256, interpolation=T.InterpolationMode.BILINEAR, antialias=True):
        self.w = w
        self.h = h

        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, sample):
        if isinstance(sample, typing.Sequence):
            # apply to all elements if element is a sequence
            return tuple(self.do(i) for i in sample)
        else:
            return self.do(sample)

    def do(self, image):
        if isinstance(image, torch.Tensor):
            h_1, w_1 = image.shape[-2:]
        else:
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
                return F.resize(image, [self.h, self.w], interpolation=self.interpolation, antialias=self.antialias)

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w], interpolation=self.interpolation, antialias=self.antialias)

        else:
            return F.resize(image, [self.h, self.w], interpolation=self.interpolation, antialias=self.antialias)

# it was revealed to me in a dream
PASCAL_VOC_2012_MEAN=[0.485, 0.456, 0.406]
PASCAL_VOC_2012_STD=[0.229, 0.224, 0.225]
PASCAL_VOC_2012_CLASS_MAX=20


class ClipMaskClasses():
    def __init__(self, max_class: int) -> None:
        self.max_class = max_class

    def __call__(self, sample):
        if isinstance(sample, typing.Sequence):
            # apply to all elements if element is a sequence
            return tuple(self.__call__(i) for i in sample)
        elif isinstance(sample, torchvision.datapoints.Mask):
            return self.do(sample)
        else:
            return sample

    def do(self, x):
        x = torchvision.datapoints.Mask(torch.where(x <= 20, x, 0))
        return x

def make_transforms(mean, std, augment_level=0):
    # apply anti-aliasing for resize operations, this will be skipped automagically for masks of type
    # torchvision.datapoints.Mask
    antialias = True

    oplist = []
    
    oplist.append(T.ToImageTensor())

    if not augment_level:
        oplist.append(T.Resize(size=256, antialias=antialias))
        oplist.append(T.CenterCrop(256))
    
    if augment_level >= 1:
        oplist.append(T.RandomResizedCrop(size=256, scale=(0.3, 1.0), ratio=(1,1), antialias=antialias))
        oplist.append(T.RandomHorizontalFlip())
    if augment_level >= 2:
        oplist.append(T.GaussianBlur(kernel_size=[5, 5], sigma=2))
    
    assert augment_level <= 2, "Augmentation level "+str(augment_level)+"?? What is this, the future??"

    oplist.append(T.ConvertImageDtype(torch.float32))
    oplist.append(T.Normalize(mean=mean, std=std))
    oplist.append(ClipMaskClasses(PASCAL_VOC_2012_CLASS_MAX))

    return T.Compose(oplist)

def inv_normalize(mean, std):
    return T.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                                std=[1 / s for s in std])
