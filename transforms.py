
from typing import List
import torch
import torchvision.transforms
import torchvision.transforms.functional as F


#https://github.com/pytorch/vision/issues/6236#issuecomment-1175971587
class Resize_with_pad:
    def __init__(self, w=1024, h=768, interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
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

# it was revealed to me in a dream
PASCAL_VOC_2012_MEAN=[0.485, 0.456, 0.406]
PASCAL_VOC_2012_STD=[0.229, 0.224, 0.225]
PASCAL_VOC_2012_CLASS_MAX=20

def input_transform(mean: List[float], std: List[float]):
    return torchvision.transforms.Compose([
        Resize_with_pad(256,256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=mean, std=std
        ),
        #torchvision.transforms.ToPILImage(),
        #torchvision.transforms.ToTensor()
    ])

class ClipMaskClasses():
    def __init__(self, max_class: int) -> None:
        self.max_class = max_class
    
    def __call__(self, x):
        x = torch.where(x <= self.max_class, x, 0).to(dtype=torch.long)
        return x

def inv_normalize(mean, std):
    return torchvision.transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                                std=[1 / s for s in std])

def target_transform(max_class: int):
    return torchvision.transforms.Compose([
        Resize_with_pad(256,256, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.Lambda(lambda x: x[0,:,:].to(dtype=torch.int64)), # remove dim1, convert to index type
        ClipMaskClasses(max_class=max_class)
    ])
