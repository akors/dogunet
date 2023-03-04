
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


input_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] # it was revealed to me in a dream
    ),
    torchvision.transforms.ToPILImage(),
    Resize_with_pad(256,256),
    torchvision.transforms.ToTensor()
])

target_transform = torchvision.transforms.Compose([
    Resize_with_pad(256,256, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
    torchvision.transforms.PILToTensor(),
])
