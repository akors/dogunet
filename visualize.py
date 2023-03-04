
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import torch

def imshow_tensor(img: torch.Tensor, ax: Optional[matplotlib.axes.Axes]=None):
    img = img.permute(1, 2, 0).numpy()
    if ax is None:
        ax = plt
    return ax.imshow(img)

def imshow_mask_tensor(mask: torch.Tensor, ax: Optional[matplotlib.axes.Axes]=None):
    mask = mask.permute(0, 1).numpy()
    if ax is None:
        ax = plt
    return ax.imshow(mask, cmap='tab20', interpolation_stage='rgba')

def plot_prediction_comparison(img: torch.Tensor, prediction_mask: torch.Tensor, target_mask: torch.Tensor):
    fig, axs = plt.subplots(1,3, gridspec_kw={'wspace': 0.05})

    imshow_tensor(img, ax=axs[0])
    axs[0].set_title("Input image")
    imshow_mask_tensor(prediction_mask, ax=axs[1])
    axs[1].set_title("Predicted mask")
    imshow_mask_tensor(target_mask, ax=axs[2])
    axs[2].set_title("Target mask")


    # Remove the ticks on the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    return fig
