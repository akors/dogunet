
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.utils

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

def cm_to_tensor(cm: matplotlib.colors.ListedColormap):
    # turn listed colormap into a torch tensor
    cm_t = torch.tensor(cm.colors)

    # map class zero to black
    cm_t = torch.cat((torch.Tensor([[0, 0, 0]]), cm_t), dim=0)

    return cm_t

cm_tab20_t = cm_to_tensor(matplotlib.cm.tab20)

def make_comparison_grid(img: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_colormapped = torch.zeros((3, *mask.shape[-2:]), device=mask.device)
    for k in mask.to(dtype=torch.int64).unique():
        pixels_in_class_idx = (mask == k)
        mask_colormapped[:, pixels_in_class_idx] = cm_tab20_t.to(device=mask.device)[k, :].unsqueeze(-1)
        
    pred_colormapped = torch.zeros((3, *prediction.shape[-2:])).to(prediction.device)
    for k in prediction.to(dtype=torch.int64).unique():
        pixels_in_class_idx = (prediction == k)
        pred_colormapped[:, pixels_in_class_idx] = cm_tab20_t.to(device=mask.device)[k, :].unsqueeze(-1)

    comparison_fig_t = torchvision.utils.make_grid([img, pred_colormapped, mask_colormapped])
    return comparison_fig_t
