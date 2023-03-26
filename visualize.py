
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.utils

def imshow_tensor(img: torch.Tensor, ax: Optional[matplotlib.axes.Axes]=None):
    img = img.permute(1, 2, 0).numpy()
    if ax is None:
        ax = plt
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    return ax.imshow(img)

def imshow_mask_tensor(mask: torch.Tensor, ax: Optional[matplotlib.axes.Axes]=None):
    mask = mask.permute(0, 1).numpy()
    if ax is None:
        ax = plt
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
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

def classmask_to_colormask(mask: torch.Tensor, cm: torch.Tensor = cm_tab20_t) -> torch.Tensor:
    assert mask.dim() == 3, "Mask should be a 3D tensor with B x H x W"

    cm=cm.to(device=mask.device)

    # create output tensor
    mask_colormapped = torch.zeros((mask.shape[0], 3, *mask.shape[-2:]), device=mask.device)

    # loop over all classes in mask
    for k in mask.to(dtype=torch.int64).unique():
        # get indices for all pixels in batch where the class equals k
        pixels_in_class_idx = (mask == k).argwhere()

        # insert current class color into dim 1 for all index tuples
        mask_colormapped[pixels_in_class_idx[:, 0], :, pixels_in_class_idx[:, 1], pixels_in_class_idx[:, 2]] = \
            cm[k, :].unsqueeze(0)

    return mask_colormapped

def make_comparison_grid(img: torch.Tensor, prediction: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    assert img.size(-2) == prediction.size(-2), "img, prediction must have same height"
    assert img.size(-1) == prediction.size(-1), "img, prediction must have same width"
    assert img.size(0) == prediction.size(0), "img, prediction must have same batch size"

    if mask is not None:
        assert img.size(-2) == mask.size(-2), "img, mask must have same height"
        assert img.size(-1) == mask.size(-1), "img, mask must have same width"
        assert img.size(0) == mask.size(0), "img, mask must have same batch size"

    cm = cm_tab20_t.to(prediction.device)

    pred_colormapped = classmask_to_colormask(mask=prediction, cm=cm)

    imglist = [img, pred_colormapped]
    if mask is not None:
        mask_colormapped = classmask_to_colormask(mask=mask, cm=cm)
        imglist.append(mask_colormapped)
        
    # stack up images
    grid = torch.stack(imglist, dim=1)
    grid = grid.view(img.shape[0]*len(imglist), cm.shape[1], img.shape[-2], img.shape[-1])

    comparison_fig_t = torchvision.utils.make_grid(grid, nrow=len(imglist))
    return comparison_fig_t
