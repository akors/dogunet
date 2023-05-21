
import math
from typing import Iterable, List, Optional, Sequence
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils

import datasets

PASCAL_VOC_OBJECT_CLASS_MAX=20

def cm_to_tensor(cm: matplotlib.colors.ListedColormap):
    # turn listed colormap into a torch tensor
    cm_t = torch.tensor(cm.colors)[:,0:3]

    ## map class zero to black
    #cm_t = torch.cat((torch.Tensor([[0, 0, 0]]), cm_t), dim=0)

    return cm_t

# create custom 21-class colormap by prepending black. black will be used as background
cm_tab21 = mcolors.ListedColormap([(0.0, 0.0, 0.0, 1.0)] + [matplotlib.cm.tab20(i) for i in range(20)])
cm_tab21_t = cm_to_tensor(cm_tab21)


def imshow_tensor(img: torch.Tensor, ax: Optional[matplotlib.axes.Axes]=None):
    img = img.permute(1, 2, 0).numpy()
    img = img.clip(0.0, 1.0)
    if ax is None:
        ax = plt
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    return ax.imshow(img)

def imshow_mask_tensor(mask: torch.Tensor, ax: Optional[matplotlib.axes.Axes]=None):
    mask = mask.permute(0, 1).numpy()
    if ax is None:
        ax = plt
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    return ax.imshow(mask, cmap=cm_tab21, vmin=0, vmax=PASCAL_VOC_OBJECT_CLASS_MAX, interpolation_stage='rgba')


def plot_prediction_comparison(
    img: torch.Tensor,
    prediction_mask: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    fig=None
):
    fig = fig if fig else plt.gcf()

    axs = fig.subplots(1, 2 + int(target_mask is not None), gridspec_kw={'wspace': 0.05})

    imshow_tensor(img, ax=axs[0])
    axs[0].set_title("Input image")

    imshow_mask_tensor(prediction_mask, ax=axs[1])
    axs[1].set_title("Predicted mask")

    if target_mask is not None:
        imshow_mask_tensor(target_mask, ax=axs[2])
        axs[2].set_title("Target mask")

    # Remove the ticks on the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    return fig, axs


def classmask_to_colormask(mask: torch.Tensor, cm: torch.Tensor = cm_tab21_t) -> torch.Tensor:
    assert mask.dim() == 3, "Mask should be a 3D tensor with B x H x W"

    cm=cm.to(device=mask.device)

    # create output tensor
    mask_colormapped = torch.zeros((mask.shape[0], 3, *mask.shape[-2:]), device=mask.device, dtype=cm.dtype)

    # loop over all classes in mask
    for k in mask.to(dtype=torch.int64).unique():
        # get indices for all pixels in batch where the class equals k
        pixels_in_class_idx = (mask == k).argwhere()

        # insert current class color into dim 1 for all index tuples
        mask_colormapped[pixels_in_class_idx[:, 0], :, pixels_in_class_idx[:, 1], pixels_in_class_idx[:, 2]] = \
            cm[k, :].unsqueeze(0)

    return mask_colormapped


def make_colormap_legend(ax, fig, cmap, class_names: List[str], only_classes: Optional[List[int]]=None):
    if only_classes is None:
        only_classes = range(len(class_names))

    # turn any iterable into list, because we want random access
    if not isinstance(class_names, List) and isinstance(class_names, Iterable):
        class_names = [n for n in class_names]

    # Create a dictionary with the class names and their corresponding color from the colormap
    legend_dict = dict(zip(class_names, cmap(np.linspace(0, 1, len(class_names)))))
    legend_dict = {class_names[c] : cmap(c) for c in only_classes}

    for class_idx in only_classes:
        class_name = class_names[class_idx]
        ax.plot([], [], color=legend_dict[class_name], linewidth=10, label=class_name)
    
    fig.legend(loc='lower center', ncol=5)

def plot_colormap_legend(cmap, class_names: List[str], only_classes: Optional[List[int]]=None):
    if only_classes is None:
        only_classes = range(len(class_names))

    # turn any iterable into list, because we want random access
    if not isinstance(class_names, List) and isinstance(class_names, Iterable):
        class_names = [n for n in class_names]

    # Create a dictionary with the class names and their corresponding color from the colormap
    legend_dict = dict(zip(class_names, cmap(np.linspace(0, 1, len(class_names)))))
    legend_dict = {class_names[c] : cmap(c) for c in only_classes}

    # Create a legend with the class names and their corresponding color
    fig, ax = plt.subplots(figsize=(3,3))
    for class_idx in only_classes:
        class_name = class_names[class_idx]
        ax.plot([], [], color=legend_dict[class_name], linewidth=10, label=class_name)

    # Remove the x and y axis tick marks and labels
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add the legend to the plot
    ax.legend(loc='center', ncol=2, title='Classes')

    plt.show()

def make_comparison_grid(img: torch.Tensor, prediction: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    assert img.size(-2) == prediction.size(-2), "img, prediction must have same height"
    assert img.size(-1) == prediction.size(-1), "img, prediction must have same width"
    assert img.size(0) == prediction.size(0), "img, prediction must have same batch size"

    if mask is not None:
        assert img.size(-2) == mask.size(-2), "img, mask must have same height"
        assert img.size(-1) == mask.size(-1), "img, mask must have same width"
        assert img.size(0) == mask.size(0), "img, mask must have same batch size"

    cm = cm_tab21_t.to(prediction.device)

    # image is expected to have floats for color values between 0.0 and 1.0
    img = img.clamp(min=0.0, max=1.0)

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


def plot_img_classmasked(img: torch.Tensor, pred: torch.Tensor, fig=None, limit_classes=[], show_orig=True):
    """Plot image masked by output probabilities for each class.

    This gives an idea about which parts of the image the network sees as which class.

    Parameters
    ----------
    img : torch.Tensor
        Input image. 3D-Tensor [C,H,W] (no batch dim!). This should already be denormalized.
    pred : torch.Tensor
        Output prediction (sigmoid). 3D-Tensor (no batch dim!), [N,H,W].
    fig : matplotlib.figure.Figure, optional
        Figure to which to add the plots, by default None
    limit_classes : List[int], optional
        Only show masked image for the classes within this list. If empty, show all classes. by default []
    show_orig : bool
        Show original image in first subplot. by default True

    Returns
    -------
    matplotlib.figure.Figure
        Output figure
    np.array[matplotlib.axes.Axes]
        Axes containing plots
    """
    fig =  plt.gcf() if fig is None else fig

    with torch.no_grad():
        pred_amax = torch.argmax(pred, dim=1)
        pred_unique_classes = pred_amax.unique()

    denormalized_img = img

    ncols = min(3, len(pred_unique_classes) + bool(show_orig))
    nrows = math.ceil((len(pred_unique_classes) + bool(show_orig)) / ncols)
    axs = fig.subplots(nrows=nrows, ncols=ncols)
    axs_raveled = np.ravel(axs)
    if show_orig:
        ax = axs_raveled[0]
        ax.set_title("Original")
        imshow_tensor(denormalized_img, ax=ax)
    
    # disable xticks, yticks, and axis. The axis will be re-enabled for subplots with content in them.
    for ax in axs_raveled:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    unique_class_gen = (u.item() for u in pred_unique_classes if u in limit_classes or len(limit_classes)==0)
    for idx, unique_class in enumerate(unique_class_gen, start=bool(show_orig)):
        ax = axs_raveled[idx]

        # We will abuse the sigmoid model output of this class as the alpha channel
        # probs are between 0 and 1, alpha should be between 0 and 255
        channelmask =  pred[0,unique_class,:,:].unsqueeze(0)# * 256.0

        # stack RGB data with output probablity as alpha
        masked_img = torch.cat(dim=0, tensors=(denormalized_img, channelmask))

        # numpy the torch
        masked_img = masked_img.permute(1, 2, 0).numpy()

        ax.imshow(masked_img)
        ax.set_title(datasets.CLASSNAMES[unique_class])
        ax.axis('on')

    
    fig.subplots_adjust(wspace=0, hspace=.2)

    return fig, axs


def plot_class_heatmap(img, pred, fig=None, limit_classes=[], show_orig=True):
    """Plot heatmaps for predicted class probabilities

    This gives an idea about which parts of the image the network sees as which class.

    Parameters
    ----------
    img : torch.Tensor
        Input image. 3D-Tensor [C,H,W] (no batch dim!). This should already be denormalized.
    pred : torch.Tensor
        Output prediction (sigmoid). 3D-Tensor (no batch dim!), [N,H,W].
    fig : matplotlib.figure.Figure, optional
        Figure to which to add the plots, by default None
    limit_classes : List[int], optional
        Only show masked image for the classes within this list. If empty, show all classes. by default []
    show_orig : bool
        Show original image in first subplot. by default True

    Returns
    -------
    matplotlib.figure.Figure
        Output figure
    np.array[matplotlib.axes.Axes]
        Axes containing plots
    """
    fig =  plt.gcf() if fig is None else fig

    with torch.no_grad():
        pred_amax = torch.argmax(pred, dim=0)
        pred_unique_classes = pred_amax.unique()

    ncols = min(3, len(pred_unique_classes) + bool(show_orig))
    nrows = math.ceil((len(pred_unique_classes) + bool(show_orig)) / ncols)
    axs = fig.subplots(nrows=nrows, ncols=ncols)
    axs_raveled = np.ravel(axs)
    if show_orig:
        ax = axs_raveled[0]
        ax.set_title("Original")
        imshow_tensor(img, ax=ax)

    unique_class_gen = (u.item() for u in pred_unique_classes if u in limit_classes or len(limit_classes)==0)
    for idx, unique_class in enumerate(unique_class_gen, start=bool(show_orig)):
        ax = axs_raveled[idx]
        ax.imshow(pred[unique_class,:,:].numpy(), vmin=0.0, vmax=1.0, cmap='inferno')

        ax.set_title(datasets.CLASSNAMES[unique_class])

    for ax in axs_raveled:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    fig.subplots_adjust(wspace=0.1, hspace=0)

    return fig, axs
