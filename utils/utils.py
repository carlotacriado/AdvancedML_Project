import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

from IPython.display import display, clear_output

#from torchinfo import summary

from utils.globals import *
from dataloader import *

# VISUALISATION AND LOGGERS
class TrainingMetrics:
    def __init__(self, meta=False):
        # initialise with empty lists for metric tracking:
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        self.step = 0
        self.epoch = 0
        self.epoch_steps = [0] # assume validation happens after every epoch

        # for plotting:
        self.fig = None
        self.plot_axes = None

        self.iter_steps = [] # steps at which training iteration changes

        self.meta = meta # if True, renames train/val loss etc.

    def log_train(self, loss:float, acc:float):
        """record batch-wise training metrics at each step (and increment step)"""
        assert isinstance(loss, float) and isinstance(acc, float), f"log_train expects float inputs but got: {type(loss)} and {type(acc)}"
        self.train_loss.append(loss)
        self.train_acc.append(acc)

        self.step += 1

    def log_val(self, loss:float, acc:float):
        """record total val metrics at end of epoch (and increment epoch)"""
        assert isinstance(loss, float) and isinstance(acc, float), f"log_val expects float inputs but got: {type(loss)} and {type(acc)}"
        self.val_loss.append(loss)
        self.val_acc.append(acc)

        if len(self.train_loss) == 0:
            self.step += 1 # increment step if no training data has been logged
        else:
            assert self.step not in self.epoch_steps, f"Validation metrics already logged for epoch {self.epoch} - you should submit only the aggregate metrics at end of epoch"""            
        self.epoch_steps.append(self.step)
        self.epoch += 1

    @property
    def epoch_loss(self):
        """report the training and validation loss over the previous epoch"""
        cur_epoch_step = self.epoch_steps[-1]
        last_epoch_step = self.epoch_steps[-2]
        epoch_train_loss = self.train_loss[last_epoch_step:cur_epoch_step]
        epoch_val_loss = self.val_loss[-1]
        return np.mean(epoch_train_loss), np.mean(epoch_val_loss)

    @property        
    def epoch_acc(self):
        """report the training and validation accuracy over the previous epoch"""
        cur_epoch_step = self.epoch_steps[-1]
        last_epoch_step = self.epoch_steps[-2]
        epoch_train_acc = self.train_acc[last_epoch_step:cur_epoch_step]
        epoch_val_acc = self.val_acc[-1]
        return np.mean(epoch_train_acc), np.mean(epoch_val_acc)

    @property
    def best_val_acc(self):
        return np.max(self.val_acc)
    
    def plot(self,
             title:str = None, # optional figure title
             baseline_accs: dict[str:float] = None, # optional dict of name->accuracy baseline benchmarks
             baseline_metrics = None, # optional metrics object for comparison curves
             alpha=0.1, # smoothing parameter for train loss
             live=True, # for animated plotting during training (slightly experimental)
             max_step=0, # should be set to the expected number of steps for animated plots
             ):
        if (self.fig is None) or (not live):
            self.fig, self.plot_axes = plt.subplots(1,2)
        loss_ax, acc_ax = self.plot_axes
        
        x_step = np.arange(0, self.step)

        # smooth out the per-step training numbers for neater curves:
        smooth_train_loss = pd.Series(self.train_loss).ewm(alpha=alpha).mean()
        smooth_train_acc = pd.Series(self.train_acc).ewm(alpha=alpha).mean()

        # plot train loss at every step:
        loss_ax.clear()
        if len(self.train_loss) > 0:
            loss_ax.plot(x_step, smooth_train_loss, c='tab:orange', linestyle=':', label='train loss' if not self.meta else 'Meta-train query loss')
        # val loss is plotted at every epoch:
        loss_ax.plot(self.epoch_steps[1:], self.val_loss, c='tab:orange', linestyle='-', label='val loss' if not self.meta else 'Meta-test query loss')


        ### plot acc:
        acc_ax.clear()
        if len(self.train_acc) > 0:
            acc_ax.plot(x_step, smooth_train_acc, c='tab:blue', linestyle=':', label='train acc' if not self.meta else 'Meta-train query acc')
        acc_ax.plot(self.epoch_steps[1:], self.val_acc, c='tab:blue', linestyle='-', label='val acc' if not self.meta else 'Meta-test query acc')
 
        
        if baseline_metrics is not None:
            # optionally show the training curves of a previous metric object as baseline
            # (by extracting the line data from the plots saved to that object)
            baseline_train_x = baseline_metrics.plot_axes[0].lines[0].get_xdata()
            baseline_train_loss = baseline_metrics.plot_axes[0].lines[0].get_ydata()
            baseline_val_x = baseline_metrics.plot_axes[0].lines[1].get_xdata()
            baseline_val_loss = baseline_metrics.plot_axes[0].lines[1].get_ydata()
            baseline_train_acc = baseline_metrics.plot_axes[1].lines[0].get_ydata()
            baseline_val_acc = baseline_metrics.plot_axes[1].lines[1].get_ydata()

            loss_ax.plot(baseline_train_x, baseline_train_loss, c='lightgray', linestyle=':', label=None, zorder=-1)
            loss_ax.plot(baseline_val_x, baseline_val_loss, c='lightgray', linestyle='-', label=None, zorder=-1)
            acc_ax.plot(baseline_train_x, baseline_train_acc, c='lightgray', linestyle=':', label=None, zorder=-1)
            acc_ax.plot(baseline_val_x, baseline_val_acc, c='lightgray', linestyle='-', label=None, zorder=-1)
            baseline_end_step = max(baseline_train_x)
        else:
            baseline_end_step = 0

        # format y-axis as percentage on right:
        acc_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
        acc_ax.yaxis.tick_right()
        acc_ax.yaxis.set_label_position('right')
    
        # draw line/s for comparison with baseline if given:
        if baseline_accs is not None:
            for b_name, b_acc in baseline_accs.items():
                acc_ax.axhline(b_acc, c=[0.8]*3, linestyle=':')
                # add text label as well:
                acc_ax.text(0, b_acc+0.003, b_name, c=[0.6]*3, size=8)

        # and to demarcate self-training iterations:
        if len(self.iter_steps) > 0:
            for iter_step in self.iter_steps:
                loss_ax.axvline(iter_step, c=[0.8]*3, linestyle='--')
                acc_ax.axvline(iter_step, c=[0.8]*3, linestyle='--')
                

        # add epoch tick markers and legend:
        end_step = max(max_step, self.step, baseline_end_step)
        
        num_epochs = len(self.epoch_steps)
        if num_epochs > 1:
            num_expected_epochs = end_step // self.epoch_steps[1]
            tick_spacing = max(num_expected_epochs // 10, 1)
        else:
            tick_spacing = 1
            num_expected_epochs = 1
        loss_ax.set_xticks(self.epoch_steps[::tick_spacing], range(0, num_epochs, tick_spacing))
        loss_ax.tick_params(axis='both', which='minor', labelsize=6)
        loss_ax.set_xlabel('Epoch'  if not self.meta else 'Meta-Evaluation Episode')
        loss_ax.set_ylabel('Loss')
        loss_ax.legend(); 
        
        # set x lims based on max number of steps
        loss_ax.set_xlim([0-(end_step*0.05), end_step*1.05])
        acc_ax.set_xlim([0-(end_step*0.05), end_step*1.05])
        acc_ax.set_xticks(self.epoch_steps[::tick_spacing], range(0, num_epochs, tick_spacing))
        acc_ax.tick_params(axis='both', which='minor', labelsize=6)
        
        acc_ax.set_xlabel('Epoch' if not self.meta else 'Meta-Evaluation Episode')
        acc_ax.set_ylabel('Accuracy (%)')
        acc_ax.legend()
        
        # and y lims based on the expectation that loss goes down and acc goes up
        loss_ax.set_ylim([-0.05, None])
        acc_ax.set_ylim([-0.05, 1.05])

        
        self.fig.suptitle(title)
        plt.tight_layout()

        if live:
            # redraw using ipython display
            display(self.fig)
            clear_output(wait=True)
        else:
            # just show plot normally
            plt.show()

### function to verify the contents and labels of a dataset object:
def inspect_dataset(dataset: Dataset,
                    layout=(4,4), # rows and cols of the grid to display
                    scale=0.7, 
                   ):
    """accepts a Dataset object
    and plots images from the data with optional class annotations"""
    num_examples = min([np.prod(layout), len(dataset)])
    example_idxs = np.random.choice(range(len(dataset)), num_examples, replace=False)

    if hasattr(dataset, 'data'):
        images = dataset.data
    else:
        images = dataset.tensors[0]

    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        targets = dataset.tensors[1]
    
    images = [images[idx] for idx in example_idxs]
    targets = [targets[idx] for idx in example_idxs]


    if hasattr(dataset, 'name'):
        title = f'Dataset: {dataset.name}'
    else:
        title = None
    
    a = 0
    num_rows, num_cols = layout
    fig_width = 2 * scale * num_cols
    fig_height = 2.3 * scale * num_rows + ((title is not None) * 0.3)
    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(fig_width, fig_height))
    
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            img = images[a]

            # if is tensor, cast to numpy:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            
            # if is CHW, reshape to HWC:
            if img.shape[0] == 3:
                img = img.transpose([1,2,0]) 

            # unnormalise:
            if img.min() < 0:
                img = (img - img.min()) / (img.max() - img.min())
            
            ax.imshow(img)
            
            label = f'{targets[a]}: {dataset.classes[targets[a]]}'
            ax.set_title(label, fontsize=10*scale**0.5)
            
            # tidy up axis:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            a += 1

    fig.suptitle(title)
    fig.tight_layout()
    
    plt.show()        


### function for inspecting the correctness of the outputs of a model:
def inspect_batch(images: torch.Tensor, # batch of images as torch tensors
        labels: torch.Tensor=None,      # optional vector of ground truth label integers
        predictions: torch.Tensor=None, # optional vector/matrix of model predictions
        # display parameters:
        class_names: list[str]=None, # optional list or dict of class idxs to class name strings
        title: str=None,       # optional title for entire plot    
        # figure display/sizing params:
        center_title=True,
        max_to_show=25,
        num_cols = 5,
        scale=0.7,
        ):
    """accepts a batch of images as a torch tensor or list of tensors,
    and plots them in a grid for manual inspection.
    optionally, you can supply ground truth labels
    and/or model predictions, to display those as well."""

    if type(images) is tuple:
        raise Exception('Expected first input as torch.Tensor, but got tuple; make sure to pass images and labels separately!')
    
    max_to_show = min([max_to_show, len(images)]) # cap at number of images
    
    num_rows = int(np.ceil(max_to_show / num_cols))

    # add extra figure height if needed for captions:
    extra_height = (((labels is not None)*0.3 + (predictions is not None))*0.3)

    fig_width = 2 * scale * num_cols
    fig_height = (2+extra_height) * scale * num_rows + ((title is not None) * 0.3)
    
    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(fig_width, fig_height))
    all_axes = []
    for ax_row in axes:
        all_axes.extend(ax_row)

    # translate labels and predictions to class names if given:
    if class_names is not None:
        if labels is not None:
            labels = [f'{l}:{class_names[int(l)]}' for l in labels]
        if predictions is not None:
            if len(predictions.shape) == 2:
                # probability distribution or onehot vector, so argmax it:
                predictions = predictions.argmax(dim=1)
            predictions = [f'{p}:{class_names[int(p)]}' for p in predictions]
    
    for b, ax in enumerate(all_axes):
        if b < max_to_show:
            # rearrange to H*W*C:
            img = images[b].permute([1,2,0]) 
            # un-normalise:
            img = (img - img.min()) / (img.max() - img.min())
            # to numpy:
            img = img.cpu().detach().numpy()
            
            ax.imshow(img)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            if labels is not None:                    
                ax.set_title(f'{labels[b]}', fontsize=10*scale**0.5)
            if predictions is not None:
                ax.set_title(f'pred: {predictions[b]}', fontsize=10*scale**0.5)
            if labels is not None and predictions is not None:
                if labels[b] == predictions[b]:
                    ### matching prediction, mark as correct:
                    mark, color = '✔', 'green'
                else:
                    mark, color = '✘', 'red'
                
                ax.set_title(f'label:{labels[b]}    \npred:{predictions[b]} {mark}', color=color, fontsize=8*scale**0.5)
        else:
            ax.axis('off')
    if title is not None:
        if center_title:
            x, align = 0.5, 'center'
        else:
            x, align = 0, 'left'
        fig.suptitle(title, fontsize=14*scale**0.5, x=x, horizontalalignment=align)
    fig.tight_layout()
    plt.show()


def count_labels(dataset):
    """Print out a count of the individual classes in a classification dataset.
    If there are more than 10, show only the first and last 10."""
    assert dataset.classes is not None
    if isinstance(dataset, TensorDataset):
        targets = dataset.tensors[1]
    else:
        targets = dataset.targets
    label_counts = np.unique(targets, return_counts=True)[1]
    

    num_classes = len(dataset.classes)
    max_len = max([len(cl) for cl in dataset.classes])
    if isinstance(dataset.classes, dict):
        idxs, names = dataset.classes.keys(), dataset.classes.values()
    elif isinstance(dataset.classes, list):
        idxs, names = range(num_classes), dataset.classes

    if num_classes > 10:
        # show only the first 10
        show_idxs = list(range(5)) + list(range(num_classes))[-5:]
    else:
        show_idxs = list(range(num_classes))

    print(f'\n{dataset.name}: {len(dataset):,} samples')
    for idx, name in zip(idxs, names):
        num_samples = label_counts[idx]
        if idx in show_idxs:
            print(f'  {idx}: {name:<{max_len}} ({num_samples:,} samples)')
        elif idx == 5:
            print('  ...')
            


### class prediction accuracy:
def top1_acc(pred: torch.FloatTensor, 
             y: torch.LongTensor):
    """calculates accuracy over a batch as a float
    given predicted logits 'pred' and integer targets 'y'"""
    return (pred.argmax(axis=1) == y).float().mean().item()  

def top5_acc(pred: torch.FloatTensor, 
             y: torch.LongTensor):
    """calculates top5 accuracy (whether any of the top 5 logits
    correspond to the true label), given predicted logits 'pred' 
    and integer targets 'y'"""
    top5_idxs = torch.sort(pred, dim=1, descending=True).indices[:,:5]
    correct = (top5_idxs == y.unsqueeze(1)).any(dim=1)
    avg_acc = correct.float().mean().item()
    return avg_acc    

# VISUALISATION

def visualize_episode(images, n_way, n_shot, n_query):
    """
    Visualizes a meta-learning episode as a grid.
    
    Args:
        images: Tensor of shape [Batch_Size, C, H, W]
        n_way: Number of classes (rows)
        n_shot: Support shots (left columns)
        n_query: Query shots (right columns)
    """
    # 1. Prepare dimensions
    images_per_class = n_shot + n_query
    _, c, h, w = images.shape
    
    # 2. Reshape: [Batch_Size, ...] -> [N_Way, K+Q, C, H, W]
    # We use .contiguous() to ensure memory is safe for .view()
    try:
        batch_grid = images.contiguous().view(n_way, images_per_class, c, h, w)
    except RuntimeError:
        print("Error: Batch size does not match n_way * (n_shot + n_query)")
        return

    rows_list = []

    # 3. Build the Grid
    for cls_idx in range(n_way):
        row_imgs = []
        for img_idx in range(images_per_class):
            
            # Extract single image tensor
            img = batch_grid[cls_idx, img_idx]
            
            # Convert: Tensor(C,H,W) -> Numpy(H,W,C)
            img = img.permute(1, 2, 0).cpu().numpy()
            
            # De-normalize if necessary (Assuming 0-1 range for now)
            # If your images look black, multiply by 255 here.
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Convert RGB (PyTorch) to BGR (OpenCV)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Add a thin black border around every image for clarity
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])
            
            row_imgs.append(img)

        # 4. Create the Support/Query Separator (A Red Vertical Line)
        separator = np.zeros((h + 2, 5, 3), dtype=np.uint8) 
        separator[:] = (0, 0, 255) # BGR Red

        # 5. Concatenate the row: [Support Images] + [Red Line] + [Query Images]
        support_part = cv2.hconcat(row_imgs[:n_shot])
        query_part   = cv2.hconcat(row_imgs[n_shot:])
        
        full_row = cv2.hconcat([support_part, separator, query_part])
        rows_list.append(full_row)

    # 6. Stack all rows vertically
    final_grid = cv2.vconcat(rows_list)

    # 7. Display
    cv2.imshow(f"Episode: {n_way}-Way (Rows), {n_shot}-Shot (Left) | Query (Right)", final_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_sample(dataset, idx):
    """
    Fetches a single sample from the dataset, converts it to OpenCV format,
    and displays it with metadata in the title.
    """
    # 1. Get the Tensor and Label Index
    # dataset[idx] triggers __getitem__, so transparency fix and transforms happen here
    img_tensor, label_idx = dataset[idx] 
    
    # 2. Retrieve Metadata using our new lookups
    name = dataset.idx_to_name[label_idx]
    dex = dataset.idx_to_dex[label_idx]
    
    # 3. Convert PyTorch Tensor (C, H, W) -> Numpy (H, W, C)
    # .permute(1, 2, 0) moves Channel from first to last
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # 4. Scale and Cast
    # PyTorch Tensors are usually 0.0-1.0 floats. OpenCV needs 0-255 uint8.
    img_np = (img_np * 255).astype(np.uint8)
    
    # 5. Convert Color Space (RGB -> BGR for OpenCV)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 6. Optional: Resize (Zoom in) 
    # 84x84 is tiny on screen, let's make it 4x bigger (336x336)
    img_cv2 = cv2.resize(img_cv2, (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    # 7. Create Window Title
    window_title = f"ID: {label_idx} | Dex: #{dex} | Name: {name}"
    
    print(f"Displaying -> {window_title}")
    
    cv2.imshow(window_title, img_cv2)
    cv2.waitKey(0) # Wait for any key press
    cv2.destroyAllWindows()