import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

from IPython.display import display, clear_output
#from torchinfo import summary

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils import utils
from utils import globals


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# THE PARTY BEGINS
np.random.seed(0)
example_task = utils.create_task(num_support=3, num_query=3)

example_task.inspect(summarise=False)

### META-TEST SET generation:

# for our meta test set, we generate 15 different few-shot tasks
# from the classes in our 'meta test classes' set.
# note that none of these are available during training!

NUM_TEST_TASKS = 15  # number of random meta-test tasks to create

np.random.seed(1)
meta_test_tasks = [utils.create_task(num_support = globals.SUPPORT_SIZE,    # only 6 examples per class!
                               num_query = globals.QUERY_SIZE,
                               task_name = f'Meta-test task #{t}',
                               meta_test=True) 
                   for t in range(NUM_TEST_TASKS)]


meta_test_task_classes = []
for t, task in enumerate(meta_test_tasks):
    task.inspect(summarise=True) # this time, they are single-row summaries
    meta_test_task_classes.append(tuple([str(c) for c in task.classes]))
    
max_episodes = 1000          # Increased for more convergence time
meta_evaluate_interval = 50

epsilon = 1e-1 #the value that tells us how much of the finetune model affect the metamodel

# initialise your metamodel:
meta_model = ConvBackbone() 

# Outer loop optimizer (RepTile update)
meta_lr = 1e-1 # HIGH meta-learning rate
meta_opt = torch.optim.Adam(meta_model.parameters(), lr=meta_lr) 
# and use this to track its metrics on the query sets across meta-training:
meta_metrics = TrainingMetrics(meta=True) 

# iterate across meta-training episodes:
for m in tqdm(range(max_episodes)):

    # A. Sample a task from the task distribution.
    task = create_task(num_support = SUPPORT_SIZE, num_query = QUERY_SIZE)

    # B. Fine-tune the model on this task. 
    
    # Save the original weights (theta)
    theta = {name: param.clone().detach() for name, param in meta_model.named_parameters()}
    
    # Initialize a task-specific backbone from the meta-model's current state
    task_backbone = ConvBackbone()
    task_backbone.load_state_dict(meta_model.state_dict())
    
    # Perform inner-loop fine-tuning (this updates task_backbone to theta')
    task_metrics, task_head = finetune_backbone_on_task(task_backbone, task, 
                                                       progress_bar=False, 
                                                       plot=False, 
                                                       lr=FT_LR, 
                                                       l2_reg=FT_L2)
    
    # C. Update the meta-model (theta) towards the fine-tuned model (theta')
    meta_opt.zero_grad()
    
    # Calculate the pseudo-gradient (theta - theta') for each parameter
    with torch.no_grad():
        for name, param in meta_model.named_parameters():
            # Get theta' (the fine-tuned weights)
            theta_prime = task_backbone.state_dict()[name]
            # Calculate pseudo-gradient: (theta - theta')
            pseudo_grad = epsilon * (theta[name] - theta_prime) 
            # Apply the pseudo-gradient to the meta-model's parameter
            param.grad = pseudo_grad 

    # Apply the meta-model's outer loop update
    meta_opt.step()

    # Log the performance on the query set of the meta-TRAIN task:
    meta_metrics.log_train(loss = task_metrics.val_loss[-1],   # loss over the query set of the meta-TRAIN tasks
                           acc = task_metrics.val_acc[-1])     # accuracy over the query set of the meta-TRAIN tasks
    
    # meta-evaluate occasionally:
    if (m % meta_evaluate_interval == 0):
        print(f'Meta-training episode {m}/{max_episodes}')
        meta_eval_results = meta_evaluate(meta_model, # backbone object to evaluate
                                          model_name = f'Reptile meta-model (episode {m})',
                                          inspect_tasks = False, # visualise each meta-test task
                                          progress_bars = False, # show progress bars for fine-tuning each meta-test task
                                          show_taskwise_accuracy = True, # show plot with query accuracy on each meta-test task
                                          baseline_avg = baseline_metatest_query_acc # show the performance of our baseline model on that plot
                                          )

        # unpack the meta-evaluation metrics:
        test_support_loss, test_support_acc, test_query_loss, test_query_acc = meta_eval_results
        
        # but support performance doesn't matter, our final target is the test query performance:
        meta_metrics.log_val(loss = test_query_loss,
                              acc = test_query_acc)

        # show meta-training plot across episodes:
        meta_metrics.plot(
            title=f'Meta-training (Reptile)',
            baseline_accs={'baseline': baseline_metatest_query_acc},
            live=False)
