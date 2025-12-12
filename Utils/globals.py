from torchvision import transforms
import torch.nn as nn

BATCH_SIZE = 64 # though in practice, datasets are so small that each epoch is one batch
SUPPORT_SIZE = 5     # num examples per class in each support set
QUERY_SIZE = 1       # num examples per class in each query set

# --- HYPERPARAMETERS ---
N_WAY = 5
N_SHOT = 5     # Support Size (K)
N_QUERY = 1   # Query Size (Q)

MAX_EPOCHS = 100       # Total training duration
EPISODES_PER_EPOCH = 100 # How many episodes in one loader iteration

INNER_LR = 0.01        # Learning rate for the clone
INNER_STEPS = 5        # How many gradient steps the clone takes
EPSILON = 0.1          # Reptile outer learning rate

VAL_SPLIT = 0.2

SEED = 151             # Seed to mantain the same Train set (split) in all models

SUPPORT_AUGMENTATIONS = nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
)

EVAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((84, 84)), 
        transforms.ToTensor()
    ])