from torchvision import transforms
import torch.nn as nn

BATCH_SIZE = 64 
SUPPORT_SIZE = 5     # num examples per class in each support set
QUERY_SIZE = 1       # num examples per class in each query set

# --- HYPERPARAMETERS ---
N_WAY = 5
N_SHOT = 5     # Support Size (K)
N_QUERY = 1   # Query Size (Q)

MAX_EPOCHS = 200       # Total training duration
EPISODES_PER_EPOCH = 100 # How many episodes in one loader iteration

# REPTILE SPECIFIC HYPERPARAMETERS
INNER_LR = 0.0005        # Learning rate for the clone
INNER_STEPS = 12        # How many gradient steps the clone takes
EPSILON = 0.1          # Reptile outer learning rate

VAL_SPLIT = 0.2

SEED = 151             # Seed to mantain the same splits in all models


# --- TRANSFORMATIONS ---
SUPPORT_AUGMENTATIONS = nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), fill=1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
)

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), fill=255),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor()         
])


EVAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((84, 84)), 
        transforms.ToTensor()
    ])