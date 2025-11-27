BATCH_SIZE = 64 # though in practice, datasets are so small that each epoch is one batch
SUPPORT_SIZE = 3     # num examples per class in each support set
QUERY_SIZE = 3       # num examples per class in each query set

# --- HYPERPARAMETERS ---
N_WAY = 5
N_SHOT = 3     # Support Size (K)
N_QUERY = 3   # Query Size (Q)

MAX_EPOCHS = 100       # Total training duration
EPISODES_PER_EPOCH = 100 # How many episodes in one loader iteration

INNER_LR = 0.01        # Learning rate for the clone
INNER_STEPS = 5        # How many gradient steps the clone takes
EPSILON = 0.1          # Reptile outer learning rate