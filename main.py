from dataloader import *
from utils.utils import *
from utils.globals import *

from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

if __name__ == '__main__':
    # 1. Initialize Dataset
    transform_pipeline = transforms.Compose([
        transforms.Resize((84, 84)), # Standard size for meta-learning (e.g., 84x84)
        transforms.ToTensor()        # Converts PIL Image -> Tensor (C, H, W)
    ])

    ds = PokemonMetaDataset(csv_file='pokemon_data_gen1-5.csv', 
                            root_dir='pokemon_sprites',
                            transform=transform_pipeline)

    # TEST
    # Check 10 random images to be sure
    import random
    for i in range(300*15,450*15,15):
        rand_idx = random.randint(0, len(ds)-1)
        print("Checking random sample...")
        visualize_sample(ds, i)
    # TEST END
    
    
    # 2. Create Loaders (e.g., 5-way, 3-shot)
    train_loader, test_loader = get_meta_dataloaders(
        ds, n_way=5, n_shot=SUPPORT_SIZE, n_query=QUERY_SIZE, episodes=100
    )
    
    # 3. Training Loop
    for batch_idx, (images, labels) in enumerate(train_loader):
        # images shape: [batch_size, C, H, W]
        if batch_idx == 0:
            print("Visualizing first episode...")
            visualize_episode(images, n_way=5, n_shot=SUPPORT_SIZE, n_query=QUERY_SIZE)
        
        p = 5 # n_way
        k = SUPPORT_SIZE # n_shot
        q = QUERY_SIZE # n_query
        
        # Reshape to: [N_way, K_shot + Q_query, C, H, W]
        x = images.view(p, k + q, 3, 84, 84) # Assuming image size 84x84
        
        x_support = x[:, :k].contiguous() # [5, 3, 3, 84, 84]
        x_query   = x[:, k:].contiguous() # [5, 3, 3, 84, 84]
        
        # Apply your meta-learning algorithm (e.g., ProtoNet) here...
        print(f"Batch {batch_idx}: Support {x_support.shape}, Query {x_query.shape}")