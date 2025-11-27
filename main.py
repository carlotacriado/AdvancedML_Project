from dataloader import *
from utils.utils import *
from utils.globals import *

from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

from Models.Reptile import *
from Models.Baseline import *

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Dataset
    print("Creating Dataset...")
    transform_pipeline = transforms.Compose([
        transforms.Resize((84, 84)), 
        transforms.ToTensor()
    ])
    
    ds = PokemonMetaDataset('pokemon_data_gen1-5.csv', 'pokemon_sprites', transform=transform_pipeline)

    # 2. Create Loaders
    # Note: 'episodes' arg controls how long the loop runs before stopping to validate
    train_loader, test_loader = get_meta_dataloaders(
        ds, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, episodes=EPISODES_PER_EPOCH
    )

    # 3. INITIALIZE MODEL
    print("Initializing Model...")
    meta_model = ConvBackbone().to(device)

    # 4. START TRAINING
    train_reptile(meta_model, train_loader, test_loader, device)
    