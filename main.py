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
    # Train on Gen 1, 2, 3. Test on Gen 4.
    train_labels, test_labels, val_labels = get_structured_splits(
        ds, 
        split_mode='generation', 
        train_vals=['generation-i', 'generation-ii'],
        val_vals=['generation-iii'],
        test_vals=['generation-iv']
    )

    # train_labels, test_labels = get_structured_splits(
    # ds, 
    # split_mode='type', 
    # train_vals=['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Electric'],
    # test_vals=['Dragon', 'Ghost', 'Ice', 'Steel']
    # )

    train_loader, test_loader, val_loader = get_meta_dataloaders(ds, train_labels, test_labels, val_labels, N_WAY, N_SHOT, N_QUERY, EPISODES_PER_EPOCH)

    # 3. INITIALIZE MODEL
    print("Initializing Model...")
    meta_model = ConvBackbone().to(device)

    # 4. START TRAINING
    #train_reptile(meta_model, train_loader, test_loader, device)