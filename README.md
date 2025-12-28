The project is organized as follows:

```bash
ADVANCEDML_PROJECT/
├── Data/                  # Raw images and CSV metadata
├── Dataloaders/           # Custom PyTorch dataloaders and samplers
├── Dataset_Analysis/      # EDA, evolutionary 'Family ID' generation, and data visualization
├── logs/                  # Local training logs
├── Main/                  # Core execution scripts 
├── Models/                # Architecture definitions (Baseline, Reptile, Hypernet)
├── tests/                 # Test loops
├── trains/                # Training loops/functions and episodic trainers
├── Utils/                 # Helper functions and configuration parsers
├── wandb/                 # Weights & Biases tracking data
└── Results/               # Figures and plots for the report
