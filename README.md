# Advanced Machine Learning Project

This repository hosts the development of a project investigating the application of Meta-learning algorithms within the Pokémon domain.  

## Repository Guide
To help you navigate the code, here is a detailed breakdown of the folder structure and the purpose of each component.

### 📂 Data
* **`pokemon_data_linked.csv`**: Expliación
* **`pokemon_sprites2.tar.gz`**: Explicación

### 📂 Dataloaders
* **`dataloader.py`**: Expliación
* **`dataloader_baseline.py`**: Expliación
* **`sampler.py`**: Expliación
  
### 📂 Dataset_Analysis
* **`first_analysis.py`**: Expliación
* **`get_family_id.py`**: Expliación
* **`heatmap_tipos.png`**: Expliación
* **`resultados_analisis.txt`**: Expliación
  
### 📂 Main
* **`Main_baseline.py`**: Expliación
* **`Main_hyper.py`**: Expliación
* **`Main_hyper_individual.py`**: Expliación

### 📂 Models
* **`Baseline.py`**: Expliación
* **`Hypernetwork.py`**: Expliación
* **`Reptile.py`**: Expliación

### 📂 Utils
* **`globals.py`**: Expliación
* **`utils.py`**: Expliación

### 📂 Models
* **`test_hyper.py`**: Expliación
* 📂 Tests_Baseline: Expliación
  * **`test_evolution_task.py`**: Expliación
  * **`test_with_finetuning.py`**: Expliación



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
