# Advanced Machine Learning Project

This repository hosts the development of a project investigating the application of Meta-learning algorithms within the Pokémon domain.  

## Repository Guide
To help you navigate the code, here is a detailed breakdown of the folder structure and the purpose of each component.

### 📂 Data
* **`pokemon_data_linked.csv`**: Expliación
* **`pokemon_sprites2.tar.gz`**: Explicación

### 📂 Dataloaders
* **`dataloader.py`**: Expliación
* **`dataloader_baseline.py`**: Implements the data pipeline for the Baseline's supervised pre-training, utilizing a MappedSubset wrapper to transform global Pokémon IDs into contiguous local classification targets.
* **`sampler.py`**: Expliación
  
### 📂 Dataset_Analysis
* **`first_analysis.py`**: This code performs a comprehensive structural audit of the dataset, classifying Pokémon by evolutionary role, mapping CSV metadata to physical image directories, and analyzing class distributions (Type/Generation) to validate suitability for meta-learning tasks.
* **`get_family_id.py`**: Implements a recursive ancestry tracer to map every species to its 'Family Leader' and appends unique integer identifiers for evolutionary grouping.
* **`heatmap_tipos.png`**: The Heatmap image (in png) of Pokémon types given each generation.
* **`resultados_analisis.txt`**: The results from the "first_analysis.py" --> to execute it once and keep the results.
  
### 📂 Main
* **`Main_baseline.py`**: This script runs the global pre-training routine for the Baseline model with configurable splits to produce the initial feature extractor weights.
* **`Main_hyper.py`**: Expliación
* **`Main_hyper_individual.py`**: Expliación

### 📂 Models
* **`Baseline.py`**: Defines the shared Conv-4 Backbone utilized by all three architectures (Baseline, Reptile, HyperNetwork) for feature extraction, alongside the specific linear ClassifierHead used during the Baseline's supervised pre-training.
* **`Hypernetwork.py`**: Expliación
* **`Reptile.py`**: Expliación

### 📂 Utils
* **`globals.py`**: Expliación
* **`utils.py`**: Expliación

### 📂 Tests
* **`test_hyper.py`**: Expliación
* 📂 **Tests_Baseline**: Expliación
  * **`test_evolution_task.py`**: Performs episodic evaluation on the 'Oak' evolution task using test-time fine-tuning.
  * **`test_with_finetuning.py`**: Executes the meta-testing pipeline for standard classification, utilizing the Pokedex sampler to generate random episodes and measuring the efficacy of the test-time adaptation loop on unseen species.

### 📂 Trains
* **`train_baseline.py`**: Orchestrates the end-to-end training loop for the feature extractor, integrating aggressive data augmentation, structured dataset splitting (Random/Generation/Type), and Weights & Biases experiment tracking.
* **`train_baseline_evolution.py`**: Runs the supervised training loop for the Evolutionary Task, re-mapping target labels from Species IDs to Family IDs to enforce morphological generalization during the backbone optimization phase.
* **`train_hyper.py`**: Expliación
* **`train_rept.py`**: Expliación

### 📄 Gotta_learn__em_all.pdf
* Report of the project

