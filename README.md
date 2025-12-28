# Advanced Machine Learning Project

This repository hosts the development of a project investigating the application of Meta-learning algorithms within the Pokémon domain.  

## Repository Guide
To help you navigate the code, here is a detailed breakdown of the folder structure and the purpose of each component.

### 📂 Data
* **`pokemon_data_linked.csv`**: The CSV contains useful data about all Pokémon used in the project, such as name, pokédex ID, and family ID
* **`pokemon_sprites2.tar.gz`**: This compressed file has all the sprites from gen I to gen V for all the Pokémon used in this project (gen I to IV). The structure inside is a folder with each Pokémon name, inside of which are all the available images.

### 📂 Dataloaders
* **`dataloader.py`**: This file contains the functions needed to create dataloaders for train and test, as well as getting the structured splits according to the selected partition (random, type, generation)
* **`dataloader_baseline.py`**: Implements the data pipeline for the Baseline's supervised pre-training, utilizing a MappedSubset wrapper to transform global Pokémon IDs into contiguous local classification targets.
* **`sampler.py`**: This file contains the samplers used for the different tasks. This will sample the images that will be present in the support and query sets.
  
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
* **`Reptile.py`**: Here you can find the necessary functions to define the reptile algorithm, as well as to train and test it. In this case it is used with our Backbone, but can be applied to any other.

### 📂 Utils
* **`globals.py`**: Contains several global functions, such as n-way, k-shot, epochs…, utilised in multiple files.
* **`utils.py`**:  In this file you can find multiple utility functions to visualise and plot results, set a seed to make experiments reproducible and augment data

### 📂 Tests
* **`test_hyper.py`**: Expliación
* 📂 **Tests_Baseline**: Expliación
  * **`test_evolution_task.py`**: Performs episodic evaluation on the 'Oak' evolution task using test-time fine-tuning.
  * **`test_with_finetuning.py`**: Executes the meta-testing pipeline for standard classification, utilizing the Pokedex sampler to generate random episodes and measuring the efficacy of the test-time adaptation loop on unseen species.

### 📂 Trains
* **`train_baseline.py`**: Orchestrates the end-to-end training loop for the feature extractor, integrating aggressive data augmentation, structured dataset splitting (Random/Generation/Type), and Weights & Biases experiment tracking.
* **`train_baseline_evolution.py`**: Runs the supervised training loop for the Evolutionary Task, re-mapping target labels from Species IDs to Family IDs to enforce morphological generalization during the backbone optimization phase.
* **`train_hyper.py`**: Expliación
* **`train_rept.py`**: In this code you will find the code to train the reptile algorithm. It runs the algorithm with the selected data split (random, generation or type) and using (or not) data augmentation.

### 📄 Gotta_learn__em_all.pdf
* Report of the project

