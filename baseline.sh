#!/bin/bash
#SBATCH --job-name=baseline_train   # Nombre del trabajo
#SBATCH --output=logs/logs_baseline/BT_%j.out   # Archivo de salida (asegúrate de crear la carpeta logs)
#SBATCH --error=logs/logs_baseline/BT_%j.err     # Archivo de errores
#SBATCH --ntasks=1                    # Número de tareas
#SBATCH --cpus-per-task=4             # CPUs
#SBATCH --mem=16G                     # Memoria RAM
#SBATCH --gres=gpu:1                  # SOLICITAR GPU (Importante para no usar solo CPU)
#SBATCH --partition=tfg               # Nombre de la partición (pregunta a tu admin cuál es)

# 1. Cargar modulos necesarios (esto depende de tu cluster)
module load python/3.9
module load cuda/11.7

# 2. Activar tu entorno virtual (conda o venv)
source aml_venv/bin/activate

# 3. Ejecutar el script de python
PYTHONPATH=. python3 Main/Main_baseline.py