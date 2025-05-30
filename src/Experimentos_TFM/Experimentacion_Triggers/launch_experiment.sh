#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=cpu

# Activar entorno virtual
source ../../../../.venv/bin/activate

# Lanzar experimento con los otros clasificadores
srun --exclusive --cpus-per-task=10 python3 -u run_benchmark_calib_all_triggers.py > isotonic_all_datasets_hydra.txt