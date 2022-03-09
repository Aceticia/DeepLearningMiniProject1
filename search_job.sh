#!/bin/bash
#SBATCH --partition=oermannlab
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G

python optuna_script.py &
python optuna_script.py &
python optuna_script.py &
python optuna_script.py
