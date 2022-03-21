#!/usr/bin/env python3
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

# Load the study database
study = optuna.load_study(study_name="DL2022_multi", storage="sqlite:///multi.db")

# Plot the slice plot for each kernel size and skip kernel size

# Plot hyperparam importance

# Plot Optimization history
