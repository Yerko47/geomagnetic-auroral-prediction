"""
Script for creating the project structure
"""
import os
from variables import *
from pathlib import Path

def structure_proyect(proyect_file, files_plots, files):
    """
    Create the project structure.
    
    Args:
    ----------
        proyect_file : str
            Path to the project directory.
        files_plots : list
            List of subdirectories for plots.
        files : list
            List of subdirectories for the project.
    """
    # Create the main project directory
    os.makedirs(proyect_file, exist_ok=True)

    # Create subdirectories
    for file in files:
        os.makedirs(os.path.join(proyect_file, file), exist_ok=True)

    # Create subdirectories for plots
    for file in files_plots:
        os.makedirs(os.path.join(proyect_file, 'plots', file), exist_ok=True)
        


project_file = Path(__file__).resolve().parent.parent



structure_proyect(project_file, files_plots, files)