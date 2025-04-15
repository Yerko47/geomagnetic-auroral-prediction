"""
Script for creating the project structure
"""
import os

def structure_proyect(proyect_file, files):
    for file in files:
        file_proyect = proyect_file + file
        if not os.path.exists(file_proyect):
            os.makedirs(file_proyect)

proyect_file = f'/home/yerko/Desktop/Proyects/auroral_prediction/'
omni_file = f'/data/omni/hro_1min/'

files = ['data/raw/', 'data/processed/',
        'models/', 'notebooks/',
        'src/', 'tests/', 'docs/', 
        'plots/']

structure_proyect(proyect_file, files)