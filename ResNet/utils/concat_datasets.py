#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:08:55 2018

@author: ryanclark

Header:
This file concats the torch dataset objects for the train, test, and validate datasets into
a single dataset object. This is to allow K-Folds cross validation and random batch pulling
for the CNNs. The full dataset will be saved along side the input datasets with the name
'XXXX_Complete_dataset' where XXXX is either 'roads' or 'buildings'
"""

# Libraries
import os
import glob
import torch.utils.data as data_utils
import torch

# PARAMETERS: CHANGE THIS TO NAVIGATE TO THE DESIRED DATASET DIRECTORY.
img_width = 250
img_height = 250

# Constants:
os.chdir('..')
home = os.getcwd() # Get the Home Directory

# Constructed from the parameters
# String to represent output resolution
resolution = str(img_width) + 'x' + str(img_height)

# Directories for the roads and buildings datasets
buildings_directory = home + '/datasets/resolution_' + resolution + '/buildings'
roads_directory = home + '/datasets/resolution_' + resolution + '/roads'

# Navigate to the buildings directory, glob the datasets, and concat them as one
# object and save them.
os.chdir(buildings_directory)
b_datasets = glob.glob('*.pkl')
print('Buildings Datasets:')
print(b_datasets)

buildings_dataset = data_utils.ConcatDataset(
    [torch.load(b_datasets[0]),
    torch.load(b_datasets[1]),
    torch.load(b_datasets[2])])

torch.save(buildings_dataset, 'buildings_complete_dataset.pkl')

# Navigate to the roads directory, glob the datasets, and concat them as one
# object and save them.

os.chdir(roads_directory)
r_datasets = glob.glob('*.pkl')
print('Roads Datasets:')
print(r_datasets)

roads_dataset = data_utils.ConcatDataset(
    [torch.load(r_datasets[0]),
    torch.load(r_datasets[1]),
    torch.load(r_datasets[2])])

torch.save(roads_dataset, 'roads_complete_dataset.pkl')

