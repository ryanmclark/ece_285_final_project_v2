#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:08:55 2018

@author: ryanclark
Header:
This file is used to take the raw data provided by the original creator's download.sh
script on https://github.com/mitmul/ssai-cnn and convert it into a serialized torch
dataset object that gets saved to your disk. The raw data is approximately 4,000 .tiff
1500x1500 resolution images that are either ~6MB (sat) or ~2MB (map) in size, totally 
at about 12GB of data. It is important to note that though the serialized object is easy
to use, it is appoximately 7 times the size of the original data, so downsampling isn't
only necessary to reduce the complexity of the cnn model, but also for data handling.
This script assumes the the input is in the same structure provided by the download.sh
script and that this script lives outside the 'data' directory provided by the
download.sh script. The output directories will be created and the serialized dataset
objects will be saved according to their adjusted resolutions.

Input:
  data:
    mass_roads:
      train:
        sat:
          *.tiff
        map:
          *.tif
       test:
         sat:
          *.tiff
         map:
          *.tif
       valid:
          ...
    mass_buildings:
       train:
         ...
       test:
         ...
       valid:
         ...
         
Output:
  datasets:
    ...
    Resolution_###:
      buildings:
        train_dataset:
        test_dataset:
        valid_dataset:
        complete_dataset:
      roads:
        train_dataset:
        test_dataset:
        valid_dataset:
        complete_dataset:
    Resolution_###:
      ... 
"""

# Libraries
import torch
import glob
import os
import cv2
import numpy as np
import time
import torch.utils.data as data_utils

# PARAMETERS: CHANGE THIS TO GET DESIRED OUTPUT.
img_width = 250
img_height = 250

# Constants:
channels = 3 #RGB
os.chdir('..')
home = os.getcwd() # Get the Home Directory

# Define the directories to navigate to the raw data.
directories = dict()
directories['buildings'] = dict()
directories['buildings']['train'] = dict()
directories['buildings']['validate'] = dict()
directories['buildings']['test'] = dict()

directories['buildings']['train']['images'] = home+'/data/mass_buildings/train/sat'
directories['buildings']['train']['labels'] = home+'/data/mass_buildings/train/map'

directories['buildings']['validate']['images'] = home+'/data/mass_buildings/valid/sat'
directories['buildings']['validate']['labels'] = home+'/data/mass_buildings/valid/map'

directories['buildings']['test']['images'] = home+'/data/mass_buildings/test/sat'
directories['buildings']['test']['labels'] = home+'/data/mass_buildings/test/map'

directories['roads'] = dict()
directories['roads']['train'] = dict()
directories['roads']['validate'] = dict()
directories['roads']['test'] = dict()

directories['roads']['train']['images'] = home+'/data/mass_roads/train/sat'
directories['roads']['train']['labels'] = home+'/data/mass_roads/train/map'

directories['roads']['validate']['images'] = home+'/data/mass_roads/valid/sat'
directories['roads']['validate']['labels'] = home+'/data/mass_roads/valid/map'

directories['roads']['test']['images'] = home+'/data/mass_roads/test/sat'
directories['roads']['test']['labels'] = home+'/data/mass_roads/test/map'



def _create_dataset(directories):      
    '''
    Creates torch tensor arrays for the images (sat) and labels (map). Reads each image 
    one by one, downsamples it, reorders the axis, casts as torch tensor abject, and 
    appends to the array. Once one image is done, navigates to the labels to find its
    respective label. Once finished, creates a torch dataset object from the images with
    their respective labels. Exits if the data becomes too large for the RAM to handle.
    
    Parameters: directories (list of strings)
    
    Outputs: subset (torch.dataset)
    '''
    # Navigate to directory of interest and group all .tiff file names.
    os.chdir(directories['images'])
    fnames = glob.glob('*.tiff')
    
    # Number of samples is the number of files
    num_of_samples = len(fnames)
    
    # Ad hoc safety net used to prevent maxing out the RAM and crashing the computer.
    #if img_width > 500 and num_of_samples >= 300:
    #    num_of_samples = 300
    #    print('Limiting the dataset to 300 samples due to large dataset size.')
    
    # Define tensor for the immages (sat) and label (map) data
    images = torch.Tensor(num_of_samples, channels, img_height, img_width)
    labels = torch.Tensor(num_of_samples, channels, img_height, img_width)

    # Loop through all files and create the tensor objects accordingly.
    for count, fname in enumerate(fnames):
        
        if np.mod(count, 10) == 0:
            print('Sample Number: %d/%d' % (count+1, len(fnames)))
        
        # Break if count exceeded the number of samples.
        if count > num_of_samples:
          break
        
        # Import image, reshape, reorder axes, create tensor object, store in tensor array.
        img = cv2.imread(fname)
        #img = cv2.resize(img, (img_height, img_width))
        img = np.moveaxis(img, [0, 1], [1, 2])
        img = torch.from_numpy(img)
        images[count,:,:,:] = img
        
        # Navigate to label (map) directory.
        os.chdir(directories['labels'])
        fname = fname[:-1] # Image is .tiff but label is .tif
        
        # Import image, reshape, reorder axes, create tensor object, store in tensor array.
        img = cv2.imread(fname)
        #img = cv2.resize(img, (img_height, img_width))
        img = np.moveaxis(img, [0, 1], [1, 2])
        img = torch.from_numpy(img)
        labels[count,:,:,:] = img        
        
        # Navigate back to image (sat) directory.
        os.chdir(directories['images'])
    
    os.chdir(home) # Navigate back to the home directory.
    
    print('Processed %d samples' % num_of_samples)
    
    print('Images Shape: ' + str(images.shape))
    print('Labels Shape: ' + str(labels.shape))
    print('')
    
    # Create the dataset object
    subset = data_utils.TensorDataset(images, labels)
    
    return subset



# Performance Timer
start = time.time()

# String to represent output resolution
output_resolution = str(img_width) + 'x' + str(img_height)

# Create dataset directory if doesn't exist.
datasets_dir = home + '/datasets'
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

# Define strings output directories 
output_dir = datasets_dir + '/resolution_' + output_resolution
output_buildings_dir = output_dir + '/buildings'
output_roads_dir = output_dir + '/roads'

# Makes output directories if they do not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(output_buildings_dir)
    os.makedirs(output_roads_dir)

print('Constructing the datasets at an image resolution of' + output_resolution + '.')

for subset in ['test' ,'train', 'validate']:
    
    print('Constructing the buildings ' + subset + ' dataset.')
    buildings = _create_dataset(directories['buildings'][subset])
    os.chdir(output_buildings_dir)
    torch.save(buildings, 'buildings_' + subset + '_dataset.pkl')
    os.chdir(home)
    
    print('Constructing the roads ' + subset + ' dataset.')
    roads = _create_dataset(directories['roads'][subset])
    os.chdir(output_roads_dir)
    torch.save(roads, 'roads_' + subset + '_dataset.pkl')
    os.chdir(home)
    
end = time.time()
    
print('Program took %0.2f seconds to run' %  (end - start))

