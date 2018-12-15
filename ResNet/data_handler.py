from copy import copy
from random import shuffle
import cv2
import torch
import torchvision
from torch.utils.data.dataset import Dataset
import os
import glob
import numpy as np

class Data_Handler(Dataset):
    
    def __init__(self, dataset_type, testset, ext = '.png'):
        self.home = os.getcwd()
        self.dataset_type = dataset_type
        self.images_dir = self.home + '/data/' + dataset_type + '/' + testset + '/sat/'
        self.labels_dir = self.home + '/data/' + dataset_type + '/' + testset + '/map/'
        os.chdir(self.images_dir)
        self.fnames = glob.glob('*' + ext)
        self.dim = cv2.imread(self.fnames[0]).shape
        os.chdir(self.home)
        self.norm = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        
    def get_files(self):
        return self.fnames
    
    def create_dataset(self, shuffle_data = True):
        dataset = copy(self.fnames)
        if shuffle_data:
            shuffle(dataset)
            
        return dataset
    
    def create_batch(self, dataset, batch_size):
        
        if len(dataset) < batch_size:
            batch_size = len(dataset)
        
        images = torch.Tensor(batch_size, self.dim[2], self.dim[0], self.dim[1])
        labels = torch.Tensor(batch_size, self.dim[0], self.dim[1])
            
        for cnt in range(batch_size):
            fname = dataset.pop()
            im, la = self._load_data(fname)
            images[cnt] = im
            labels[cnt] = la
            
        return images, labels
            
    def _load_data(self, fname):
        
        os.chdir(self.images_dir)
        im = cv2.imread(fname)
        im = np.moveaxis(im, [0, 1], [1, 2])
        im = torch.from_numpy(im).type(torch.FloatTensor)
        im = self._normalize_image(im)
        os.chdir(self.labels_dir)
        la = cv2.imread(fname)[:,:,2]
        la = self._normalize_label(la)
        la = torch.from_numpy(la).type(torch.LongTensor)
        os.chdir(self.home)
        return im, la
    
    def _normalize_label(self, label):
        label[label != 0] = 1
        return label
    
    def _normalize_image(self, image):
        for chl in range(self.dim[2]):
            chl_max = torch.max(image[chl,:,:])
            chl_min = torch.min(image[chl,:,:])
            if chl_max == chl_min:
                chl_min = chl_max - 1
            image[chl,:,:] = 2*((image[chl,:,:] - chl_min) / (chl_max - chl_min)) - 1
                        
        return image
