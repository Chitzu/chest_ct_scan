from torch.utils.data import Dataset
import os
import glob
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, path="./", mode="train"):
        self.data = []
        self.labels = []
        self.path = path
        self.mode = mode
        self._process_dataset(self.path, self.mode)   
        self.transform = transforms.Compose(
            [
            transforms.Resize((500, 500)),
            transforms.RandomCrop((400, 400)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(),
            transforms.ToTensor()
            
        ]) 
            
    def _process_dataset(self, path, mode="train"):
        classes_dirs = glob.glob(os.path.join(path, "*/"))
        classes_dirs.sort()

        for i, classes in enumerate(classes_dirs):
            images = []
            images = glob.glob(os.path.join(classes, "*.png"))
            labels = [i for _ in range(len(images))]

            self.data = self.data + images
            self.labels = self.labels + labels
    
    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        label = self.labels[index]

        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)
    
