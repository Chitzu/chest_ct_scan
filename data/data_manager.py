from torch.utils.data import Dataset
import os
import glob
import torch
from data.base_dataset import BaseDataset
from data.base_dataset_test import BaseDatasetTest


class DataManager:
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config["batch_size"]

    def get_train_valid_test_dataloaders(self, num_workers=2):

        dataset = BaseDataset(path=self.config["train_path"], mode="train")
        dataset_valid = BaseDataset(path=self.config["valid_path"], mode="valid")
        dataset_test = BaseDatasetTest(path=self.config["test_path"], mode="test")

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.batch_size,
                                                   pin_memory=False, shuffle=True, num_workers=num_workers)

        validation_loader = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=self.batch_size,
                                                        pin_memory=False, num_workers=num_workers)
        
        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                        batch_size=self.batch_size,
                                                        pin_memory=False, num_workers=num_workers)
        
        return train_loader, validation_loader, test_loader
    


