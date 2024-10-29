import torch
import os 
import json
from data.data_manager import DataManager

if __name__ == "__main__":
    
    config = json.load(open('config.json'))
    data_manager = DataManager(config)
    train_loader, validation_loader, test_loader = data_manager.get_train_valid_test_dataloaders()
    
    print(len(train_loader), len(validation_loader))