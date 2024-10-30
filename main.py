import torch
import os 
import json
from data.data_manager import DataManager
from train import Trainer
from network import Network

if __name__ == "__main__":
    
    config = json.load(open('config.json'))
    data_manager = DataManager(config)
    train_loader, validation_loader, test_loader = data_manager.get_train_valid_test_dataloaders()
    
    model = Network()
    trainer = Trainer(config, train_loader, validation_loader, test_loader, model)
    trainer.train()
    print(len(train_loader), len(validation_loader))