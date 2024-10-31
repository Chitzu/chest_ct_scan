import torch
import os
import torch.nn as nn
from tqdm import tqdm

class Trainer():
    def __init__(self, config, train_loader, valid_loader, test_loader, model) -> None:
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = model
        self.best_train_model = -1
        self.latest_train_acc = -1
        self.best_valid_model = -1
        self.latest_valid_acc = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

        self.model.train().to(self.device).float()
        for x, y in tqdm(self.train_loader, bar_format=BAR_FORMAT):
            x = x.to(self.device) / 255.0
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x).float()
            
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()

            pred = pred.softmax(1).argmax(1)
            acc = (pred == y).sum() / len(pred) * 100
            
            self.latest_train_acc = acc
            
            if self.latest_train_acc > self.best_train_model:
                print(f'model saved with {self.latest_train_acc}% acc')
                self.best_train_model = self.latest_train_acc 
                self.save_model(train=True)
        print(loss.item(), acc.item())

            

    def valid_epoch(self, epoch):
        BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

        self.model.load_state_dict(torch.load(os.path.join(self.config["save_model_path"], "best_model_train.pth"), weights_only=True))
        self.model.eval().to(self.device).float()
        for x, y in tqdm(self.valid_loader, bar_format=BAR_FORMAT):
            x = x.to(self.device) / 255.0
            y = y.to(self.device)

            with torch.no_grad():
                pred = self.model(x).float()
                loss = self.loss_fn(pred, y)

            pred = pred.softmax(1).argmax(1)
            acc = (pred == y).sum() / len(pred) * 100

            if self.latest_valid_acc > self.best_valid_model:
                self.best_valid_model = self.best_valid_model 
                self.save_model()
        print(loss.item(), acc.item())
        # print(pred[:10], y[:10])

    def save_model(self, train=False):
        if train:
            torch.save(self.model.state_dict(), os.path.join(self.config["save_model_path"], "best_model_train.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config["save_model_path"], f"best_model.pth"))

    def train(self):

        for epoch in range(self.config["train_epochs"]):
            print(f"Training on epoch: {epoch}")
            self.train_epoch(epoch)
        
        print(" ")
        for epoch in range(self.config["valid_epochs"]):
            print(f"Valid on epoch: {epoch}")
            self.valid_epoch(epoch)
            