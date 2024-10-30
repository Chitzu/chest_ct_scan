import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer():
    def __init__(self, config, train_loader, valid_loader, test_loader, model) -> None:
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = model
        self.best_model = -1
        self.latest_acc = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):

        BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

        self.model.train().to(self.device).float()
        for x, y in tqdm(self.train_loader, bar_format=BAR_FORMAT):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x).float()
            
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()

            pred = pred.softmax(1).argmax(1)

            acc = (pred == y).sum() / len(pred) * 100
        # print(pred.shape, y.shape)
        # print(len(self.train_loader))
        print(acc)
        print(pred[:10], y[:10])

            

    def valid_epoch(self, epoch):
        BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

        self.model.eval().to(self.device).float()
        for x, y in tqdm(self.valid_loader, bar_format=BAR_FORMAT):
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                pred = self.model(x).float()
                loss = self.loss_fn(pred, y)

            pred = pred.softmax(1).argmax(1)

            acc = (pred == y).sum() / len(pred) * 100
        # print(pred.shape, y.shape)
        # print(len(self.train_loader))
        print(acc)
        print(pred[:10], y[:10])

    def save_model(self, epoch, best=False):
        pass

    def train(self):

        for epoch in range(self.config["train_epochs"]):
            print(f"Training on epoch: {epoch}")
            self.train_epoch(epoch)
        
        print(" ")
        for epoch in range(self.config["valid_epochs"]):
            print(f"Valid on epoch: {epoch}")
            self.valid_epoch(epoch)

            if epoch % self.config["save_train_epochs"]:
                self.save_model(epoch)
            
            if self.latest_acc > self.best_model:
                self.save_model(epoch, best=True)

            