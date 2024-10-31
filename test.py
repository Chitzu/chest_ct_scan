import torch
import os 
import json
import argparse
import cv2
import numpy as np
from data.data_manager import DataManager
from train import Trainer
from network import Network

if __name__ == "__main__":
    
    config = json.load(open('config.json'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.load_state_dict(torch.load(os.path.join(config["save_model_path"], "best_model.pth"), weights_only=True))
    model.eval().to(device).float()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path")
    parser.add_argument("-l", "--label")
    args = parser.parse_args()

    img = cv2.imread(args.path)
    img.resize((400, 400, 3))
    img = img / 255.0
    img = np.permute_dims(img, (2, 1, 0))
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img).to(device).float()
    pred = model(img)

    print(pred, pred.detach().cpu().softmax(1).argmax(1))
    print(pred.detach().cpu().softmax(1).argmax(1) == int(args.label))
