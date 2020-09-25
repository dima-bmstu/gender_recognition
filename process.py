import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json

BATCH_SIZE = 100

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def get_device():
    if torch.cuda.is_available():
        print("Running on the GPU")
        return torch.device("cuda:0")
    else:
        print("Running on the CPU")
        return torch.device("cpu")    

def get_tranformed_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    return np.array(img)/255

def process_batch(net, device, batch, result, labels): 
    net_out = net(torch.Tensor(batch["data"]).to(device).view(-1,1,50,50))
    for img_name, img_result in zip(batch["names"], net_out):
        result[img_name] = labels[torch.argmax(img_result)]

def process(path_to_process):
    device = get_device()
    net = torch.load("full_model_94.pt").to(device)
    net.eval()

    with torch.no_grad():
        batch = {"names": [], "data" : []}
        result = {}
        result_labels = ['male', 'female']
        for f in tqdm(os.listdir(path_to_process)):
            if "jpg" in f:
                try:
                    img_path = os.path.join(path_to_process, f)
                    batch["data"].append(get_tranformed_image(img_path))
                    batch["names"].append(f)
                except Exception as e:
                    print("Failed to load image:", e)
                if len(batch["data"]) == BATCH_SIZE:
                    process_batch(net, device, batch, result, result_labels)
                    batch = {"names": [], "data" : []}
            else:
                print(f"file {f} is not jpg image")
        if len(batch["data"]):
            process_batch(net, device, batch, result, result_labels)
            batch = {"names": [], "data" : []}
        
        with open('process_results.json', 'w') as outfile:
            json.dump(result, outfile)
        


def print_help():
    print("Usage: {program_name} {path/to/process}",
          "for example: python3 process.py path/to/process", sep="\n")

def main(argv):
    if len(argv) > 1:
        process(argv[1])
    else:
        print_help()
        return

if __name__== "__main__":
    main(sys.argv)