import torch
import torchvision
import torch.nn.functional as F
from utils import model_arch
import logging
from fastai.vision.all import *

class generator:
    def __init__(self, name, num_class, image_size):
        self.name = name
        self.num_class = num_class
        self.image_size = image_size

        if self.name == "resnet50":
            self.model = resnet50(self.num_class)  
        elif self.name == "XResnet50":
            self.model = XResnet50(self.num_class)




class resnet50(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = torchvision.models.resnet50(weights=None)
        self.network.fc = torch.nn.Linear(in_features=2048, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)

class XResnet50(model_arch.Module):
    def __init__(self, num_class):
        super().__init__()
        self.network = xresnet50(pretrained=True)
        self.network[-1] = torch.nn.Linear(in_features=2048, out_features=num_class, bias=True)
    
    def forward(self, xb):
        return self.network.forward(xb)
