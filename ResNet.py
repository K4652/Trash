import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):

    def __init__(self,num_classes):
        super(ResNet,self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self,x ):
        return self.resnet(x)