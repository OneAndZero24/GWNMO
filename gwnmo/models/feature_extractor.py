from torch import nn
import torch
import torchvision as tv
from .backbone import Conv4, Conv4Pool, Conv4S, Conv6, ResNet10, ResNet18, ResNet34, ResNet50, ResNet101
from typing import Union, Literal

# 50 and 101 were not used in papers
feature_extractors = dict(
    Conv4 = Conv4,
    Conv4Pool = Conv4Pool,
    Conv4S = Conv4S,
    Conv6 = Conv6,
    ResNet10 = ResNet10,
    ResNet18 = ResNet18,
    ResNet34 = ResNet34,
    ResNet50 = ResNet50,
    ResNet101 = ResNet101,
)

# Initial pretrained feature extractor (ResNet18)
class FeatureExtractor(nn.Module):
    """
    Simple wrapper around pretrained ResNet18 used for feature extraction.
    Doesn't include last layer to allow for more flexibility.
    Output size is `[512]`
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        resnet: tv.models.ResNet = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)     # Pre-trained ResNet18
        layers: list[nn.Module] = list(resnet.children())

        self.model = nn.Sequential(*layers[:-1])
        for param in self.model.parameters():
            param.requires_grad = False
        

    def forward(self, x):
        """
        Input must have ResNet18 digestible format
        """

        return self.model(x)


class SimpleFeatureExtractor(nn.Module):
    """
    Simple, trainable CNN FE
    Output size is `[512]`
    """

    def __init__(self):
        super(SimpleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(3, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = torch.flatten(x, 1) 
        return x


# Trainable feature extractors used previously in https://github.com/gmum/few-shot-hypernets-public
# TrainableFeatureExtractor assumes that all layers are trainable
class TrainableFeatureExtractor(nn.Module):
    """
    Wrapper around popular feature extractors
    """
    def __init__(self, 
                 backbone_name = Union
                 [
                    Literal["Conv4"], 
                    Literal["Conv4Pool"], 
                    Literal["Conv4S"],
                    Literal["Conv6"],
                    Literal["ResNet10"],
                    Literal["ResNet18"],
                    Literal["ResNet34"],
                    Literal["ResNet50"],
                    Literal["ResNet101"]
                ],
                flatten = True
    ):
        super(TrainableFeatureExtractor, self).__init__()
        self.model = feature_extractors[backbone_name](flatten=flatten)
    
    def forward(self, x):
        output = self.model(x)
        output = torch.reshape(self.model(x),(output.shape[0], output.shape[1], 1, 1))
        return output