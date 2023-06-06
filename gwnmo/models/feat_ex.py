from torch import nn
import torchvision as tv


class FeatEx(nn.Module):
    """
    Simple wrapper around pretrained ResNet18 used for feature extraction.
    Doesn't include last layer to allow for more flexibility.
    Output size is `[512]`
    """

    def __init__(self):
        super(FeatEx, self).__init__()

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
