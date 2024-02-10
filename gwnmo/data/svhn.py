import os
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights

_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
DATASET_DIR = os.getenv('DATASET_DIR')
if DATASET_DIR is None:
    DATASET_DIR = '/shared/sets/datasets'

def setup_SVHN(batch_size: int = 32):
    """
    Returns properly setup SVHN Dataloader:
    `(train_loader, test_loader)`
    """

    trans = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        ResNet18_Weights.DEFAULT.transforms(antialias=True), 
    ])
    train_loader = DataLoader(
        datasets.SVHN(DATASET_DIR, split='train', download=True, transform=trans),
        batch_size=batch_size, shuffle=True, drop_last=True, **_kwargs
    )
    test_loader = DataLoader(
        datasets.SVHN(DATASET_DIR, split='test', download=True, transform=trans),
        batch_size=batch_size, shuffle=False, drop_last=True, **_kwargs
    )

    return (train_loader, test_loader)