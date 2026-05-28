# Путь: src/datasets/mnist.py

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size: int = 128, root: str = './data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_fashion_mnist_loaders(batch_size: int = 128, root: str = './data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader