"""
CIFAR-10 dataloader with separate train/test transforms.
Augmentation is applied ONLY to real images fed to the discriminator,
not to generated images (per the assignment specification).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Normalization parameters: scale [0,1] -> [-1,1] to match Tanh output
CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD  = (0.5, 0.5, 0.5)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_train_transform(image_size: int = 32) -> transforms.Compose:
    """
    Training transform with augmentation for the discriminator's real images.
    RandomHorizontalFlip + ColorJitter make sense for CIFAR-10 natural objects.
    We avoid rotations >15 deg and RandomResizedCrop that distort class features.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


def get_test_transform(image_size: int = 32) -> transforms.Compose:
    """Test/validation transform: only ToTensor + Normalize, no augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


def get_dataloaders(
    data_root: str = "./data/cifar10",
    batch_size: int = 64,
    image_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for CIFAR-10.
    Train loader uses augmentation; test loader does not.
    """
    train_transform = get_train_transform(image_size)
    test_transform  = get_test_transform(image_size)

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,   # Ensures full batches for WGAN-GP gradient penalty
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
