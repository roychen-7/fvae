import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Utility functions for data processing

def normalize_data(data):
    """
    Normalize the input data to have zero mean and unit variance.
    :param data: Input data tensor
    :return: Normalized data tensor
    """
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / std


def load_dataset(dataset_name, batch_size):
    """
    Load and return a dataset and its DataLoader.
    :param dataset_name: Name of the dataset to load
    :param batch_size: Batch size for the DataLoader
    :return: DataLoader for the specified dataset
    """
    # Placeholder for dataset loading logic
    # This should be replaced with actual dataset loading code
    if dataset_name == '2d_shapes':
        # Example transformation pipeline
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Example dataset
        dataset = Dataset()  # Replace with actual dataset
    elif dataset_name == '3d_shapes':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = Dataset()  # Replace with actual dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def augment_data(data):
    """
    Apply data augmentation techniques to the input data.
    :param data: Input data tensor
    :return: Augmented data tensor
    """
    # Example augmentation: random horizontal flip
    transform = transforms.RandomHorizontalFlip(p=0.5)
    return transform(data)