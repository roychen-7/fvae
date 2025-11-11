import qlib
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader


class QlibDataset(Dataset):
    """
    Dataset class for loading qlib data for FactorVAE training.
    Loads raw OHLCV data directly from qlib storage.
    """
    def __init__(self, start_time, end_time, num_stocks=100):
        """
        Initialize the qlib dataset.

        Args:
            start_time: Start date for data (e.g., '2020-01-01')
            end_time: End date for data (e.g., '2020-12-31')
            num_stocks: Number of stocks to load
        """
        # Initialize qlib
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')

        from qlib.data import D

        # Get list of available stock files
        features_path = os.path.expanduser('~/.qlib/qlib_data/cn_data/features/')
        all_stocks = [f for f in os.listdir(features_path) if not f.startswith('.')]

        # Use only first num_stocks for faster training
        stocks_to_use = all_stocks[:num_stocks]

        print(f"Loading data from {start_time} to {end_time}...")
        print(f"Total available stocks: {len(all_stocks)}")
        print(f"Using {len(stocks_to_use)} stocks for training")

        # Load basic OHLCV features
        feature_names = ['$close', '$open', '$high', '$low', '$volume']

        all_data = []
        success_count = 0

        for stock in stocks_to_use:
            try:
                # Load data for this stock
                df = D.features(
                    [stock],
                    feature_names,
                    start_time=start_time,
                    end_time=end_time
                )

                if df is not None and not df.empty and len(df) > 0:
                    data = df.values
                    # Remove any rows with all NaN
                    data = data[~np.isnan(data).all(axis=1)]
                    if len(data) > 0:
                        all_data.append(data)
                        success_count += 1

            except Exception as e:
                if success_count < 3:  # Show first few errors for debugging
                    print(f"Error loading {stock}: {e}")
                continue

        print(f"Successfully loaded data from {success_count} stocks")

        if not all_data:
            raise ValueError("No data loaded from qlib!")

        # Concatenate all data
        self.data = np.vstack(all_data)

        print(f"Raw data shape before processing: {self.data.shape}")

        # Handle NaN values
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize the data (z-score normalization)
        self.mean = np.mean(self.data, axis=0, keepdims=True)
        self.std = np.std(self.data, axis=0, keepdims=True) + 1e-8
        self.data = (self.data - self.mean) / self.std

        # Get feature dimension
        self.num_features = self.data.shape[1]

        print(f"Final dataset: {len(self.data)} samples with {self.num_features} features")
        print(f"Data statistics - Min: {self.data.min():.2f}, Max: {self.data.max():.2f}, Mean: {self.data.mean():.2f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Returns a torch tensor of the features.
        """
        sample = self.data[idx]
        return torch.FloatTensor(sample)


def get_qlib_dataloader(dataset_name='alpha158',
                        start_time='2020-01-01',
                        end_time='2020-12-31',
                        batch_size=64,
                        num_stocks=100):
    """
    Create a dataloader for qlib data.

    Args:
        dataset_name: Dataset name (for compatibility)
        start_time: Start date for data
        end_time: End date for data
        batch_size: Batch size for dataloader
        num_stocks: Number of stocks to load

    Returns:
        DataLoader object and number of features
    """
    dataset = QlibDataset(start_time, end_time, num_stocks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader, dataset.num_features
