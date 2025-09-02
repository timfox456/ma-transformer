# SPDX-License-Identifier: Apache-2.0
 
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class TickDataset(Dataset):
    def __init__(self, file_path, sequence_length=10):
        self.sequence_length = sequence_length
        
        # Load and preprocess data
        df = pd.read_csv(file_path)
        
        # Calculate microstructure features
        df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
        df['wap'] = (df['bid_price'] * df['ask_volume'] + df['ask_price'] * df['bid_volume']) / (df['bid_volume'] + df['ask_volume'])
        df['order_book_imbalance'] = df['bid_volume'] / (df['bid_volume'] + df['ask_volume'])
        df['time_diff'] = df['timestamp'].diff().fillna(0)

        # Select and normalize features
        features_to_normalize = ['price', 'bid_ask_spread', 'wap', 'order_book_imbalance', 'time_diff']
        self.feature_means = df[features_to_normalize].mean()
        self.feature_stds = df[features_to_normalize].std()

        normalized_df = (df[features_to_normalize] - self.feature_means) / self.feature_stds
        
        self.features = normalized_df.values

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        features = self.features[idx:idx+self.sequence_length]
        # The target is the next WAP
        target = self.features[idx+self.sequence_length, 2] # Index 2 corresponds to WAP
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def create_data_loaders(file_path, sequence_length, batch_size, train_split=0.8):
    dataset = TickDataset(file_path, sequence_length)
    
    # Split dataset into training and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset # Return dataset to access means/stds for visualization
