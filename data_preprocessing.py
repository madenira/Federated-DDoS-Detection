# This script handles synthetic data generation, preprocessing, and creating PyTorch DataLoaders.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_data():
    np.random.seed(42)
    num_samples = 10000
    num_packets = np.random.randint(1, 1000, num_samples)
    byte_count = np.random.randint(100, 100000, num_samples)
    duration = np.random.uniform(0.1, 1000, num_samples)
    protocol_type = np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples)
    protocol_type_num = pd.get_dummies(protocol_type)
    labels = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

    data = pd.DataFrame({
        'num_packets': num_packets,
        'byte_count': byte_count,
        'duration': duration,
        'protocol_TCP': protocol_type_num['TCP'],
        'protocol_UDP': protocol_type_num['UDP'],
        'protocol_ICMP': protocol_type_num['ICMP'],
        'label': labels
    })

    data.to_csv('network_traffic.csv', index=False)
    print("Dataset created and saved to 'network_traffic.csv'")

def create_data_loaders(batch_size):
    data = pd.read_csv('network_traffic.csv')
    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = TabularDataset(X_train, y_train)
    test_data = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
