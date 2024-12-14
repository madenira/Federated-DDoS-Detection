# This is the entire code  for the project. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
from torchvision import transforms

# Step 1: Generate Synthetic Data
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

# Hyperparameters
args = {
    'lr': 0.01,
    'batch_size': 64,
    'epochs': 5,
    'clients': 5,  # Number of clients
    'max_batches': 50,  # Total maximum batches to process
    'increment': 10  # Increment of batches processed
}

# Custom PyTorch Dataset for Tabular Data
class TabularDataset(tud.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 2: Preprocess the Data and create DataLoaders
def create_data_loaders():
    data = pd.read_csv('network_traffic.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = TabularDataset(X_train, y_train)
    test_data = TabularDataset(X_test, y_test)

    train_loader = tud.DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    test_loader = tud.DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)
    return train_loader, test_loader

# Step 3: Build the DNN Model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # Adjusted input layer for 6 features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to initialize weights dictionary for aggregation
def initialize_weights(model):
    return {key: torch.zeros_like(param) for key, param in model.state_dict().items()}

# Function to aggregate weights
def aggregate_weights(weights_list):
    aggregated_weights = {key: torch.zeros_like(weights_list[0][key]) for key in weights_list[0]}
    for key in aggregated_weights.keys():
        for weights in weights_list:
            aggregated_weights[key] += weights[key]
        aggregated_weights[key] /= len(weights_list)
    return aggregated_weights
# Step 4: Federated learning Process 
# Train function for each client
def train(model, device, train_loader, optimizer, criterion, client_id):
    model.train()
    for epoch in range(args['epochs']):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= args['max_batches'] // args['increment']:  # Stop after max_batches / increment
                break

            # Increment by 10 batches per training step
            for increment in range(0, args['increment'], 10):
                if batch_idx + increment < args['max_batches']:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    print(f'Client {client_id}, Epoch {epoch + 1}/{args["epochs"]}, Batch {batch_idx + increment + 1}, Loss: {loss.item():.4f}')
        
        print(f'Client {client_id}, Epoch {epoch + 1}/{args["epochs"]}, Average Loss: {epoch_loss / ((batch_idx + 1) * args["increment"]):.4f}')

# Function to test the model
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

# Dummy function for DDoS detection
def ddos_detection(model, device, ddos_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in ddos_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    average_loss = total_loss / len(ddos_loader)
    print(f'DDoS Detection Average Loss: {average_loss:.4f}')

# Federated learning process
def federated_learning(args, device, train_loader, test_loader, ddos_loader):
    global_model = DNN().to(device)
    global_weights = initialize_weights(global_model)

    for round in range(args['epochs']):
        local_weights = []

        for client in range(args['clients']):
            local_model = DNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=args['lr'])
            criterion = nn.CrossEntropyLoss()

            print(f'Training client {client + 1}/{args["clients"]} for round {round + 1}/{args["epochs"]}')
            train(local_model, device, train_loader, optimizer, criterion, client + 1)

            local_weights.append(local_model.state_dict())

        global_weights = aggregate_weights(local_weights)
        global_model.load_state_dict(global_weights)

        print(f'Round {round + 1}/{args["epochs"]} completed.')

        # Evaluate global model after aggregation
        test(global_model, device, test_loader)

        # Run DDoS detection
        ddos_detection(global_model, device, ddos_loader, criterion)

    return global_model

# Create a custom DataLoader for DDoS detection based on TabularDataset
def create_ddos_data_loader():
    # Reuse or create a smaller version of the original dataset
    data = pd.read_csv('network_traffic.csv')
    
    # Optionally filter or re-sample the data to simulate different characteristics of DDoS attacks
    ddos_data = data.sample(n=200)  # Select a smaller subset for DDoS detection, for example

    # Split features and labels
    X_ddos = ddos_data.drop('label', axis=1)
    y_ddos = ddos_data['label']

    # Standardize using the same scaler as training data (for consistency)
    scaler = StandardScaler()
    X_ddos = scaler.fit_transform(X_ddos)

    # Create a DataLoader for the DDoS detection data
    ddos_dataset = TabularDataset(X_ddos, y_ddos)
    ddos_loader = tud.DataLoader(ddos_dataset, batch_size=args['batch_size'], shuffle=True)
    
    return ddos_loader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data processing
train_loader, test_loader = create_data_loaders()

# Create the DDoS detection data loader
ddos_loader = create_ddos_data_loader()

# Run federated learning
global_model = federated_learning(args, device, train_loader, test_loader, ddos_loader)

# Test the global model
test(global_model, device, test_loader)
