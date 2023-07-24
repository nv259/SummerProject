"""
PyTorch Deep Learning Model Life-Cycle:
1. Prepare the Data
2. Define the Model
3. Train the Model
4. Evaluate the Model
5. Make Predictions
"""

# Import dependencies
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset  # Base class for datasets, e.g. Image (torchvision),
                                      # Text (torchtext), audio (torchaudio)
from torchvision import datasets  # Standard datasets in Computer Vision
from torchvision import transforms  # Transformation we can perform on our dataset
from torch.utils.data import DataLoader  # Give easier dataset management and create mini_batch


""" 0. Specify device """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" 1. Prepare the Data """
# Dataset
train_dataset = datasets.MNIST(root='data/', train=True,
                               transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data/', train=False,
                              transform=transforms.ToTensor(), download=True)

# # Iterate through Dataset
# x, y = train_dataset[0]
# plt.imshow(x.squeeze())
# plt.show()

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# # Iterate through DataLoader
# X, y = next(iter(train_loader))
# print(f'Features batch size: {X.shape}')
# print(f'Labels batch size: {y.shape}')
# plt.imshow(X[0].squeeze())
# plt.show()

""" 2. Define the Model """
# Hyper-parameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size*sequence_length, num_classes),
                                nn.Softmax())

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagation
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

""" 3. Train the Model """
for epoch in range(num_epochs):
    for batch_idx, (X, y) in enumerate(train_loader):
        # Get data to CUDA if possible
        X = X.to(device).squeeze(1)
        y = y.to(device)

        # Forward propagation
        y_hat = model(X)
        loss = criterion(y_hat, y)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

    print(f'Epoch {epoch}: loss = {loss}')

""" 4. Evaluate the Model """
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on testing data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device).squeeze(1)
            y = y.to(device)

            y_hat = model(X)
            _, predictions = y_hat.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy '
              f'{float(num_correct) / float(num_samples) * 100.0:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
