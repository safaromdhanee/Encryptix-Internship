import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linearlayout1 = nn.Linear(input_size, hidden_size)  # First linear layout
        self.linearlayout2 = nn.Linear(hidden_size, hidden_size)  # Second linear layout
        self.linearlayout3 = nn.Linear(hidden_size, num_classes)  # Last layout for the number of classes
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linearlayout1(x)
        out = self.relu(out)
        out = self.linearlayout2(out)  # Pass 'out' instead of 'x' to the next layer
        out = self.relu(out)
        out = self.linearlayout3(out)
        return out
