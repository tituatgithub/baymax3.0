import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # First linear layer: input_size -> hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        # Second linear layer: hidden_size -> hidden_size
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Output layer: hidden_size -> num_classes
        self.l3 = nn.Linear(hidden_size, num_classes)
        # ReLU activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First layer + ReLU
        out = self.l1(x)
        out = self.relu(out)
        # Second layer + ReLU
        out = self.l2(out)
        out = self.relu(out)
        # Output layer (no activation - will be handled by cross entropy loss)
        out = self.l3(out)
        return out
    