import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class SimpleFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = tc.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x