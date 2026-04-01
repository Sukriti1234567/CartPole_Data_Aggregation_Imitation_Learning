import torch
import torch.nn as nn
import torch.nn.functional as function

# neural network for the behavioral cloning
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 32) 
        self.layer2 = nn.Linear(32,32)
        self.layer3 = nn.Linear(32,2)
    
    def forward(self, pass_):
        pass_ = function.relu(self.layer1(pass_))
        pass_ = function.relu(self.layer2(pass_))
        pass_ = self.layer3(pass_)
        return pass_ 
    