import torch
import torch.nn as nn
import torch.nn.functional as function
import torch.optim as optim
import numpy as np
import cloning_network
from cloning_network import Policy

#load expert data
ACTIONS_PATH = r"C:\Users\sehgs\OneDrive\Desktop\CartPole_Project_Dataset_Aggregation\actions.pt"
expert_actions = torch.load(ACTIONS_PATH)
expert_actions=expert_actions.long()
STATES_PATH = r"C:\Users\sehgs\OneDrive\Desktop\CartPole_Project_Dataset_Aggregation\states.pt"
expert_states = torch.load(STATES_PATH)
expert_states = expert_states.float()

#create model object
model = Policy()

#loss function
loss = nn.CrossEntropyLoss()

#opimizer
optimizer = optim.Adam(model.parameters(), lr =0.003)

#training loop
epoch = 0
num_epoch = 50
while epoch<=num_epoch:
    optimizer.zero_grad()
    logits = model(expert_states)
    loss_value = loss(logits, expert_actions)
    loss_value.backward()
    optimizer.step()
    if epoch%2 == 0:
        print(loss_value.item())
    epoch+=1

torch.save(model.state_dict(), "policy_bc.pth")
