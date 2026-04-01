from cloning_network import Policy
import torch
import numpy as np
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
import torch.nn as nn
import torch.optim as optim
from evaluate_bc import policy_evaluation
import matplotlib.pyplot as plt

#setting up
model= Policy()
model.load_state_dict(torch.load("policy.pth"))
model.eval()
environment = gym.make("CartPole-v1")
expert_data = PPO.load("ppo_expert_data")
ACTIONS_PATH = r"C:\Users\sehgs\OneDrive\Desktop\CartPole_Project_Dataset_Aggregation\actions.pt"
expert_actions = torch.load(ACTIONS_PATH)
expert_actions=expert_actions.long()
STATES_PATH = r"C:\Users\sehgs\OneDrive\Desktop\CartPole_Project_Dataset_Aggregation\states.pt"
expert_states = torch.load(STATES_PATH)
expert_states = expert_states.float()

plot_data = []

#carries out the Dataset Aggregation operation
def run_Dataset_Aggregation(x):
    global expert_actions, expert_states

    for cycle in range(x):
        states_new=[]

        num_episodes = 20
        for episode in range(num_episodes):

            observation = environment.reset()
            state = observation[0]
            terminated = False
            truncated = False

            while truncated == False and terminated == False:
                tensor_state = torch.tensor(state, dtype = torch.float32).unsqueeze(dim = 0)
                logits = model(tensor_state)
                action = logits.argmax(dim=1).item()

                new_next_state, reward, terminated, truncated, info = environment.step(action)
                state = new_next_state
                states_new.append(state)

        for new_state in states_new:
            expert_action = expert_data.predict(new_state)[0]
            new_expert_state = torch.from_numpy(new_state).unsqueeze(0)
            new_expert_action = torch.from_numpy(expert_action).unsqueeze(0)
            expert_actions= torch.cat((expert_actions, new_expert_action), dim =0)
            expert_states = torch.cat((expert_states,new_expert_state), dim =0)
                
        #retrain on new dataset
        num_epoch = 50
        loss_function = nn.CrossEntropyLoss()
        
        optimizer_function = optim.Adam(model.parameters(), lr = 0.003)
        for epoch in range( num_epoch):
            optimizer_function.zero_grad()
            logits = model(expert_states)
            loss_value = loss_function(logits, expert_actions)
            loss_value.backward()
            optimizer_function.step()

        #storing data for plotting
        torch.save(model.state_dict(), "policy.pth")
        plot_data.append(policy_evaluation(model, 20, "policy.pth"))

run_Dataset_Aggregation(10)

#plotting data
plt.plot(plot_data)
plt.xlabel("Data Aggregation Cycle")
plt.ylabel("Average Reward")
plt.title("Data Aggregation Learning Curve")
plt.savefig("data_aggregation_learning_curve.png")
plt.show()
