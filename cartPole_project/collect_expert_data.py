# trains PPO and collects expert data for behavioral cloning

import gymnasium
from stable_baselines3 import PPO
import numpy as np
import torch

env = gymnasium.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose =1)
model.learn(50000)
model.save("ppo_expert_data")

#outer loop
num_epochs = 50
states = []
actions = []
reward_total =0
 
for epoch in range (num_epochs):
    #inner loop
    truncated = False
    terminated = False
    state = env.reset()[0]

    while truncated == False and terminated == False:
        states.append(state)
        action = model.predict(state)[0]
        actions.append(action)
        state, reward, terminated, truncated, other = env.step(action)
        reward_total += reward

#saving the data
x_states = np.array(states)
y_states = torch.tensor(x_states, dtype=torch.float32)
x_actions = np.array(actions)
y_actions = torch.tensor(x_actions, dtype=torch.long)

torch.save(y_states, "states.pt")
torch.save(y_actions, "actions.pt")
