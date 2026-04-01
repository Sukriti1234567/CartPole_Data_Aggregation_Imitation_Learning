import torch
import numpy as np
from cloning_network import Policy
import gymnasium as gym

#runs policy
def policy_evaluation(model, episodes, path):
    environment = gym.make("CartPole-v1")
    num_episodes=episodes
    total_reward =0
    model.load_state_dict(torch.load(path))
    model.eval()

    for episode in range(num_episodes):
            
        observation = environment.reset(options = {"low": -0.2,"high":0.2})
        state = observation[0]
        terminated = False
        truncated = False

        while truncated == False and terminated == False:
            tensor_state = torch.tensor(state, dtype = torch.float32).unsqueeze(dim = 0)
            logits = model(tensor_state)
            action = logits.argmax(dim=1).item()

            new_next_state, reward, terminated, truncated, info = environment.step(action)
            total_reward+=reward
            state = new_next_state
    print("average reward: ", total_reward/num_episodes)
     
    return total_reward/num_episodes

#In case I need to run this method for the behavioral cloning model:
model = Policy()
if __name__ == "__main__":
    policy_evaluation(model, 20, "policy_bc.pth")
