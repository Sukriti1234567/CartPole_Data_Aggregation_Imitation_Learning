import torch
import numpy as np
from cloning_network import Policy
import gymnasium as gym
from evaluate_bc import policy_evaluation

model = Policy()
policy_evaluation(model, 20, "policy.pth")
