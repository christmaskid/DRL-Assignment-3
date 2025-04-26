import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np
import torch
from model import DQNAgent
import random

n_skip, n_stack = 8, 4
img_size = (84, 84)
state_size = (n_stack, *img_size) #env.observation_space.shape
print("state_size", state_size) # (4, 84, 84)
action_size = 12 #env.action_space.n
print("action size: ", action_size)

agent = DQNAgent(state_size, action_size, n_skip, batch_size=32, 
                 gamma=0.99, soft_tau=0.1, update_interval=32, device="cpu")
agent.load("q_net_debug_1e-5_32_new.pt", "target_net_debug_1e-5_32_new.pt")
# agent.load("q_net__2e-5_32.pt", "target_net__2e-5_32.pt")
agent.q_net.eval()
agent.target_net.eval()

# Do not modify the input of the 'act' function and the '__init__' function. 
frames= []

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

    def act(self, observation):
        if random.random() < 0.1: # Exploration
            return self.action_space.sample()

        obs = np.array(observation)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) # grayscale
        obs = cv2.resize(obs, img_size) # resize

        if len(frames) == 0:
            obs = np.stack([obs] * n_stack) # frame stack
        else:
            obs = np.stack(frames[-n_stack:] + [obs])
        obs = obs.astype(np.float32) / 255.0
        action = agent.get_action(obs, deterministic=True)
        # print("action: ", action, flush=True)
        return action