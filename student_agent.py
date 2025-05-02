import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np
import torch
from train_new import DQNAgent, QNet
import random

n_skip, n_stack = 4, 4
img_size = (84, 84)
state_size = (n_stack, *img_size) #env.observation_space.shape
print("state_size", state_size) # (4, 84, 84)
action_size = 12 #env.action_space.n
print("action size: ", action_size)


# state_dict = torch.load("mario_net_1.chkpt")["model"]
# new_state_dict = {}
# for key in state_dict.keys():
#     if "online" in key:
#         new_key = key.replace("online.", "")
#         if new_key.split(".")[0] in ["0", "1", "2", "3", "4"]:
#             new_state_dict["cnn."+new_key] = state_dict[key]
#         elif new_key.split(".")[0] in ["7", "9"]:
#             new_state_dict["fc."+str(eval(new_key.split(".")[0])-7)+"."+new_key.split(".")[1]] = state_dict[key]
# agent.q_net.load_state_dict(new_state_dict)

# Do not modify the input of the 'act' function and the '__init__' function. 

class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(action_size)

        self.frames = []
        self.repeat = 0 #n_skip-1
        self.last_action = 0

        self.agent = DQNAgent(state_size=state_size, action_size=action_size, device="cpu")
        self.agent.load(load_dir="new7", ckpt_name="ckpt.pt")
        self.agent.q_net.eval()

    @staticmethod
    def _preprocess(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, img_size, interpolation=cv2.INTER_AREA)
        return obs.astype(np.float32) / 255.0

    def _get_stacked_state(self):
        # skip and stack
        # stacked = np.stack(self.frames[-n_skip*n_stack::n_skip], axis=0)  # (4, 84, 84)
        stacked = np.stack(self.frames[-n_skip*(n_stack-1)-1::n_skip] + [self.frames[-1]], 0)[:-n_stack]
        stacked = stacked[..., np.newaxis]  # (4, 84, 84, 1)
        return stacked

    def act(self, observation):
        obs = np.array(observation, dtype=np.float32)
        obs = self._preprocess(obs)

        # Initialization
        if not self.frames:
            self.frames = [obs.copy()] * (n_skip * n_stack)
        else:
            self.frames.append(obs.copy())
            if len(self.frames) > n_skip * n_stack:
                self.frames.pop(0)

        # Skip frame: repeat last action for n_skip-1 times
        if self.repeat > 0:
            self.repeat -= 1
            return self.last_action

        print()

        if random.random() < 0:
            action = self.action_space.sample()
        else:
            stacked_state = self._get_stacked_state()
            assert stacked_state.shape == (n_stack, *img_size, 1), f"stacked_state shape: {stacked_state.shape}"
            action = self.agent.get_action(stacked_state, sample=False, logging=False)

        self.last_action = action
        self.repeat = n_skip - 1
        return action

    def act2(self, obs):
        action = self.agent.get_action(obs, sample=False, logging=False)
        return action
"""
    def act(self, observation):
        obs = np.array(observation, dtype=np.float32)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, img_size)
        obs /= 255.0

        if len(self.frames) == 0:
            self.frames = [obs.copy()] * (n_stack * n_skip)
        else:
            self.frames.append(obs)
            if len(self.frames) > n_stack * n_skip:
                self.frames.pop(0)

        # Skip frame extraction
        if len(self.frames) >= n_stack * n_skip:
            stacked = np.stack(self.frames[::n_skip], axis=0)
        else:
            stacked = np.stack([obs] * n_stack, axis=0)

        # inference noise
        if random.random() < 0.001:
            action = self.action_space.sample()
        else:
            action = agent.get_action(stacked, sample=False, logging=False)

        return action
"""