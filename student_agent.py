import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np
import torch
from train_new_stage_eps import DQNAgent, QNet
import random

n_skip, n_stack = 4, 4
img_size = (84, 84)
state_size = (n_stack, *img_size) #env.observation_space.shape
action_size = 12 #env.action_space.n

# Do not modify the input of the 'act' function and the '__init__' function. 

class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(action_size)

        self.frames = []
        self.repeat = 0
        self.last_action = 0

        self.agent = DQNAgent(state_size=state_size, action_size=action_size, device="cpu")
        self.agent.load(load_dir="new8_2", ckpt_name="ckpt_7508.pt")
        self.agent.q_net.eval()

    @staticmethod
    def _preprocess(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, img_size, interpolation=cv2.INTER_AREA)
        return obs.astype(np.float32) / 255.0

    def _get_stacked_state(self):
        # skip and stack
        stacked = np.stack(self.frames[-n_skip*(n_stack-1)-1::n_skip])
        stacked = stacked[..., np.newaxis]  # (4, 84, 84, 1)
        return stacked

    # def act2(self, observation):
    #     action = self.agent.get_action(observation, sample=False)
    #     return action

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

        if random.random() < 0: #.001:
            action = self.action_space.sample()
        else:
            stacked_state = self._get_stacked_state()
            assert stacked_state.shape == (n_stack, *img_size, 1), f"stacked_state shape: {stacked_state.shape}"
            action = self.agent.get_action(stacked_state, sample=False, logging=False)

        self.last_action = action
        self.repeat = n_skip - 1
        return action
