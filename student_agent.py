import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from model import *

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
state_size = (4, 84, 84) #env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size, device="cpu")
agent.load("q_net.pt", "target_net.pt")

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

    def act(self, observation):
        # return self.action_space.sample()
        action = agent.get_action(observation)
        return action