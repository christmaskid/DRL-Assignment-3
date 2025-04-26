import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from model import DQNAgent
import torch
import numpy as np

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward, done, info = 0, False, {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def train(env, agent, num_episodes, start_episode=0, update_interval=32, epsilon = 0.9, 
          decay_rate = 0.99999975, min_epsilon = 0.1):
  
  reward_history = [] # Store the total rewards for each episode
  for episode in tqdm(range(start_episode, num_episodes)):
      obs = env.reset() # (4, 84, 84, 1)
      total_reward = 0
      done = False
      step = 0
      print()

      while not done:
          if random.random() < epsilon: # Exploration
            action = agent.get_action(obs, deterministic=False)
          else: # Exploitation
            action = agent.get_action(obs, deterministic=True)

          next_obs, reward, done, info = env.step(action)
          agent.replay_buffer.add(obs, action, reward, next_obs, done)
          
          if step % update_interval == 0:
            state = torch.tensor(obs, dtype=torch.float32).squeeze(-1).unsqueeze(0).to(agent.device) / 255.0
            # print("Before: ", agent.q_net(state), action, reward, done, flush=True)
            agent.train()
            agent.update(agent.target_net, agent.q_net)
            # print("After: ", agent.q_net(state), flush=True)
            
          # if step % 1024 == 0:
          #   print("obs: ", obs[0].shape, obs[0].dtype, obs[0].min(), obs[0].max())
          #   print("next_obs: ", next_obs[0].shape, next_obs[0].dtype, next_obs[0].min(), next_obs[0].max())
          #   print("action: ", action, "reward: ", reward, "done: ", done, "info: ", info)
          #   plt.subplot(1, 2, 1)
          #   plt.imshow(obs[0], cmap='gray')
          #   plt.title("obs")
          #   plt.subplot(1, 2, 2)
          #   plt.imshow(next_obs[0], cmap='gray')
          #   plt.title("next_obs")
          #   plt.show()

          total_reward += reward
          obs = next_obs
          step += 1
          print(f"\rStep: {step}, Reward: {total_reward}   ", end="", flush=True)
      
      epsilon = max(epsilon*decay_rate, min_epsilon)

      if (episode+1) % 3 == 0:
        agent.target_net.load_state_dict(agent.q_net.state_dict()) # Hard update
        agent.save()

      print(f"\rEpisode {episode}, Reward: {total_reward}, Step: {step}, Epsilon: {epsilon:.2f}", flush=True)
      reward_history.append(total_reward)

      if (episode+1)%10 == 0:
        plt.plot(reward_history)
        a = [np.mean(reward_history[i-min(i, 100):i-min(i, 100)+100]) for i in range(len(reward_history))]
        plt.plot(a, label="smoothed")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.savefig("train_debug_1e-4_32_new.png")
        plt.close()

      # for i in range(n_stack):
      #   plt.subplot(1, n_stack, i+1)
      #   plt.imshow(obs[i])
      # plt.show()
      # print()


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
n_skip, n_stack = 8, 4
img_size = 84
env = SkipFrame(env, skip=n_skip)
env = GrayScaleObservation(env)
env = ResizeObservation(env, img_size)
env = FrameStack(env, n_stack)

state_size = env.observation_space.shape
print("state_size", state_size) # (4, 84, 84)
action_size = env.action_space.n
print("action size: ", action_size)

""" 1
agent = DQNAgent(state_size, action_size, n_skip, batch_size=1024, 
                 gamma=0.99, soft_tau=0.1, update_interval=32, lr=1e-5,
                 buffer_size=1e5, min_training_buffer=1e3,
                 q_net_save_path="q_net_debug.pt", target_net_save_path="target_net_debug.pt")
agent.load("q_net_debug.pt", "target_net_debug.pt")
train(env, agent,  start_episode=2709, num_episodes=40000, epsilon = 0.9 * (0.999 ** 2709),
      decay_rate=0.999, min_epsilon=0.01,)
"""

""" 2, 3
agent = DQNAgent(state_size, action_size, n_skip, batch_size=32, 
                 gamma=0.99, soft_tau=0.1, update_interval=32, lr=1e-3,
                 buffer_size=1e4, min_training_buffer=1e2,
                 q_net_save_path="q_net_debug_1e-3_32.pt", target_net_save_path="target_net_debug_1e-3_32.pt")
# agent.load("q_net_8-4_099.pt", "target_net_8-4_099.pt")
train(env, agent, num_episodes=40000, decay_rate=0.99, min_epsilon=0.01)
"""

# """ 4
agent = DQNAgent(state_size, action_size, n_skip, batch_size=32, 
                 gamma=0.99, soft_tau=0.1, update_interval=32, lr=1e-5,
                 buffer_size=1e5, min_training_buffer=1e4,
                 q_net_save_path="q_net_debug_1e-5_32_new.pt", target_net_save_path="target_net_debug_1e-5_32_new.pt")
agent.load("q_net_debug_1e-5_32_new.pt", "target_net_debug_1e-5_32_new.pt")
train(env, agent, start_episode=1553, epsilon=0.9*(0.999975**1552), num_episodes=40000, decay_rate=0.999975, min_epsilon=0.01)
# """

""" 5
agent = DQNAgent(state_size, action_size, n_skip, batch_size=32, 
                 gamma=0.99, soft_tau=0.1, update_interval=32, lr=1e-5,
                 buffer_size=1e5, min_training_buffer=1e4,
                 q_net_save_path="q_net_debug_1e-4_32_new.pt", target_net_save_path="target_net_debug_1e-4_32_new.pt")
# agent.load("q_net_8-4_099.pt", "target_net_8-4_099.pt")
train(env, agent, num_episodes=40000, decay_rate=0.999975, min_epsilon=0.01)
"""