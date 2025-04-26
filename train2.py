import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from model import DQNAgent

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

      while not done:
          if random.random() < epsilon: # Exploration
            action = agent.get_action(obs, deterministic=False)
          else: # Exploitation
            action = agent.get_action(obs, deterministic=True)

          next_obs, reward, done, info = env.step(action)
          agent.replay_buffer.add(obs, action, reward, next_obs, done)
          
          if step % update_interval == 0:
            agent.train()

          total_reward += reward
          obs = next_obs
          step += 1
          print(f"\rStep: {step}, Reward: {total_reward}   ", end="", flush=True)
      
      epsilon = max(epsilon*decay_rate, min_epsilon)
      if (episode+1) % 10 == 0:
        agent.save()

      print(f"\rEpisode {episode}, Reward: {total_reward}, Step: {step}, Epsilon: {epsilon:.2f}", flush=True)
      reward_history.append(total_reward)

      if (episode+1)%10 == 0:
        plt.plot(reward_history)
        plt.savefig("train.png")

      # for i in range(n_stack):
      #   plt.subplot(1, n_stack, i+1)
      #   plt.imshow(obs[i])
      # plt.show()
      # print()


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
n_skip, n_stack = 4, 4
img_size = 84
env = SkipFrame(env, skip=n_skip)
env = GrayScaleObservation(env)
env = ResizeObservation(env, img_size)
env = FrameStack(env, n_stack)

state_size = env.observation_space.shape
print("state_size", state_size) # (4, 84, 84)
action_size = env.action_space.n
print("action size: ", action_size)

agent = DQNAgent(state_size, action_size, n_skip, batch_size=1024, lr=0.00025,
                 gamma=0.99, soft_tau=0.1, update_interval=32,
                 q_net_save_path="q_net2.pt", target_net_save_path="target_net2.pt")
train(env, agent, num_episodes=40000)


