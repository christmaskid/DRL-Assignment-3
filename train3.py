from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import *

frames = []
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# state = env.reset()
# done = False
# for step in range(5000):
#     if done:
#         break
#     state, reward, done, info = env.step(env.action_space.sample())
#     print(reward, done, info, flush=True)
#     frames.append(state) #env.render()

# env.close()
# gif_path = "random_agent.gif"
# imageio.mimsave(gif_path, frames, fps=30)
# Image(filename=gif_path)

action_size = env.action_space.n
print("action size: ", action_size)
state_size = (4, 84, 84) #env.observation_space.shape[0]
print("state size: ", state_size)

agent = DQNAgent(state_size, action_size, batch_size=32, gamma=0.9, soft_tau=1, update_interval=3,
                 q_net_save_path="q_net3.pt", target_net_save_path="target_net3.pt")
# TODO: Determine the number of episodes for training
num_episodes = 40000

reward_history = [] # Store the total rewards for each episode
epsilon = 1
decay_rate = 0.99999975
min_epsilon = 0.1
n_skip = state_size[0]

for episode in tqdm(range(num_episodes)):
    # TODO: Reset the environment
    obs = env.reset()
    total_reward = 0
    done = False
    step = 0

    print()

    while not done:
        if random.random() < epsilon: # Exploration
          action = agent.get_action(obs, deterministic=False)
        else: # Exploitation
          action = agent.get_action(obs, deterministic=True)

        reward = 0
        for i in range(n_skip):
            next_obs, reward_, done, info = env.step(action)
            reward += reward_
            if done: break
            obs = next_obs
        
        agent.replay_buffer.add((obs, action, reward, next_obs, done))
        agent.train()

        # TODO: Update the state and total reward
        obs = next_obs
        total_reward += reward
        step += 1
        print(f"\rStep: {step}, Reward: {total_reward}   ", end="", flush=True)
    
    epsilon = max(epsilon*decay_rate, min_epsilon)
    agent.save()

    print(f"\rEpisode {episode}, Step: {step}, Reward: {total_reward}", flush=True)
    reward_history.append(total_reward)

    if (episode+1)%50 == 0:
      plt.plot(reward_history)
      plt.savefig("train3.png")
