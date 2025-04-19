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

agent = DQNAgent(state_size, action_size)
# TODO: Determine the number of episodes for training
num_episodes = 40000

reward_history = [] # Store the total rewards for each episode
epsilon = 0.9
n_skip = 4

for episode in tqdm(range(num_episodes)):
    # TODO: Reset the environment
    obs = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done:
        if random.random() < epsilon: # Exploration
          action = agent.get_action(obs, deterministic=False)
        else: # Exploitation
          action = agent.get_action(obs, deterministic=True)

        reward = 0
        state_stack = []
        for i in range(n_skip):
            next_obs, reward_, done, info = env.step(action)
            if done: break
            reward += reward_
            state_stack.append(next_obs)
        agent.replay_buffer.add((obs, action, reward, next_obs, done))
        agent.train()

        # TODO: Update the state and total reward
        obs = next_obs
        total_reward += reward
        step += 1
    
    epsilon = max(epsilon*0.99, 0.01)
    agent.save()

    print(f"Episode {episode}, Reward: {total_reward}, Step: {step}", flush=True)
    reward_history.append(total_reward)

    if (episode+1)%10 == 0:
      plt.plot(reward_history)
      plt.savefig("train.png")
