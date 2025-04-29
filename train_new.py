import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ref.: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
class SkipFrameObservation(gym.Wrapper):
    def __init__(self, env, n_skip=4):
        super().__init__(env)
        self.env = env
        self.n_skip = n_skip
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.n_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class QNet(nn.Module):
    def __init__(self, img_channel, img_size, n_actions):
        super(QNet, self).__init__()
        # Reference: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        img_size_div_prod = (img_size[0] // 12) * (img_size[1] // 12) # ? 84 -> 7
        self.fc = nn.Sequential(
            nn.Linear(64 * (img_size_div_prod), 512), # 3136
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, device="cuda", save_dir="tmp"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.batch_size = 32
        self.capacity = 100000
        self.save_interval = 1
        self.gamma = 0.99
        self.learning_rate = 2.5e-4
        self.sync_interval = 10000
        self.learn_interval = 1
        
        self.q_net = QNet(state_size[0], state_size[1:], action_size).to(self.device)
        self.target_net = QNet(state_size[0], state_size[1:], action_size).to(self.device)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(self.capacity, device=torch.device("cpu"))
        )
        self.loss_func = torch.nn.SmoothL1Loss() # MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.learning_rate)

        self.current_step = 0

    def add_to_buffer(self, state, action, reward, next_state, done):
        # state and next_state are `LazyFrame`s
        state = torch.tensor(np.array(state), dtype=torch.float32).squeeze(-1)
        action = torch.tensor(action).unsqueeze(-1)
        reward = torch.tensor(reward).unsqueeze(-1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).squeeze(-1)
        done = torch.tensor(done).unsqueeze(-1)

        self.replay_buffer.add(TensorDict({
            "state": state, "action": action, "reward": reward, 
            "next_state": next_state, "done": done
        }, batch_size = []))
    

    def get_action(self, state, deterministic=True):
        if deterministic:
            state = torch.tensor(np.array(state), dtype=torch.float32).squeeze(-1).unsqueeze(0).to(self.device)
            values = self.q_net(state)
            action = torch.argmax(values, axis=1).item()
        else:
            action = np.random.choice(np.arange(self.action_size))
        
        return action

    def update_target_net(self):
        # Hard update
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self):
        torch.save({
            "q_net": self.q_net.state_dict(), 
            "target_net": self.target_net.state_dict(),
            "curr_step": self.current_step
            },
            os.path.join(self.save_dir,f"ckpt.pt")
        )
        if self.current_step % 1000 == 0:
            torch.save({
                "q_net": self.q_net.state_dict(), 
                "target_net": self.target_net.state_dict(),
                "curr_step": self.current_step
                },
                os.path.join(self.save_dir,f"ckpt_copy.pt")
            )
        
    def load(self, ckpt_name=None):
        if ckpt_name is None:
            ckpt_name = sorted(os.listdir(self.save_dir))[-1]
        state_dict = torch.load(os.path.join(self.save_dir, ckpt_name))
        self.q_net.load_state_dict(state_dict["q_net"])
        self.target_net.load_state_dict(state_dict["target_net"])
        self.current_step = state_dict["curr_step"]


    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return np.NaN, np.NaN
            
        self.current_step += 1
        if self.current_step % self.learn_interval != 0:
            return np.NaN, np.NaN
        
        batch = self.replay_buffer.sample(self.batch_size).to(self.device)
        batched_state, batched_action, batched_reward, batched_next_state, batched_done \
            = batch["state"], batch["action"], batch["reward"], batch["next_state"], batch["done"]
        # for key in batch.keys():
        #     print(f"{key}: {batch[key].shape}", flush=True)
        
        # Double DQN        
        with torch.no_grad():
            next_actions = self.q_net(batched_next_state).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(batched_next_state).gather(1, next_actions)
            batched_y = batched_reward + ((1 - batched_done.float()) * self.gamma * next_q_values).float()

        batched_q = self.q_net(batched_state).gather(1, batched_action)

        loss = self.loss_func(batched_q, batched_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.current_step % self.sync_interval == 0:
            self.update_target_net()
        # if self.current_step % self.save_interval == 0:
        #     self.save()

        return batched_q.mean().item(), loss.item()


def train(env, agent, 
        epsilon=1, epsilon_decay_rate=0.99999975, epsilon_min=0.001,
        start_episode=0, num_episodes=40000):
    
    epsilon *= (epsilon_decay_rate ** start_episode)
    reward_history = []

    for episode in tqdm(range(start_episode, num_episodes)):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        losses = []
        print()

        while not done:
            if np.random.random() < epsilon:
                action = agent.get_action(state, deterministic=False)
            else:
                action = agent.get_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            agent.add_to_buffer(state, action, reward, next_state, done)
            q_avg, loss = agent.train()
            total_reward += reward
            state = next_state
            step += 1
            if done:
                break
            
            print(f"\rStep: {step}, Total reward: {total_reward}; Average Q: {q_avg:.2f}, Loss: {loss:.2f}", end="", flush=True)
            losses.append(loss)
        
        print(f"\rEpisode {episode}, Reward: {total_reward}, Step: {step}, Epsilon: {epsilon}")
        print(f"Average Q: {q_avg:.2f}, Loss: {np.mean(losses):.2f}", flush=True)
        
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

        reward_history.append(total_reward)
        agent.save()
        
        if (episode + 1) % 10 == 0:
            plt.plot(reward_history)
            plt.savefig(os.path.join(agent.save_dir, "history.png"))
            plt.close()


if __name__=="__main__":
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    img_shape = (84, 84)
    n_stack = 4
    n_skip = 4

    # Ref.: https://www.gymlibrary.dev/api/wrappers/#available-wrappers
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=img_shape)
    env = FrameStack(env, num_stack=n_stack)
    env = SkipFrameObservation(env, n_skip=n_skip)
    
    state_size = env.observation_space.shape
    print("state_size", state_size) # (4, 84, 84)
    action_size = env.action_space.n
    print("action_size", action_size, flush=True)
    
    # agent = DQNAgent(state_size=state_size, action_size=action_size, save_dir="tmp")
    # train(env, agent, epsilon=0.9, epsilon_decay_rate=0.999, epsilon_min=0.001)
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, save_dir="new3")
    train(env, agent, epsilon=1, epsilon_decay_rate=0.999975, epsilon_min=0.001)
    
