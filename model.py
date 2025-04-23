import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import random
from collections import deque
import matplotlib.pyplot as plt

class QNet(torch.nn.Module):
    def __init__(self, img_channel, img_size, n_actions):
        super(QNet, self).__init__()
        # TODO: Define the neural network
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


class ReplayBuffer:
    def __init__(self, capacity, state_shape, device="cpu"):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states      = np.empty((capacity, *state_shape, 1), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape, 1), dtype=np.uint8)
        self.actions  = np.empty(capacity, dtype=np.int16)
        self.rewards  = np.empty(capacity, dtype=np.float32)
        self.dones    = np.empty(capacity, dtype=np.bool_)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = np.array(state)
        self.next_states[self.ptr] = np.array(next_state)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity # circular queue
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32, device=self.device).squeeze(-1) / 255.0,
            torch.tensor(self.actions[idx], dtype=torch.int64, device=self.device).unsqueeze(-1),
            torch.tensor(self.rewards[idx], dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.tensor(self.next_states[idx], dtype=torch.float32, device=self.device).squeeze(-1) / 255.0,
            torch.tensor(self.dones[idx], dtype=torch.int64, device=self.device).unsqueeze(-1),
        )

class DQNAgent:
    def __init__(self, state_size, action_size, n_skip=4, device="cuda", lr=0.00025,
                 batch_size=32, gamma=0.99, soft_tau=0.01, update_interval=3, 
                 buffer_size=1e5, min_training_buffer=1e4,
                 q_net_save_path="q_net.pt", target_net_save_path="target_net.pt"):

        self.state_size = state_size # (n_stack, img_width, img_height)
        self.action_size = action_size # output classes number
        self.device = device

        self.n_skip = n_skip # skip frames
        self.n_stack = state_size[0] # stack frames
        self.img_sizes = state_size[1:]
        self.q_net = QNet(self.n_stack, self.img_sizes, action_size).to(self.device)
        self.target_net = QNet(self.n_stack, self.img_sizes, action_size).to(self.device)

        self.loss_func = torch.nn.SmoothL1Loss() #MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.min_training_buffer = min_training_buffer 
        self.update_interval = update_interval
        self.sync_interval = 1e4
        self.buffer_size = int(buffer_size)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_size, self.device)

        self.q_net_save_path = q_net_save_path
        self.target_net_save_path = target_net_save_path


    def get_action(self, state, deterministic=True):
        state = torch.tensor(state, dtype=torch.float32).squeeze(-1).unsqueeze(0).to(self.device) / 255.0
        # (4, 84, 84, 1) -> (1, 4, 84, 84)
        if deterministic:
            # print(self.q_net(state), flush=True)
            with torch.no_grad():
                return torch.argmax(self.q_net(state)).item()
        else:
            return random.choice(list(range(self.action_size)))

    def update(self, target, learning):
        for target_param, param in zip(target.parameters(), learning.parameters()):
          target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau) # Soft update

    def train(self):
        if self.replay_buffer.size < self.min_training_buffer:
          return
        
        batch = self.replay_buffer.sample(self.batch_size)
        batched_states, batched_action, batched_reward, batched_next_states, batched_done = batch # zip(*batch)
        # print(f"batched_states: {batched_states.shape}, batched_action: {batched_action.shape}, batched_reward: {batched_reward.shape}, batched_next_states: {batched_next_states.shape}, batched_done: {batched_done.shape}")
        # print("batched_action values: ", batched_action)
        
        # training target y_i: r_t  + \gamma max_{a}{Q_{target_net}(s_{t+1}, a)}
        outputs = self.target_net(batched_next_states).detach()
        batched_y = batched_reward + (1 - batched_done) * self.gamma * torch.max(outputs, dim=1)[0].unsqueeze(1) # (B, 1) + ((B, 4))

        ## DOUBLE DQN
        # next_actions = self.q_net(batched_next_states).argmax(dim=1, keepdim=True)
        # next_q_values = self.target_net(batched_next_states).gather(1, next_actions)
        # batched_y = batched_reward + (1 - batched_done) * self.gamma * next_q_values

        # theta -= \alpha * 1/N * \sum_{i}^{N} {\nebla{q_net(s_i)[a_i](y_i - q_net(s_i)[a_i])}}
        batched_q = self.q_net(batched_states).gather(1, batched_action)

        # print("batched_q: ", batched_q, flush=True)
        # print("batched_y: ", batched_y, flush=True)
        self.optimizer.zero_grad()
        loss = self.loss_func(batched_q, batched_y)
        # print("loss: ", loss.item(), flush=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        # print("updated batched_q: ", self.q_net(batched_states).gather(1, batched_action), flush=True)


    def save(self):
       torch.save(self.q_net.state_dict(), self.q_net_save_path)
       torch.save(self.target_net.state_dict(), self.target_net_save_path)

    def load(self, q_net_path="q_net.pt", target_net_path="target_net.pt"):
      if self.device == "cpu":
        self.q_net.load_state_dict(torch.load(q_net_path, map_location=torch.device('cpu')))
        self.target_net.load_state_dict(torch.load(target_net_path, map_location=torch.device('cpu')))
      else:
        self.q_net.load_state_dict(torch.load(q_net_path, weights_only=True))
        self.target_net.load_state_dict(torch.load(target_net_path, weights_only=True))
