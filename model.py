import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import random
from collections import deque

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
        img_size_div = img_size // 12 # ? 84 -> 7
        self.fc = nn.Sequential(
            nn.Linear(64 * (img_size_div**2), 512), # 3136
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    
class ReplayBuffer:
    def __init__(self, capacity):
      # TODO: Initialize the buffer
      self.buffer = deque(maxlen=capacity)

    # TODO: Implement the add method
    def add(self, item):
      self.buffer.append(item)

    # TODO: Implement the sample method
    def sample(self, n_item):
      return random.sample(self.buffer, n_item)

class DQNAgent:
    def __init__(self, state_size, action_size, device="cuda"):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        img_channels = 1 # due to grayscale
        self.img_size = state_size[1]
        self.q_net = QNet(img_channels, self.img_size, action_size).to(self.device)
        self.target_net = QNet(img_channels, self.img_size, action_size).to(self.device)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.001)
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.step = 0
        self.frame_stack = []

    def wrap_inputs(self, state):
        # State should be a stack of frames, with each sized (240, 256, 3)
        # i.e. (B, 240, 256, 3)
        data_transforms = transforms.Compose([
           transforms.Grayscale(),
           transforms.Resize((self.img_size, self.img_size)),
           transforms.Normalize(0, 255),
        ])
        state_tensor = torch.tensor(state.copy().astype(np.float32)).permute(0,3,1,2)  # -> (B, 3, 240, 256)
        state_tensor = data_transforms(state_tensor).to(self.device)
        return state_tensor

    def get_action(self, state, deterministic=True):
        state = self.wrap_inputs(state[None, ...]).to(self.device)
        if deterministic:
            with torch.no_grad():
                return torch.argmax(self.q_net(state)).item()
        else:
            return random.choice(list(range(self.action_size)))

    def update(self, target, learning):
        # TODO: Implement hard update or soft update
        alpha = 0.001
        for target_param, param in zip(target.parameters(), learning.parameters()):
          # target_param.data.copy_(param.data) # Hard update
          target_param.data.copy_(target_param.data * (1.0 - alpha) + param.data * alpha) # Soft update

    def train(self):
        # TODO: Sample a batch from the replay buffer
        batch_size = 64 #128
        if len(self.replay_buffer.buffer) < 10000: #batch_size:
          return
        batch = self.replay_buffer.sample(batch_size)
        batched_state, batched_action, batched_reward, batched_next_state, batched_done = zip(*batch)
        
        batched_state = self.wrap_inputs(np.stack(batched_state))
        batched_action = torch.tensor(batched_action, dtype=torch.int64).unsqueeze(1).to(self.device)
        batched_reward = torch.tensor(batched_reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        batched_next_state = self.wrap_inputs(np.stack(batched_next_state))
        batched_done = torch.tensor(batched_done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # TODO: Compute loss and update the model
        # training target y_i: r_t  + \gamma max_{a}{Q_{target_net}(s_{t+1}, a)}
        outputs = self.target_net(batched_next_state)
        batched_y = batched_reward + (1 - batched_done) * 0.99 * torch.max(outputs, dim=1)[0].unsqueeze(1) # (B, 1) + ((B, 4))

        # theta -= \alpha * 1/N * \sum_{i}^{N} {\nebla{q_net(s_i)[a_i](y_i - q_net(s_i)[a_i])}}
        batched_q = self.q_net(batched_state).gather(1, batched_action)

        self.optimizer.zero_grad()
        loss = self.loss_func(batched_q, batched_y)
        loss.backward()
        self.optimizer.step()

        # TODO: Update target network periodically
        self.step += 1
        if self.step % 3 == 0:
          self.update(self.target_net, self.q_net)


    def save(self):
       torch.save(self.q_net.state_dict(), "q_net.pt")
       torch.save(self.target_net.state_dict(), "target_net.pt")

    def load(self, q_net_path="q_net.pt", target_net_path="target_net.pt"):
       self.q_net.load_state_dict(torch.load(q_net_path))#, weights_only=True))
       self.target_net.load_state_dict(torch.load(target_net_path))#, weights_only=True))
