import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

#  Environment
env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)

#  Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Neural Network
class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(inputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, outputs)

 # Called with either one element to determine next action, or a batch
# during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


# Hyperparameters
BATCH_SIZE = 64 #128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2000
TARGET_UPDATE = 10
LR = 1e-3
MEMORY_CAPACITY = 10000

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]


# Model & Target Network
policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0


#epsilon‑Greedy Policy
def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() < eps_threshold:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
    with torch.no_grad():
        return policy_net(state).argmax(dim=1).view(1, 1)

#trin loop
#optimize Function
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    if non_final_next_states.size(0) != 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_q_values = (next_state_values * GAMMA) + reward_batch

    loss = nn.functional.smooth_l1_loss(q_values.squeeze(), expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Training Loop
num_episodes = 300
episode_rewards = []

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor([state], dtype=torch.float32)
    total_reward = 0.0

    while True:
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        reward = torch.tensor([reward], dtype=torch.float32)

        if not done:
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
        else:
            next_state_tensor = None

        memory.push(state, action, next_state_tensor, reward)
        state = next_state_tensor if next_state_tensor is not None else state

        optimize_model()

        if done:
            episode_rewards.append(total_reward)
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (i_episode+1) % 50 == 0:
        print(f"Episode {i_episode+1}, Reward: {total_reward:.2f}")

# Final target network update
target_net.load_state_dict(policy_net.state_dict())

#  Plot Reward Over Episodes
plt.figure(figsize=(10,6))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Reward")
plt.grid(True)
plt.show()
