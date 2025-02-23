import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define constants
N = 5  # Number of drones
K = 3  # Number of obstacles
STATE_DIM_DRONE = 3  # 3D position for each drone
STATE_DIM_OBSTACLE = 3  # 3D position for each obstacle
ACTION_DIM = 3  # 3D action for each drone (e.g., velocity change)
D_MIN = 1.0  # Minimum safe distance
EPSILON_DRONE = 0.1  # Threshold for cumulative drone-drone collision risk
EPSILON_OBSTACLE = 0.1  # Threshold for cumulative obstacle-drone collision risk
GAMMA = 0.99  # Discount factor
LR_ACTOR = 1e-4  # Learning rate for actor (policy)
LR_CRITIC = 1e-3  # Learning rate for critic (value function)
LR_LAMBDA = 1e-2  # Learning rate for Lagrangian multipliers
EPISODE_LENGTH = 100  # Episode length
CLIP_EPSILON = 0.2  # PPO clipping parameter

# Define neural networks
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Continuous action space

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CostNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CostNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the algorithm
class HybridConstrainedPPO:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.cost_net = CostNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=LR_CRITIC)
        self.cost_optimizer = optim.Adam(self.cost_net.parameters(), lr=LR_CRITIC)
        self.lambda_short = torch.tensor(1.0, requires_grad=True)
        self.lambda_long = torch.tensor(1.0, requires_grad=True)
        self.lambda_optimizer = optim.Adam([self.lambda_short, self.lambda_long], lr=LR_LAMBDA)

    def compute_short_term_cost(self, states):
        # Compute immediate collision risk
        drone_positions = states[:, :N * STATE_DIM_DRONE].reshape(-1, N, STATE_DIM_DRONE)  # Shape: (batch_size, N, 3)
        obstacle_positions = states[:, N * STATE_DIM_DRONE:].reshape(-1, K, STATE_DIM_OBSTACLE)  # Shape: (batch_size, K, 3)

        # Drone-drone collision cost
        drone_drone_distances = torch.cdist(drone_positions, drone_positions)  # Shape: (batch_size, N, N)
        drone_drone_cost = torch.sum(torch.relu(D_MIN - drone_drone_distances)) / 2  # Avoid double counting

        # Obstacle-drone collision cost
        obstacle_drone_distances = torch.cdist(drone_positions, obstacle_positions)  # Shape: (batch_size, N, K)
        obstacle_drone_cost = torch.sum(torch.relu(D_MIN - obstacle_drone_distances))

        return drone_drone_cost + obstacle_drone_cost

    def compute_long_term_cost(self, states, actions):
        # Compute cumulative collision risk using the cost network
        return self.cost_net(states, actions)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute advantages
        values = self.value_net(states)
        next_values = self.value_net(next_states)
        advantages = rewards + GAMMA * (1 - dones) * next_values - values

        # Update value network
        value_loss = advantages.pow(2).mean()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Update cost network
        cost_predictions = self.compute_long_term_cost(states, actions)
        cost_targets = rewards + GAMMA * (1 - dones) * self.compute_long_term_cost(next_states, actions)
        cost_loss = (cost_predictions - cost_targets.detach()).pow(2).mean()
        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()

        # Update policy
        old_log_probs = torch.log(self.policy(states).gather(1, actions.unsqueeze(1)))
        new_log_probs = torch.log(self.policy(states).gather(1, actions.unsqueeze(1)))
        ratio = (new_log_probs - old_log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Add constraint terms
        short_term_cost = self.compute_short_term_cost(states)
        long_term_cost = self.compute_long_term_cost(states, actions)
        policy_loss += self.lambda_short * short_term_cost + self.lambda_long * (long_term_cost - EPSILON_OBSTACLE)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update Lagrangian multipliers
        self.lambda_short.data = torch.max(torch.tensor(0.0), self.lambda_short + LR_LAMBDA * short_term_cost)
        self.lambda_long.data = torch.max(torch.tensor(0.0), self.lambda_long + LR_LAMBDA * (long_term_cost - EPSILON_OBSTACLE))

# Example usage
state_dim = N * STATE_DIM_DRONE + K * STATE_DIM_OBSTACLE  # Total state dimension
action_dim = N * ACTION_DIM  # Total action dimension (actions for all drones)
agent = HybridConstrainedPPO(state_dim, action_dim)

# Simulate training loop
for episode in range(1000):
    states = np.random.randn(EPISODE_LENGTH, state_dim)  # Random states for illustration
    actions = np.random.randn(EPISODE_LENGTH, action_dim)  # Random actions for illustration
    rewards = np.random.randn(EPISODE_LENGTH)  # Random rewards for illustration
    next_states = np.random.randn(EPISODE_LENGTH, state_dim)  # Random next states for illustration
    dones = np.zeros(EPISODE_LENGTH)  # No terminal states for illustration

    agent.update(states, actions, rewards, next_states, dones)
