import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import pybullet_envs

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

# Define the multi-agent PPO algorithm
class MultiAgentPPO:
    def __init__(self, state_dim, action_dim, num_agents):
        self.num_agents = num_agents
        self.policies = [PolicyNetwork(state_dim, action_dim) for _ in range(num_agents)]
        self.value_nets = [ValueNetwork(state_dim) for _ in range(num_agents)]
        self.cost_net = CostNetwork(state_dim * num_agents, action_dim * num_agents)
        self.actor_optimizers = [optim.Adam(policy.parameters(), lr=LR_ACTOR) for policy in self.policies)]
        self.critic_optimizers = [optim.Adam(value_net.parameters(), lr=LR_CRITIC) for value_net in self.value_nets)]
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

        # Update each agent's policy and critic
        for i in range(self.num_agents):
            # Compute advantages
            values = self.value_nets[i](states[:, i * STATE_DIM_DRONE: (i + 1) * STATE_DIM_DRONE])
            next_values = self.value_nets[i](next_states[:, i * STATE_DIM_DRONE: (i + 1) * STATE_DIM_DRONE])
            advantages = rewards[:, i] + GAMMA * (1 - dones[:, i]) * next_values - values

            # Update value network
            value_loss = advantages.pow(2).mean()
            self.critic_optimizers[i].zero_grad()
            value_loss.backward()
            self.critic_optimizers[i].step()

            # Update policy
            old_log_probs = torch.log(self.policies[i](states[:, i * STATE_DIM_DRONE: (i + 1) * STATE_DIM_DRONE]).gather(1, actions[:, i * ACTION_DIM: (i + 1) * ACTION_DIM].unsqueeze(1)))
            new_log_probs = torch.log(self.policies[i](states[:, i * STATE_DIM_DRONE: (i + 1) * STATE_DIM_DRONE]).gather(1, actions[:, i * ACTION_DIM: (i + 1) * ACTION_DIM].unsqueeze(1)))
            ratio = (new_log_probs - old_log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Add constraint terms
            short_term_cost = self.compute_short_term_cost(states)
            long_term_cost = self.compute_long_term_cost(states, actions)
            policy_loss += self.lambda_short * short_term_cost + self.lambda_long * (long_term_cost - EPSILON_OBSTACLE)

            self.actor_optimizers[i].zero_grad()
            policy_loss.backward()
            self.actor_optimizers[i].step()

        # Update Lagrangian multipliers
        self.lambda_short.data = torch.max(torch.tensor(0.0), self.lambda_short + LR_LAMBDA * short_term_cost)
        self.lambda_long.data = torch.max(torch.tensor(0.0), self.lambda_long + LR_LAMBDA * (long_term_cost - EPSILON_OBSTACLE))

# Initialize PyBullet Gym environment
env = gym.make("DroneSwarm-v0")  # Replace with your PyBullet Gym environment
state_dim = env.observation_space.shape[0]  # State dimension from the environment
action_dim = env.action_space.shape[0]  # Action dimension from the environment

# Initialize the multi-agent PPO
agent = MultiAgentPPO(state_dim // N, action_dim // N, N)

# Training loop
for episode in range(1000):
    state = env.reset()
    episode_reward = np.zeros(N)

    for step in range(EPISODE_LENGTH):
        # Select actions for all agents
        state_tensor = torch.FloatTensor(state)
        actions = np.array([agent.policies[i](state_tensor[:, i * STATE_DIM_DRONE: (i + 1) * STATE_DIM_DRONE]).detach().numpy() for i in range(N)])

        # Take actions in the environment
        next_state, reward, done, info = env.step(actions.flatten())

        # Store experience
        agent.update([state], [actions.flatten()], [reward], [next_state], [done])

        # Update state and episode reward
        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}")

# Close the environment
env.close()
