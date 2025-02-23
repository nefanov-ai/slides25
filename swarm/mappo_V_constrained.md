To add **V-function constraints** to the **MAPPO with a centralized critic**, we need to modify the training loop to include a **Lagrangian multiplier** for the safety constraint. The V-function constraint ensures that the expected cumulative cost (e.g., time spent in unsafe states) does not exceed a predefined threshold. Below is the updated implementation with **V-function constraints** for safe RL.

---

### **Code Implementation**

#### **1. Custom PyBullet Gym Environment**
The environment remains the same as before, simulating the swarm of drones, obstacles, and the target location.

```python
import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class SwarmDroneEnv(gym.Env):
    def __init__(self, num_drones=5, num_obstacles=3, target=(10, 10, 5), min_distance=2, dt=0.1):
        super(SwarmDroneEnv, self).__init__()
        self.num_drones = num_drones
        self.num_obstacles = num_obstacles
        self.target = np.array(target)
        self.min_distance = min_distance
        self.dt = dt

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_drones * 3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_drones * 6 + num_obstacles * 3 + 3,), dtype=np.float32)

        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)  # Use p.DIRECT for headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load drones
        self.drone_ids = []
        for _ in range(num_drones):
            drone_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 1])
            self.drone_ids.append(drone_id)

        # Load obstacles
        self.obstacle_ids = []
        for _ in range(num_obstacles):
            obstacle_id = p.loadURDF("sphere_small.urdf", basePosition=[3, 3, 1])
            self.obstacle_ids.append(obstacle_id)

    def reset(self):
        # Reset drone positions and velocities
        for i, drone_id in enumerate(self.drone_ids):
            p.resetBasePositionAndOrientation(drone_id, [i, i, 1], [0, 0, 0, 1])
            p.resetBaseVelocity(drone_id, [0, 0, 0], [0, 0, 0])
        return self._get_obs()

    def _get_obs(self):
        # Get drone positions and velocities
        drone_positions = []
        drone_velocities = []
        for drone_id in self.drone_ids:
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            vel, _ = p.getBaseVelocity(drone_id)
            drone_positions.append(pos)
            drone_velocities.append(vel)

        # Get obstacle positions
        obstacle_positions = []
        for obstacle_id in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            obstacle_positions.append(pos)

        # Concatenate observations
        obs = np.concatenate([
            np.array(drone_positions).flatten(),
            np.array(drone_velocities).flatten(),
            np.array(obstacle_positions).flatten(),
            self.target.flatten()
        ])
        return obs

    def step(self, actions):
        # Reshape actions to (num_drones, 3)
        actions = actions.reshape(self.num_drones, 3)

        # Apply actions to drones
        for i, drone_id in enumerate(self.drone_ids):
            p.applyExternalForce(drone_id, -1, actions[i], [0, 0, 0], p.WORLD_FRAME)

        # Step simulation
        p.stepSimulation()

        # Compute rewards and costs
        rewards = np.zeros(self.num_drones)
        costs = np.zeros(self.num_drones)

        for i, drone_id in enumerate(self.drone_ids):
            pos, _ = p.getBasePositionAndOrientation(drone_id)

            # Reward for reaching the target
            distance_to_target = np.linalg.norm(pos - self.target)
            rewards[i] += -distance_to_target

            # Penalty for violating minimum distance
            for j, other_drone_id in enumerate(self.drone_ids):
                if i != j:
                    other_pos, _ = p.getBasePositionAndOrientation(other_drone_id)
                    distance = np.linalg.norm(pos - other_pos)
                    if distance < self.min_distance:
                        rewards[i] -= 10 * (self.min_distance - distance)
                        costs[i] += 1

            # Penalty for colliding with obstacles
            for obstacle_id in self.obstacle_ids:
                obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
                distance = np.linalg.norm(pos - obstacle_pos)
                if distance < 1:  # Assuming obstacle radius is 1
                    rewards[i] -= 100
                    costs[i] += 1

        # Check if all drones have reached the target
        done = all(np.linalg.norm(p.getBasePositionAndOrientation(drone_id)[0] - self.target) < 1 for drone_id in self.drone_ids)

        return self._get_obs(), rewards.sum(), done, {"costs": costs.sum()}

    def close(self):
        p.disconnect()
```

---

#### **2. MAPPO with Centralized Critic and V-function Constraints**
We implement MAPPO with a **centralized critic** and **V-function constraints** for safe RL. The V-function constraint is enforced using a **Lagrangian multiplier**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(CentralizedCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.critic(state)

class MAPPO:
    def __init__(self, env, num_drones, policy_kwargs=None, **kwargs):
        self.num_drones = num_drones
        self.env = env
        self.actors = [PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **kwargs) for _ in range(num_drones)]
        self.critic = CentralizedCritic(env.observation_space.shape[0])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=kwargs.get("learning_rate", 3e-4))
        self.lagrangian_multiplier = torch.tensor(1.0, requires_grad=True)
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_multiplier], lr=1e-4)
        self.cost_threshold = kwargs.get("cost_threshold", 0.1)
        self.gamma = kwargs.get("gamma", 0.99)

    def learn(self, total_timesteps, callback=None):
        for model in self.actors:
            model.learn(total_timesteps=total_timesteps // self.num_drones, callback=callback)

    def predict(self, obs, deterministic=True):
        actions = []
        for i, model in enumerate(self.actors):
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action)
        return np.array(actions).flatten()

    def update_critic(self, states, rewards, next_states, dones, costs):
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        costs = torch.FloatTensor(costs)

        # Compute value estimates
        values = self.critic(states)
        next_values = self.critic(next_states)

        # Compute advantages
        advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # Compute cost advantages
        cost_values = self.critic(states)
        next_cost_values = self.critic(next_states)
        cost_advantages = costs + self.gamma * next_cost_values * (1 - dones) - cost_values

        # Update critic
        critic_loss = advantages.pow(2).mean() + cost_advantages.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Lagrangian multiplier
        constraint_violation = torch.relu(cost_advantages.mean() - self.cost_threshold)
        lagrangian_loss = -self.lagrangian_multiplier * constraint_violation
        self.lagrangian_optimizer.zero_grad()
        lagrangian_loss.backward()
        self.lagrangian_optimizer.step()

# Create the environment
env = SwarmDroneEnv()

# Wrap the environment for vectorized training (optional)
env = make_vec_env(lambda: env, n_envs=4)

# Initialize MAPPO
mappo = MAPPO(
    env,
    num_drones=5,
    policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    cost_threshold=0.1,
    tensorboard_log="./swarm_drone_tensorboard/"
)

# Add evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./swarm_drone_best_model/",
    log_path="./swarm_drone_eval_logs/",
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Train MAPPO
mappo.learn(total_timesteps=1_000_000, callback=eval_callback)

# Save the models
for i, model in enumerate(mappo.actors):
    model.save(f"swarm_drone_mappo_{i}")
```

---

#### **3. Testing the Trained Model**
After training, you can test the trained model to see how the swarm of drones performs.

```python
# Load the trained models
mappo = MAPPO(env, num_drones=5)
for i in range(5):
    mappo.actors[i] = PPO.load(f"swarm_drone_mappo_{i}")

# Test the model
obs = env.reset()
for _ in range(1000):
    action = mappo.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    print(f"Reward: {rewards}, Costs: {info['costs']}")
    if dones:
        obs = env.reset()
```

---

### **Key Features**
- **Centralized Critic**: All agents share a single critic network that evaluates the value function \( V(s) \) for the global state \( s \).
- **V-function Constraints**: Ensures state-level safety by constraining the expected cumulative cost using a Lagrangian multiplier.
- **Custom Environment**: Simulates a swarm of drones navigating to a target while avoiding obstacles and maintaining a minimum distance.

---

### **Notes**
- The **Lagrangian multiplier** is used to enforce the V-function constraint by penalizing violations of the cost threshold.
- The **cost threshold** \( \tau \) is a hyperparameter that controls the trade-off between performance and safety.
- You can further improve the implementation by:
  - Adding more sophisticated collision avoidance mechanisms.
  - Using parallelized training with multiple environments.
  - Tuning hyperparameters for better performance.

---
Formal description:

Below is a **formal description** of the **Multi-Agent Proximal Policy Optimization (MAPPO)** algorithm with **V-function constraints** for **safe reinforcement learning (RL)**. This algorithm extends the standard PPO algorithm to a multi-agent setting with a **centralized critic** and enforces safety constraints using a **Lagrangian multiplier**.

---

### **Algorithm: MAPPO with V-function Constraints**

#### **Inputs**
- \( N \): Number of agents (drones).
- \( \mathcal{S} \): State space (global state including all agents' states, obstacles, and target).
- \( \mathcal{A} \): Action space (individual actions for each agent).
- \( \gamma \): Discount factor (\( 0 \leq \gamma < 1 \)).
- \( \tau \): Safety threshold for the expected cumulative cost.
- \( \alpha \): Learning rate for the policy and critic networks.
- \( \alpha_\lambda \): Learning rate for the Lagrangian multiplier.
- \( K \): Number of epochs for policy optimization.
- \( T \): Maximum number of timesteps per episode.

#### **Parameters**
- \( \pi_i \): Policy for agent \( i \) (parameterized by \( \theta_i \)).
- \( V \): Centralized critic (parameterized by \( \phi \)).
- \( \lambda \): Lagrangian multiplier for the safety constraint.

#### **Algorithm Steps**

1. **Initialize**:
   - Initialize policies \( \pi_i \) for each agent \( i \).
   - Initialize centralized critic \( V \).
   - Initialize Lagrangian multiplier \( \lambda = 1.0 \).

2. **For each episode**:
   - Reset the environment and observe the initial state \( s_0 \).
   - Initialize empty buffers for states, actions, rewards, costs, and next states.

3. **For each timestep \( t = 0, 1, \dots, T-1 \)**:
   - For each agent \( i \), sample action \( a_t^i \sim \pi_i(\cdot | s_t) \).
   - Execute joint action \( a_t = (a_t^1, \dots, a_t^N) \) in the environment.
   - Observe next state \( s_{t+1} \), rewards \( r_t \), and costs \( c_t \).
   - Store \( (s_t, a_t, r_t, c_t, s_{t+1}) \) in the buffers.

4. **Compute advantages and cost advantages**:
   - For each timestep \( t \), compute the value function \( V(s_t) \) and \( V(s_{t+1}) \).
   - Compute the advantage \( A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \).
   - Compute the cost advantage \( A_t^c = c_t + \gamma V(s_{t+1}) - V(s_t) \).

5. **Update centralized critic**:
   - Minimize the critic loss:
     \[
     \mathcal{L}_V(\phi) = \mathbb{E} \left[ \left( r_t + \gamma V(s_{t+1}) - V(s_t) \right)^2 \right].
     \]
   - Update \( \phi \) using gradient descent:
     \[
     \phi \leftarrow \phi - \alpha \nabla_\phi \mathcal{L}_V(\phi).
     \]

6. **Update policies with V-function constraints**:
   - For each agent \( i \), compute the policy loss:
     \[
     \mathcal{L}_{\pi_i}(\theta_i) = -\mathbb{E} \left[ \min \left( \frac{\pi_i(a_t^i | s_t)}{\pi_i^{\text{old}}(a_t^i | s_t)} A_t, \text{clip} \left( \frac{\pi_i(a_t^i | s_t)}{\pi_i^{\text{old}}(a_t^i | s_t)}, 1 - \epsilon, 1 + \epsilon \right) A_t \right) \right].
     \]
   - Add the safety constraint using the Lagrangian multiplier:
     \[
     \mathcal{L}_{\pi_i}(\theta_i) \leftarrow \mathcal{L}_{\pi_i}(\theta_i) + \lambda \cdot \max \left( 0, \mathbb{E} \left[ A_t^c \right] - \tau \right).
     \]
   - Update \( \theta_i \) using gradient ascent:
     \[
     \theta_i \leftarrow \theta_i + \alpha \nabla_{\theta_i} \mathcal{L}_{\pi_i}(\theta_i).
     \]

7. **Update Lagrangian multiplier**:
   - Compute the constraint violation:
     \[
     \text{violation} = \max \left( 0, \mathbb{E} \left[ A_t^c \right] - \tau \right).
     \]
   - Update \( \lambda \) using gradient ascent:
     \[
     \lambda \leftarrow \lambda + \alpha_\lambda \cdot \text{violation}.
     \]

8. **Repeat**:
   - Repeat steps 2–7 until convergence or a predefined number of episodes.

---

### **Key Components**

1. **Centralized Critic**:
   - The critic \( V(s) \) evaluates the value of the global state \( s \), which includes the states of all agents, obstacles, and the target.
   - It is shared across all agents and updated using the Bellman equation.

2. **V-function Constraints**:
   - The expected cumulative cost \( \mathbb{E} \left[ A_t^c \right] \) is constrained to be below the safety threshold \( \tau \).
   - The Lagrangian multiplier \( \lambda \) enforces this constraint by penalizing violations.

3. **Policy Optimization**:
   - Each agent's policy \( \pi_i \) is updated using the PPO objective, which includes a clipped surrogate objective and the safety constraint.

4. **Lagrangian Multiplier**:
   - The Lagrangian multiplier \( \lambda \) is updated to balance the trade-off between maximizing rewards and satisfying the safety constraint.

---

### **Output**
- Trained policies \( \pi_i \) for each agent \( i \).
- Centralized critic \( V(s) \) for evaluating the value of global states.

---

### **Advantages**
- **Centralized Critic**: Enables efficient value estimation for the global state, improving coordination among agents.
- **V-function Constraints**: Ensures safety by constraining the expected cumulative cost.
- **Scalability**: Can be extended to large-scale multi-agent systems.

---

### **Applications**
- **Swarm Robotics**: Coordinated control of drones, robots, or autonomous vehicles.
- **Safe RL**: Tasks requiring safety guarantees, such as collision avoidance or risk-sensitive control.

---
Specialization:

In the context of a **swarm of drones** tasked with **collision avoidance** (both between drones and with obstacles), the **V-function constraints** can be used to enforce safety by ensuring that the expected cumulative cost of collisions or near-collisions remains below a predefined threshold. Below is an example of how these constraints are applied in the swarm drone control task.

---

### **Example: V-function Constraints for Collision Avoidance**

#### **1. Problem Setup**
- **Agents**: \( N \) drones.
- **State Space (\( \mathcal{S} \))**: Includes the positions and velocities of all drones, positions of obstacles, and the target location.
- **Action Space (\( \mathcal{A} \))**: Thrust and orientation changes for each drone.
- **Reward Function (\( r(s, a) \))**:
  - Positive reward for reaching the target.
  - Negative reward for collisions or near-collisions.
- **Cost Function (\( c(s, a) \))**:
  - Penalizes collisions or near-collisions between drones and with obstacles.

#### **2. Cost Function Definition**
The cost function \( c(s, a) \) quantifies safety violations:
- **Drone-Drone Collisions**:
  \[
  c_{\text{drones}}(s, a) = \sum_{i=1}^N \sum_{j \neq i} \mathbb{I} \left( \| p_i - p_j \| < d_{\text{safe}} \right),
  \]
  where:
  - \( p_i \) and \( p_j \) are the positions of drones \( i \) and \( j \).
  - \( d_{\text{safe}}} \) is the minimum safe distance between drones.
  - \( \mathbb{I}(\cdot) \) is an indicator function that returns 1 if the condition is true, otherwise 0.

- **Drone-Obstacle Collisions**:
  \[
  c_{\text{obstacles}}(s, a) = \sum_{i=1}^N \sum_{k=1}^M \mathbb{I} \left( \| p_i - o_k \| < r_k + d_{\text{safe}}} \right),
  \]
  where:
  - \( o_k \) is the position of obstacle \( k \).
  - \( r_k \) is the radius of obstacle \( k \).
  - \( d_{\text{safe}}} \) is the minimum safe distance from obstacles.

- **Total Cost**:
  \[
  c(s, a) = c_{\text{drones}}(s, a) + c_{\text{obstacles}}(s, a).
  \]

#### **3. V-function Constraint**
The **V-function** \( V(s) \) estimates the expected cumulative cost starting from state \( s \):
\[
V(s) = \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t c(s_t, a_t) \,\bigg|\, s_0 = s \right].
\]

The **safety constraint** ensures that the expected cumulative cost remains below a threshold \( \tau \):
\[
V(s) \leq \tau \quad \forall s \in \mathcal{S}.
\]

#### **4. Lagrangian Multiplier**
To enforce the constraint, a **Lagrangian multiplier** \( \lambda \) is introduced. The augmented Lagrangian objective is:
\[
\mathcal{L}(\theta, \lambda) = \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right] - \lambda \cdot \max \left( 0, V(s) - \tau \right),
\]
where:
- \( \theta \) represents the policy parameters.
- \( \lambda \) is updated using gradient ascent:
  \[
  \lambda \leftarrow \lambda + \alpha_\lambda \cdot \max \left( 0, V(s) - \tau \right).
  \]

---

### **Algorithm Steps**

1. **Initialize**:
   - Policies \( \pi_i \) for each drone \( i \).
   - Centralized critic \( V(s) \).
   - Lagrangian multiplier \( \lambda = 1.0 \).

2. **Collect Trajectories**:
   - For each timestep \( t \), sample actions \( a_t^i \sim \pi_i(\cdot | s_t) \) for all drones.
   - Execute joint action \( a_t = (a_t^1, \dots, a_t^N) \) and observe \( s_{t+1} \), \( r_t \), and \( c_t \).

3. **Compute Advantages**:
   - Compute the advantage \( A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \).
   - Compute the cost advantage \( A_t^c = c_t + \gamma V(s_{t+1}) - V(s_t) \).

4. **Update Critic**:
   - Minimize the critic loss:
     \[
     \mathcal{L}_V(\phi) = \mathbb{E} \left[ \left( r_t + \gamma V(s_{t+1}) - V(s_t) \right)^2 \right].
     \]

5. **Update Policies**:
   - For each drone \( i \), update the policy \( \pi_i \) using the PPO objective:
     \[
     \mathcal{L}_{\pi_i}(\theta_i) = -\mathbb{E} \left[ \min \left( \frac{\pi_i(a_t^i | s_t)}{\pi_i^{\text{old}}(a_t^i | s_t)} A_t, \text{clip} \left( \frac{\pi_i(a_t^i | s_t)}{\pi_i^{\text{old}}(a_t^i | s_t)}, 1 - \epsilon, 1 + \epsilon \right) A_t \right) \right].
     \]
   - Add the safety constraint:
     \[
     \mathcal{L}_{\pi_i}(\theta_i) \leftarrow \mathcal{L}_{\pi_i}(\theta_i) + \lambda \cdot \max \left( 0, \mathbb{E} \left[ A_t^c \right] - \tau \right).
     \]

6. **Update Lagrangian Multiplier**:
   - Update \( \lambda \) using gradient ascent:
     \[
     \lambda \leftarrow \lambda + \alpha_\lambda \cdot \max \left( 0, \mathbb{E} \left[ A_t^c \right] - \tau \right).
     \]

7. **Repeat**:
   - Repeat steps 2–6 until convergence.

---

### **Example Scenario**
- **Number of Drones**: \( N = 5 \).
- **Number of Obstacles**: \( M = 3 \).
- **Safety Threshold**: \( \tau = 0.1 \) (expected cumulative cost should not exceed 0.1).
- **Minimum Safe Distance**: \( d_{\text{safe}}} = 2 \) meters.

#### **Training Outcome**
- The drones learn to navigate to the target while avoiding collisions with each other and obstacles.
- The expected cumulative cost of collisions is constrained to be below \( \tau = 0.1 \).

---

### **Key Takeaways**
- The **V-function constraints** ensure that the swarm of drones avoids collisions in expectation.
- The **Lagrangian multiplier** dynamically adjusts to enforce the safety constraint.
- This approach provides a **probabilistic safety guarantee** while optimizing for task performance.



---
Theorems:


