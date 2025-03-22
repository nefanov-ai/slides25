import compiler_gym
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit
from compiler_gym.wrappers import CompilerEnvWrapper
from compiler_gym.envs import CompilerEnv
import numpy as np
import random
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig  # Use PPO with Lagrangian constraints
from ray.tune.registry import register_env

class BaselineRuntimeWrapper(CompilerEnvWrapper):
    def __init__(self, env: LlvmEnv):
        super().__init__(env)
        self._baseline_runtime = None
        self._baseline_size = None

    def calc_baselines(self):
        self.env.reset()
        self._baseline_runtime = self.env.observation["Runtime"]
        self._baseline_size = self.env.observation["TextSizeBytes"]
        return self._baseline_runtime, self._baseline_size

    def reset(self, *args, **kwargs):
        _obs = super().reset(*args, **kwargs)
        self._baseline_runtime = self.env.observation["Runtime"]
        self._baseline_size = self.env.observation["TextSizeBytes"]
        return _obs

def penaltized_reward_function(env, baseline_runtime=0., runtime=0., penalty_factor=0.5) -> float:
    text_file_size_oz = env.observation["TextSizeOz"]
    text_file_size_bytes = env.observation["TextSizeBytes"]
    size_improvement = 1 - text_file_size_bytes / text_file_size_oz

    if runtime <= baseline_runtime:
        runtime_factor = 1.0
    else:
        runtime_factor = max(0, 1 - (runtime - baseline_runtime) / baseline_runtime)

    reward = size_improvement * runtime_factor

    if runtime > baseline_runtime:
        penalty = -penalty_factor * (runtime - baseline_runtime) / baseline_runtime
        reward += penalty

    return reward

def make_env(reward_func=penaltized_reward_function, max_episode_steps=200, benchmark="cbench-v1/bzip2", observation_space="Autophase"):
    env = compiler_gym.make("llvm-v0", benchmark=benchmark, observation_space=observation_space)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = BaselineRuntimeWrapper(env)

    original_step = env.step
    env.reset()

    def custom_step(action):
        observation, _, done, info = original_step(int(action))
        reward = reward_func(env, env._baseline_runtime[0], env.observation['Runtime'][0], 0.5)
        return observation, reward, done, info

    env.step = custom_step
    return env

# Define a custom environment for Ray
class CompilerGymEnv(CompilerEnv):
    def __init__(self, config):
        super().__init__()
        self.env = make_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def get_constraint(self):
        # Return the runtime as the constraint value
        return self.env.observation['Runtime'][0]

# Register the custom environment with Ray
register_env("CompilerGymEnv", lambda config: CompilerGymEnv(config))

# Configure PPO with Lagrangian constraints
config = (
    PPOConfig()  # Use PPO with Lagrangian constraints
    .environment("CompilerGymEnv")
    .framework("torch")  # Use PyTorch
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        model={"fcnet_hiddens": [128, 128]},
    )
    .constraints(
        constraints={
            "runtime_constraint": (0.0, 1.0)  # Constraint limits (adjust as needed)
        }
    )
)

# Train the agent
tune.run(
    "PPO",  # Use PPO with Lagrangian constraints
    config=config.to_dict(),
    stop={"timesteps_total": 100000},
    checkpoint_at_end=True,
)


# Evaluate the trained agent
env = make_env()
obs = env.reset()
done = False
total_reward = 0
total_cost = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    total_cost += info.get("cost", 0)

print(f"Total reward: {total_reward}")
print(f"Total cost: {total_cost}")

# Get the final optimized IR
optimized_ir = env.compiler_env.ir

# Print some statistics
print(f"Benchmark: {env.benchmark}")
print(f"Final bytes count: {env.compiler_env.observation['TextSizeBytes']}; \
      Profit: {(1 - env.compiler_env.observation['TextSizeBytes'] / env.compiler_env.observation['TextSizeOz']) * 100}%")

print(f"Final runtime: {env.compiler_env.observation['Runtime'][0]}; Profit: \
      {(1 - env.compiler_env.observation['Runtime'][0] / env._baseline_runtime[0]) * 100}%")

# Close the environment
env.close()
